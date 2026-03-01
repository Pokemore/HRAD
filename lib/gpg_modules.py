import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CoordConv(nn.Module):
    """CoordConv: Adds coordinate channels to input feature maps"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CoordConv, self).__init__()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, out_channels, H, W)
        """
        B, C, H, W = x.shape
        # Create coordinate channels
        y_coords = torch.arange(H, dtype=x.dtype, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        x_coords = torch.arange(W, dtype=x.dtype, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        # Normalize coordinates to [-1, 1]
        if H > 1:
            y_coords = (y_coords / (H - 1)) * 2 - 1
        else:
            y_coords = y_coords * 0
        if W > 1:
            x_coords = (x_coords / (W - 1)) * 2 - 1
        else:
            x_coords = x_coords * 0
        
        x_with_coords = torch.cat([x, x_coords, y_coords], dim=1)
        return self.conv(x_with_coords)


class MCC(nn.Module):
    """
    Multiscale CoordConv Module
    Extracts multi-receptive-field local features using CoordConv and multi-scale depthwise separable convolutions
    According to the paper formula 4: E_i^j = DWConv_j(Conv([E_i, coord_row, coord_col]))
    """
    def __init__(self, in_channels, out_channels=None, scales=[1, 3, 5]):
        super(MCC, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        
        self.scales = scales
        self.convs = nn.ModuleList()
        
        # Calculate channels per branch to ensure total equals out_channels
        num_branches = len(scales)
        channels_per_branch = out_channels // num_branches
        remainder = out_channels % num_branches
        
        # Multi-scale branches: Conv([E_i, coord]) + DWConv
        for i, scale in enumerate(scales):
            padding = scale // 2
            # Distribute remainder channels to first few branches
            branch_channels = channels_per_branch + (1 if i < remainder else 0)
            
            # First: CoordConv to concatenate coordinates and apply 1x1 conv
            coord_conv = CoordConv(in_channels, branch_channels, kernel_size=1, padding=0)
            
            # Then: Depthwise Separable Convolution (DWConv)
            # Depthwise convolution
            depthwise = nn.Conv2d(branch_channels, branch_channels, kernel_size=scale, 
                                  padding=padding, groups=branch_channels, bias=False)
            # Pointwise convolution
            pointwise = nn.Conv2d(branch_channels, branch_channels, kernel_size=1, bias=False)
            
            self.convs.append(
                nn.Sequential(
                    coord_conv,
                    depthwise,
                    pointwise,
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Fusion layer (CBR: Conv + BatchNorm + ReLU)
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - E_i from Swin Block
        Returns:
            out: (B, out_channels, H, W) - E_i^L (multi-scale local features)
        """
        multi_scale_features = []
        for conv in self.convs:
            multi_scale_features.append(conv(x))
        
        # Concatenate multi-scale features
        fused = torch.cat(multi_scale_features, dim=1)
        out = self.fusion(fused)
        return out


class CA(nn.Module):
    """
    Cross-modal Attention Module
    Captures language-aware global visual information E_i^g
    According to paper formulas 5-9:
    - Formula 5: L' = GELU(LN(L))
    - Formula 6: Q_Ei = CoordConv(LN(E_i)) W_qi (with InstanceNorm)
    - Formula 7-8: K_L' = LN(L') W_ki, V_L' = LN(L') W_vi
    - Formula 9: E_i^g = E_i + Conv(softmax((Q_Ei^T K_L') / sqrt(C_i)) V_L'^T)
    """
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, 
                 out_channels=None, num_heads=8, dropout=0.1):
        super(CA, self).__init__()
        if out_channels is None:
            out_channels = value_channels
            
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        head_dim = key_channels // num_heads
        self.scale = (head_dim) ** -0.5
        
        # Language preprocessing: Formula 5 - L' = GELU(LN(L))
        self.ln_l = nn.LayerNorm(l_in_channels)
        self.gelu = nn.GELU()
        
        # Visual path: CoordConv + LayerNorm + InstanceNorm + Query projection
        # Formula 6: Q_Ei = CoordConv(LN(E_i)) W_qi
        self.ln_v = nn.LayerNorm([v_in_channels])  # LayerNorm for spatial features
        self.coord_conv = CoordConv(v_in_channels, v_in_channels, kernel_size=1, padding=0)
        self.instance_norm = nn.InstanceNorm2d(v_in_channels)
        self.w_qi = nn.Conv2d(v_in_channels, key_channels, kernel_size=1)
        
        # Language projections: Formula 7-8
        # K_L' = LN(L') W_ki, V_L' = LN(L') W_vi
        self.w_ki = nn.Conv1d(l_in_channels, key_channels, kernel_size=1)
        self.w_vi = nn.Conv1d(l_in_channels, value_channels, kernel_size=1)
        
        # Output projection (CBR)
        self.out_proj = nn.Sequential(
            nn.Conv2d(value_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
    def forward(self, v_feat, l_feat, l_mask):
        """
        Args:
            v_feat: (B, v_in_channels, H, W) - E_i from Swin Block
            l_feat: (B, l_in_channels, N_l) - L (language features)
            l_mask: (B, N_l, 1) - language mask
        Returns:
            out: (B, out_channels, H, W) - E_i^g (language-aware global features)
        """
        B, C, H, W = v_feat.shape
        N_l = l_feat.shape[-1]
        
        # Formula 5: Language preprocessing L' = GELU(LN(L))
        # l_feat: (B, l_in_channels, N_l) -> (B, N_l, l_in_channels) for LN
        l_feat_transposed = l_feat.permute(0, 2, 1)  # (B, N_l, l_in_channels)
        l_prime = self.ln_l(l_feat_transposed)  # (B, N_l, l_in_channels)
        l_prime = self.gelu(l_prime)  # (B, N_l, l_in_channels)
        l_prime = l_prime.permute(0, 2, 1)  # (B, l_in_channels, N_l)
        
        # Formula 6: Query from visual features Q_Ei = CoordConv(LN(E_i)) W_qi
        # Apply LayerNorm: (B, C, H, W) -> (B, H, W, C) -> LN -> (B, C, H, W)
        v_feat_norm = v_feat.permute(0, 2, 3, 1)  # (B, H, W, C)
        v_feat_norm = self.ln_v(v_feat_norm)  # (B, H, W, C)
        v_feat_norm = v_feat_norm.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # CoordConv + InstanceNorm + Query projection
        v_coord = self.coord_conv(v_feat_norm)  # (B, C, H, W)
        v_coord = self.instance_norm(v_coord)  # (B, C, H, W)
        q_ei = self.w_qi(v_coord)  # (B, key_channels, H, W)
        
        # Reshape for attention: (B, key_channels, H, W) -> (B, H*W, key_channels)
        q_ei = q_ei.view(B, self.key_channels, H * W).permute(0, 2, 1)  # (B, H*W, key_channels)
        
        # Formula 7-8: Key and Value from language features
        l_mask_expanded = l_mask.permute(0, 2, 1)  # (B, 1, N_l)
        k_l_prime = self.w_ki(l_prime) * l_mask_expanded  # (B, key_channels, N_l)
        v_l_prime = self.w_vi(l_prime) * l_mask_expanded  # (B, value_channels, N_l)
        
        # Reshape for multi-head attention
        q_ei = q_ei.view(B, H * W, self.num_heads, self.key_channels // self.num_heads)
        q_ei = q_ei.permute(0, 2, 1, 3)  # (B, num_heads, H*W, head_dim)
        
        k_l_prime = k_l_prime.view(B, self.num_heads, self.key_channels // self.num_heads, N_l)
        v_l_prime = v_l_prime.view(B, self.num_heads, self.value_channels // self.num_heads, N_l)
        
        # Formula 9: Cross-attention computation
        # Attention: softmax((Q_Ei^T K_L') / sqrt(C_i)) V_L'^T
        attn = torch.matmul(q_ei, k_l_prime) * self.scale  # (B, num_heads, H*W, N_l)
        
        # Apply mask: set padding positions to large negative value
        mask_expanded = l_mask_expanded.unsqueeze(1)  # (B, 1, 1, N_l)
        attn = attn.masked_fill(mask_expanded == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)  # (B, num_heads, H*W, N_l)
        
        # Apply attention to values
        v_enhanced = torch.matmul(attn, v_l_prime.permute(0, 1, 3, 2))  # (B, num_heads, H*W, head_dim)
        
        # Reshape back: (B, num_heads, H*W, head_dim) -> (B, value_channels, H, W)
        v_enhanced = v_enhanced.permute(0, 2, 1, 3).contiguous()
        v_enhanced = v_enhanced.view(B, H * W, self.value_channels)
        v_enhanced = v_enhanced.permute(0, 2, 1).view(B, self.value_channels, H, W)
        
        # Formula 9: Residual connection E_i^g = E_i + Conv(...)
        out = self.out_proj(v_enhanced)  # (B, out_channels, H, W)
        out = v_feat + out  # Residual connection with original E_i
        
        return out


class SA(nn.Module):
    """
    Sampling Attention Module
    Enables fine-grained vision context alignment under semantic guidance
    According to paper formulas 11-15:
    - Formula 11: Q_E^f = LN(E_i^f) w_qi
    - Formula 12: K_E^g = MaxPooling(LN(E_i^g) w_ki) (window size 16)
    - Formula 13: V_E^g = AvgPooling(LN(E_i^g) w_vi) (window size 16)
    - Formula 14: A_i = softmax((Q_E^f K_E^g^T) / sqrt(C_i)) V_E^g
    - Formula 15: E_i^a = E_i^f ⊙ w_oi (DWConv(Q_E^f) + A_i)
    """
    def __init__(self, v_channels, l_channels, hidden_dim=256, dropout=0.1, pool_size=16):
        super(SA, self).__init__()
        self.v_channels = v_channels  # This is dim*2 (concatenated E_i^f)
        self.l_channels = l_channels
        self.hidden_dim = hidden_dim
        self.out_channels = v_channels // 2  # Output should be same as single feature channel
        self.pool_size = pool_size  # Pooling window size (16 according to paper)
        
        # Formula 11: Query projection from E_i^f
        self.ln_e_f = nn.LayerNorm([v_channels])
        self.w_qi = nn.Linear(v_channels, hidden_dim)
        
        # Formula 12-13: Key and Value projections from E_i^g
        self.ln_e_g = nn.LayerNorm([v_channels // 2])
        self.w_ki = nn.Linear(v_channels // 2, hidden_dim)
        self.w_vi = nn.Linear(v_channels // 2, hidden_dim)
        
        # Formula 15: Output projection and Depthwise Convolution
        self.w_oi = nn.Linear(hidden_dim, hidden_dim)
        self.dw_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, 
                                 groups=hidden_dim, bias=False)  # Depthwise conv
        
        # Projection for E_i^f to match hidden_dim for element-wise multiplication
        self.e_f_proj = nn.Conv2d(v_channels, hidden_dim, kernel_size=1)
        
        # Output projection (CBR)
        self.out_proj = nn.Sequential(
            nn.Conv2d(hidden_dim, self.out_channels, kernel_size=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
        
        self.scale = (hidden_dim) ** -0.5
        
    def forward(self, e_f, e_g, l_feat, l_mask):
        """
        Args:
            e_f: (B, v_channels, H, W) - E_i^f (concatenated E_i^L and E_i^g, dim*2)
            e_g: (B, v_channels//2, H, W) - E_i^g (language-aware global features, dim)
            l_feat: (B, l_channels, N_l) - L (language features) - not used in SA
            l_mask: (B, N_l, 1) - language mask - not used in SA
        Returns:
            out: (B, out_channels, H, W) - E_i^a (aligned features, dim)
        """
        B, C_f, H, W = e_f.shape
        B, C_g, H_g, W_g = e_g.shape
        
        # Formula 11: Q_E^f = LN(E_i^f) w_qi
        # Reshape: (B, C_f, H, W) -> (B, H, W, C_f) -> LN -> (B, H*W, C_f)
        e_f_reshaped = e_f.permute(0, 2, 3, 1)  # (B, H, W, C_f)
        e_f_norm = self.ln_e_f(e_f_reshaped)  # (B, H, W, C_f)
        e_f_norm = e_f_norm.view(B, H * W, C_f)  # (B, H*W, C_f)
        q_e_f = self.w_qi(e_f_norm)  # (B, H*W, hidden_dim)
        
        # Formula 12: K_E^g = MaxPooling(LN(E_i^g) w_ki)
        # Reshape: (B, C_g, H_g, W_g) -> (B, H_g, W_g, C_g) -> LN
        e_g_reshaped = e_g.permute(0, 2, 3, 1)  # (B, H_g, W_g, C_g)
        e_g_norm = self.ln_e_g(e_g_reshaped)  # (B, H_g, W_g, C_g)
        e_g_norm = e_g_norm.view(B, H_g * W_g, C_g)  # (B, H_g*W_g, C_g)
        k_e_g_proj = self.w_ki(e_g_norm)  # (B, H_g*W_g, hidden_dim)
        
        # Reshape back and apply MaxPooling
        k_e_g_proj = k_e_g_proj.permute(0, 2, 1).view(B, self.hidden_dim, H_g, W_g)  # (B, hidden_dim, H_g, W_g)
        # Dynamically adjust pooling window to avoid erasing small feature maps
        pool_k = max(1, min(self.pool_size, H_g, W_g))
        k_e_g = F.max_pool2d(k_e_g_proj, kernel_size=pool_k, stride=pool_k)  # (B, hidden_dim, H_p, W_p)
        H_p, W_p = k_e_g.shape[2], k_e_g.shape[3]
        k_e_g = k_e_g.view(B, self.hidden_dim, H_p * W_p).permute(0, 2, 1)  # (B, H_p*W_p, hidden_dim)
        
        # Formula 13: V_E^g = AvgPooling(LN(E_i^g) w_vi)
        v_e_g_proj = self.w_vi(e_g_norm)  # (B, H_g*W_g, hidden_dim)
        v_e_g_proj = v_e_g_proj.permute(0, 2, 1).view(B, self.hidden_dim, H_g, W_g)  # (B, hidden_dim, H_g, W_g)
        v_e_g = F.avg_pool2d(v_e_g_proj, kernel_size=pool_k, stride=pool_k)  # (B, hidden_dim, H_p, W_p)
        v_e_g = v_e_g.view(B, self.hidden_dim, H_p * W_p).permute(0, 2, 1)  # (B, H_p*W_p, hidden_dim)
        
        # Formula 14: A_i = softmax((Q_E^f K_E^g^T) / sqrt(C_i)) V_E^g
        attn = torch.matmul(q_e_f, k_e_g.transpose(-2, -1)) * self.scale  # (B, H*W, H_p*W_p)
        attn = F.softmax(attn, dim=-1)  # (B, H*W, H_p*W_p)
        a_i = torch.matmul(attn, v_e_g)  # (B, H*W, hidden_dim)
        
        # Formula 15: E_i^a = E_i^f ⊙ w_oi (DWConv(Q_E^f) + A_i)
        # Reshape Q_E^f for DWConv: (B, H*W, hidden_dim) -> (B, hidden_dim, H, W)
        q_e_f_conv = q_e_f.permute(0, 2, 1).view(B, self.hidden_dim, H, W)  # (B, hidden_dim, H, W)
        dw_conv_q = self.dw_conv(q_e_f_conv)  # (B, hidden_dim, H, W)
        
        # Reshape A_i: (B, H*W, hidden_dim) -> (B, hidden_dim, H, W)
        a_i_conv = a_i.permute(0, 2, 1).view(B, self.hidden_dim, H, W)  # (B, hidden_dim, H, W)
        
        # Add: DWConv(Q_E^f) + A_i
        combined = dw_conv_q + a_i_conv  # (B, hidden_dim, H, W)
        
        # Apply w_oi projection: (B, hidden_dim, H, W) -> (B, H*W, hidden_dim) -> Linear -> (B, H*W, hidden_dim)
        combined_reshaped = combined.view(B, self.hidden_dim, H * W).permute(0, 2, 1)  # (B, H*W, hidden_dim)
        combined_proj = self.w_oi(combined_reshaped)  # (B, H*W, hidden_dim)
        combined_proj = combined_proj.permute(0, 2, 1).view(B, self.hidden_dim, H, W)  # (B, hidden_dim, H, W)
        
        # Element-wise multiplication with E_i^f
        # Formula 15: E_i^a = E_i^f ⊙ w_oi (DWConv(Q_E^f) + A_i)
        # Project E_i^f to hidden_dim for element-wise multiplication
        e_f_proj = self.e_f_proj(e_f)  # (B, hidden_dim, H, W)
        
        # Element-wise multiplication: E_i^f ⊙ w_oi(...)
        e_i_a = e_f_proj * combined_proj  # (B, hidden_dim, H, W)
        
        # Output projection to out_channels (dim)
        out = self.out_proj(e_i_a)  # (B, out_channels, H, W)
        
        return out


class LanguageGate(nn.Module):
    """
    Language Gate Module
    Formula 17: U_i = E_i + F_i ⊙ Linear(ReLU(Linear(Tanh(F_i))))
    Note: The E_i residual is added in the backbone forward, here we compute the gate
    """
    def __init__(self, dim, l_channels=768):
        super(LanguageGate, self).__init__()
        self.dim = dim
        
        # Formula 17: Gate = Linear(ReLU(Linear(Tanh(F_i))))
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim)
        )
        
    def forward(self, f_i, l_feat, l_mask):
        """
        Args:
            f_i: (B, H*W, dim) - F_i features
            l_feat: (B, l_channels, N_l) - language features (not used in formula 17)
            l_mask: (B, N_l, 1) - language mask (not used in formula 17)
        Returns:
            out: (B, H*W, dim) - gated F_i (to be added to E_i in backbone)
        """
        # Formula 17: F_i ⊙ Linear(ReLU(Linear(Tanh(F_i))))
        gate = self.gate(f_i)  # (B, H*W, dim)
        out = f_i * gate  # Element-wise multiplication
        return out


class FrequencyFusion(nn.Module):
    """
    Improved Frequency Fusion Module for decoder
    Formula 21: F_i+1^up = Frequency Fusion(Fi, F_i+1)
    Performs 2x upsampling using frequency domain fusion with low/high frequency separation
    
    Improvements over simple addition:
    - Uses rfft2 for computational efficiency (only processes half frequency domain)
    - Separates low-frequency and high-frequency components
    - Learnable weights for adaptive low/high frequency fusion
    - More selective fusion compared to simple addition
    """
    def __init__(self, in_channels1, in_channels2, out_channels, low_freq_ratio=0.2):
        """
        Args:
            in_channels1: Input channels for feat1
            in_channels2: Input channels for feat2
            out_channels: Output channels
            low_freq_ratio: Ratio to determine low-frequency threshold (default: 0.3)
                           Low frequencies: [0, low_freq_ratio * W]
                           High frequencies: [low_freq_ratio * W, W//2]
        """
        super(FrequencyFusion, self).__init__()
        self.in_channels1 = in_channels1
        self.in_channels2 = in_channels2
        self.out_channels = out_channels
        self.low_freq_ratio = low_freq_ratio
        
        # Projection layers
        self.proj1 = nn.Conv2d(in_channels1, out_channels, kernel_size=1)
        self.proj2 = nn.Conv2d(in_channels2, out_channels, kernel_size=1)
        
        # Learnable weights for low-frequency and high-frequency fusion
        # Using channel-wise weights for better flexibility
        self.low_weight = nn.Parameter(torch.ones(1, out_channels, 1, 1))
        self.high_weight = nn.Parameter(torch.ones(1, out_channels, 1, 1))
        
        # Optional: learnable bias for frequency fusion
        self.freq_bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        
        # Post-fusion refinement
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def _create_frequency_mask(self, H, W_freq, device):
        """
        Create low-frequency and high-frequency masks for rfft2 output
        rfft2 output shape: (B, C, H, W_freq) where W_freq = W//2 + 1
        
        Args:
            H: Height
            W_freq: Frequency width (W//2 + 1)
            device: Device for the mask
        
        Returns:
            low_mask: (1, 1, H, W_freq) - mask for low frequencies
            high_mask: (1, 1, H, W_freq) - mask for high frequencies
        """
        # Create coordinate grids
        h_coords = torch.arange(H, device=device).float()
        w_coords = torch.arange(W_freq, device=device).float()
        
        # Normalize coordinates to [0, 1]
        h_norm = h_coords / max(H - 1, 1)
        w_norm = w_coords / max(W_freq - 1, 1)
        
        # Create 2D grid
        # Note: indexing='ij' is only available in PyTorch 1.10+, for compatibility use default
        # In older versions, meshgrid returns (w_grid, h_grid), so we need to transpose
        try:
            # Try with indexing parameter (PyTorch 1.10+)
            h_grid, w_grid = torch.meshgrid(h_norm, w_norm, indexing='ij')
        except TypeError:
            # Fallback for older PyTorch versions
            w_grid, h_grid = torch.meshgrid(w_norm, h_norm)
            # Transpose to get correct order
            h_grid = h_grid.t()
            w_grid = w_grid.t()
        
        # Compute frequency distance from DC component (0, 0)
        # Low frequencies are near (0, 0), high frequencies are far from (0, 0)
        freq_distance = torch.sqrt(h_grid ** 2 + w_grid ** 2)
        
        # Threshold based on low_freq_ratio
        threshold = self.low_freq_ratio
        
        # Low frequency mask: frequencies near DC component
        low_mask = (freq_distance <= threshold).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W_freq)
        
        # High frequency mask: frequencies far from DC component
        high_mask = (freq_distance > threshold).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W_freq)
        
        return low_mask, high_mask
        
    def forward(self, feat1, feat2):
        """
        Args:
            feat1: (B, in_channels1, H1, W1) - F_i from encoder
            feat2: (B, in_channels2, H2, W2) - F_{i+1}' from previous decoder stage
        Returns:
            out: (B, out_channels, H1*2, W1*2) - upsampled fused features (2x upsampling)
        """
        B, C1, H1, W1 = feat1.shape
        B, C2, H2, W2 = feat2.shape
        
        # Project features
        feat1_proj = self.proj1(feat1)  # (B, out_channels, H1, W1)
        feat2_proj = self.proj2(feat2)  # (B, out_channels, H2, W2)
        
        # Upsample feat2 to match feat1's spatial size first
        if H2 != H1 or W2 != W1:
            feat2_proj = F.interpolate(feat2_proj, size=(H1, W1), mode='bilinear', align_corners=True)
        
        # Frequency domain transformation using rfft2 (computationally efficient)
        feat1_fft = torch.fft.rfft2(feat1_proj, norm='ortho')  # (B, out_channels, H1, W1//2+1)
        feat2_fft = torch.fft.rfft2(feat2_proj, norm='ortho')  # (B, out_channels, H1, W1//2+1)
        
        B, C, H_freq, W_freq = feat1_fft.shape
        
        # Create frequency masks
        low_mask, high_mask = self._create_frequency_mask(H_freq, W_freq, feat1_fft.device)
        
        # Separate low-frequency and high-frequency components
        feat1_low = feat1_fft * low_mask  # (B, out_channels, H_freq, W_freq)
        feat1_high = feat1_fft * high_mask
        feat2_low = feat2_fft * low_mask
        feat2_high = feat2_fft * high_mask
        
        # Fuse low-frequency components (global structure information)
        fused_low = feat1_low + feat2_low  # Simple addition for low frequencies
        
        # Fuse high-frequency components (detail texture information)
        fused_high = feat1_high + feat2_high  # Simple addition for high frequencies
        
        # Apply learnable weights to low and high frequency components
        # Low frequencies: global structure, usually more important
        # High frequencies: details, may contain noise
        weighted_low = fused_low * self.low_weight
        weighted_high = fused_high * self.high_weight
        
        # Combine weighted low and high frequency components
        fused_fft = weighted_low + weighted_high + self.freq_bias
        
        # Convert back to spatial domain with 2x upsampling
        # Target size: (H1*2, W1*2)
        target_size = (H1 * 2, W1 * 2)
        fused = torch.fft.irfft2(fused_fft, s=target_size, norm='ortho')  # (B, out_channels, H1*2, W1*2)
        
        # Upsample feat1_proj to match fused size
        feat1_upsampled = F.interpolate(feat1_proj, size=target_size, mode='bilinear', align_corners=True)
        
        # Concatenate and fuse (combine spatial and frequency domain information)
        concat = torch.cat([feat1_upsampled, fused], dim=1)  # (B, out_channels*2, H1*2, W1*2)
        out = self.fusion(concat)  # (B, out_channels, H1*2, W1*2)
        
        return out


class GLF(nn.Module):
    """
    Global Language Fusion Module
    Maintains long-term vision-language alignment in the decoder
    According to the paper, GLF is used in decoder stages
    Stage 1 uses Tanh version (formulas 18-20), Stage 2-4 use standard version
    """
    def __init__(self, v_channels, l_channels, out_channels=None, dropout=0.1, use_tanh=False):
        super(GLF, self).__init__()
        if out_channels is None:
            out_channels = v_channels
            
        self.v_channels = v_channels
        self.l_channels = l_channels
        self.out_channels = out_channels
        self.use_tanh = use_tanh
        
        if use_tanh:
            # Stage 1: Formula 18-20
            # Z4 = Tanh(Conv1×1(F4))
            self.v_proj = nn.Conv2d(v_channels, out_channels, kernel_size=1)
            # Lg = Avg(Tanh(Linear(L)))
            self.l_proj = nn.Linear(l_channels, out_channels)
            # Projection from v_channels to out_channels for residual connection in Formula 20
            self.v_proj_back = nn.Conv2d(v_channels, out_channels, kernel_size=1)
        else:
            # Stage 2-4: Standard GLF
            # Visual projection
            self.v_proj = nn.Sequential(
                nn.Conv2d(v_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            
            # Language projection and aggregation
            self.l_proj = nn.Sequential(
                nn.Conv1d(l_channels, out_channels, kernel_size=1),
                nn.ReLU(inplace=True)
            )
            
            # Fusion (CBR)
            self.fusion = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            )
        
    def forward(self, v_feat, l_feat, l_mask):
        """
        Args:
            v_feat: (B, v_channels, H, W) - visual features from decoder
            l_feat: (B, l_channels, N_l) - language features
            l_mask: (B, N_l, 1) - language mask
        Returns:
            out: (B, out_channels, H, W) - language-fused visual features
        """
        B, C, H, W = v_feat.shape
        
        if self.use_tanh:
            # Stage 1: Formula 18-20
            # Formula 18: Z4 = Tanh(Conv1×1(F4))
            z4 = torch.tanh(self.v_proj(v_feat))  # (B, out_channels, H, W)
            
            # Formula 19: Lg = Avg(Tanh(Linear(L)))
            # l_feat: (B, l_channels, N_l) -> (B, N_l, l_channels) for Linear
            l_feat_transposed = l_feat.permute(0, 2, 1)  # (B, N_l, l_channels)
            l_linear = self.l_proj(l_feat_transposed)  # (B, N_l, out_channels)
            l_tanh = torch.tanh(l_linear)  # (B, N_l, out_channels)
            
            # Apply mask and average
            l_mask_expanded = l_mask.permute(0, 2, 1)  # (B, 1, N_l)
            l_tanh_masked = l_tanh.permute(0, 2, 1) * l_mask_expanded  # (B, out_channels, N_l)
            l_mask_sum = l_mask.sum(dim=1)  # (B, 1) or (B,)
            # Ensure l_mask_sum is (B, 1) for broadcasting
            if l_mask_sum.dim() == 1:
                l_mask_sum = l_mask_sum.unsqueeze(-1)  # (B,) -> (B, 1)
            # l_mask_sum is now (B, 1), reshape to (B, 1, 1) for broadcasting with (B, out_channels, 1)
            l_mask_sum = l_mask_sum.view(B, 1, 1)  # (B, 1) -> (B, 1, 1)
            l_g = l_tanh_masked.sum(dim=-1, keepdim=True) / (l_mask_sum + 1e-8)  # (B, out_channels, 1)
            # Ensure l_g is 3D: (B, out_channels, 1)
            # Remove extra dimensions if present
            while l_g.dim() > 3:
                l_g = l_g.squeeze(-1)
            # Expand to spatial dimensions: (B, out_channels, 1) -> (B, out_channels, 1, 1) -> (B, out_channels, H, W)
            l_g = l_g.unsqueeze(-1)  # (B, out_channels, 1) -> (B, out_channels, 1, 1)
            l_g = l_g.expand(-1, -1, H, W)  # (B, out_channels, 1, 1) -> (B, out_channels, H, W)
            
            # Formula 20: F_4' = F4 + Z4 ⊙ Lg
            # Project v_feat to out_channels, then add z4 * l_g
            z4_lg = z4 * l_g  # (B, out_channels, H, W)
            v_feat_proj = self.v_proj_back(v_feat)  # (B, v_channels, H, W) -> (B, out_channels, H, W)
            out = v_feat_proj + z4_lg  # Residual connection: (B, out_channels, H, W)
        else:
            # Stage 2-4: Standard GLF
            # Project visual features
            v_proj = self.v_proj(v_feat)  # (B, out_channels, H, W)
            
            # Aggregate language features globally (Average Operation)
            l_mask_expanded = l_mask.permute(0, 2, 1)  # (B, 1, N_l)
            l_proj = self.l_proj(l_feat) * l_mask_expanded  # (B, out_channels, N_l)
            # Sum over language tokens: (B, out_channels, N_l) -> (B, out_channels, 1)
            l_proj_sum = l_proj.sum(dim=-1, keepdim=True)  # (B, out_channels, 1)
            # Normalize by valid token count: (B, N_l, 1) -> (B, 1) -> (B, 1, 1)
            l_mask_sum = l_mask.sum(dim=1)  # (B, 1)
            # Ensure proper shape for broadcasting: (B, out_channels, 1) / (B, 1, 1) -> (B, out_channels, 1)
            if l_mask_sum.dim() == 2:
                l_mask_sum = l_mask_sum.unsqueeze(-1)  # (B, 1) -> (B, 1, 1)
            elif l_mask_sum.dim() == 1:
                l_mask_sum = l_mask_sum.view(B, 1, 1)  # (B,) -> (B, 1, 1)
            # Average: (B, out_channels, 1) / (B, 1, 1) -> (B, out_channels, 1)
            l_global = l_proj_sum / (l_mask_sum + 1e-8)  # (B, out_channels, 1)
            # Ensure l_global is 3D: (B, out_channels, 1)
            # Remove extra dimensions if present
            while l_global.dim() > 3:
                l_global = l_global.squeeze(-1)
            # Reshape to (B, out_channels, H, W): (B, out_channels, 1) -> (B, out_channels, 1, 1) -> (B, out_channels, H, W)
            l_global = l_global.unsqueeze(-1)  # (B, out_channels, 1) -> (B, out_channels, 1, 1)
            l_global = l_global.expand(-1, -1, H, W)  # (B, out_channels, H, W)
            
            # Dot multiplication and add (according to the architecture diagram)
            fused = v_proj * l_global + v_proj  # (B, out_channels, H, W)
            
            # Final fusion (CBR)
            out = self.fusion(fused)
            
            # Residual connection
            out = out + v_proj
        
        return out
