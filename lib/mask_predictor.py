import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------
# Stage-0: original single-stage decoder (renamed to CoarseDecoder)
# ---------------------------------------------------------
class CoarseDecoder(nn.Module):
    """Single-stage decoder used as coarse mask predictor.

    This is the original SimpleDecoding implementation, kept unchanged and
    renamed to CoarseDecoder for use as the multi-stage Stage-0 decoder.
    """

    def __init__(self, c4_dims, factor=2):
        super(CoarseDecoder, self).__init__()

        hidden_size = c4_dims // factor
        c4_size = c4_dims
        c3_size = c4_dims // (factor ** 1)
        c2_size = c4_dims // (factor ** 2)
        c1_size = c4_dims // (factor ** 3)

        # stage 4
        self.conv1_4 = nn.Conv2d(c4_size + c3_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_4 = nn.BatchNorm2d(hidden_size)
        self.relu1_4 = nn.ReLU(inplace=True)
        self.conv2_4 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_4 = nn.BatchNorm2d(hidden_size)
        self.relu2_4 = nn.ReLU(inplace=True)

        # stage 3
        self.conv1_3 = nn.Conv2d(hidden_size + c2_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_3 = nn.BatchNorm2d(hidden_size)
        self.relu1_3 = nn.ReLU(inplace=True)
        self.conv2_3 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_3 = nn.BatchNorm2d(hidden_size)
        self.relu2_3 = nn.ReLU(inplace=True)

        # stage 2
        self.conv1_2 = nn.Conv2d(hidden_size + c1_size, hidden_size, 3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(hidden_size)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(hidden_size)
        self.relu2_2 = nn.ReLU(inplace=True)

        self.conv1_1 = nn.Conv2d(hidden_size, 2, 1)

    def forward(self, x_c4, x_c3, x_c2, x_c1):
        # fuse Y4 and Y3
        if x_c4.size(-2) < x_c3.size(-2) or x_c4.size(-1) < x_c3.size(-1):
            x_c4 = F.interpolate(
                input=x_c4,
                size=(x_c3.size(-2), x_c3.size(-1)),
                mode='bilinear',
                align_corners=True,
            )
        x = torch.cat([x_c4, x_c3], dim=1)
        x = self.conv1_4(x)
        x = self.bn1_4(x)
        x = self.relu1_4(x)
        x = self.conv2_4(x)
        x = self.bn2_4(x)
        x = self.relu2_4(x)

        # fuse top-down features and Y2 features
        if x.size(-2) < x_c2.size(-2) or x.size(-1) < x_c2.size(-1):
            x = F.interpolate(
                input=x,
                size=(x_c2.size(-2), x_c2.size(-1)),
                mode='bilinear',
                align_corners=True,
            )
        x = torch.cat([x, x_c2], dim=1)
        x = self.conv1_3(x)
        x = self.bn1_3(x)
        x = self.relu1_3(x)
        x = self.conv2_3(x)
        x = self.bn2_3(x)
        x = self.relu2_3(x)

        if x.size(-2) < x_c1.size(-2) or x.size(-1) < x_c1.size(-1):
            x = F.interpolate(
                input=x,
                size=(x_c1.size(-2), x_c1.size(-1)),
                mode='bilinear',
                align_corners=True,
            )
        x = torch.cat([x, x_c1], dim=1)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)

        return self.conv1_1(x)


class SimpleDecoding(CoarseDecoder):
    """Alias wrapper that keeps the original interface unchanged.

    Any existing code calling SimpleDecoding(c4_dims, factor) remains unchanged.
    """

    def __init__(self, c4_dims, factor=2):
        super(SimpleDecoding, self).__init__(c4_dims, factor=factor)


# ---------------------------------------------------------
#  Semantic Head: generate semantic features + semantic mask from the highest resolution features
# ---------------------------------------------------------
class SemanticHead(nn.Module):
    """Simple semantic head: input x_c1, output semantic_feat + semantic_mask.

    semantic_feat: provides high-resolution fine-grained features.
    semantic_mask: guides the PFR stages (probability map).
    """
    def __init__(self, in_channels, sem_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, sem_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(sem_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(sem_channels, sem_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(sem_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv_logits = nn.Conv2d(sem_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        semantic_feat = x
        semantic_mask = torch.sigmoid(self.conv_logits(x))
        return semantic_feat, semantic_mask


# ---------------------------------------------------------
#  Multi-stage PFR modules
# ---------------------------------------------------------
class SemanticFusionModule(nn.Module):
    """Semantic Fusion Module (SFM).

    Fuses the previous stage features + mask with semantic features + semantic mask.
    """

    def __init__(self, in_channels, sem_channels, out_channels):
        super().__init__()
        # feat_prev (C_f) + semantic_feat (C_s) + mask_prev(1) + semantic_mask(1)
        self.conv1 = nn.Conv2d(in_channels + sem_channels + 2, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # multi-scale dilated conv branches
        self.dilated_conv1 = nn.Conv2d(out_channels, out_channels, 3, padding=1, dilation=1, bias=False)
        self.dilated_conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=2, dilation=2, bias=False)
        self.dilated_conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=3, dilation=3, bias=False)

        self.merge = nn.Conv2d(out_channels * 3, out_channels, 1, bias=False)
        self.merge_bn = nn.BatchNorm2d(out_channels)
        self.merge_relu = nn.ReLU(inplace=True)

    def forward(self, feat_prev, mask_prev, semantic_feat, semantic_mask):
        # use only the foreground channel from the previous mask
        if mask_prev.size(1) > 1:
            mask_prev_fg = mask_prev[:, :1]
        else:
            mask_prev_fg = mask_prev
        x = torch.cat([feat_prev, semantic_feat, mask_prev_fg, semantic_mask], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        b1 = self.dilated_conv1(x)
        b2 = self.dilated_conv2(x)
        b3 = self.dilated_conv3(x)

        merged = torch.cat([b1, b2, b3], dim=1)
        merged = self.merge(merged)
        merged = self.merge_bn(merged)
        merged = self.merge_relu(merged)
        return merged


class PFRStage(nn.Module):
    """Single PFR stage: upsample + SFM + predict higher-resolution mask."""

    def __init__(self, feat_channels, sem_channels):
        super().__init__()
        self.sfm = SemanticFusionModule(
            in_channels=feat_channels,
            sem_channels=sem_channels,
            out_channels=feat_channels,
        )
        self.conv_pred = nn.Conv2d(feat_channels, 2, 1)

    def forward(self, feat_prev, mask_prev, semantic_feat, semantic_mask):
        H, W = semantic_feat.shape[-2], semantic_feat.shape[-1]

        if feat_prev.shape[-2:] != (H, W):
            feat_prev = F.interpolate(feat_prev, size=(H, W), mode='bilinear', align_corners=True)
        if mask_prev.shape[-2:] != (H, W):
            mask_prev = F.interpolate(mask_prev, size=(H, W), mode='bilinear', align_corners=True)

        fused = self.sfm(feat_prev, mask_prev, semantic_feat, semantic_mask)
        new_mask = self.conv_pred(fused)
        return fused, new_mask


class ProgressiveFeatureRecursionModule(nn.Module):
    """Progressive Feature Recursion Module (PFR).

    Interface aligned with the original SimpleDecoding:
        forward(x_c4, x_c3, x_c2, x_c1) returns the final PFR mask (2 channels) by default.
    For multi-stage supervision during training, call forward(..., return_all=True).
    """

    def __init__(self, c4_dims, factor=2, pfr_stages=2, pfr_channels=None):
        super().__init__()
        self.factor = factor
        self.c4_dims = c4_dims

        # Stage 0: coarse decoder (identical to the original SimpleDecoding)
        self.coarse_decoder = CoarseDecoder(c4_dims, factor=factor)

        # highest-resolution channel count (for x_c1)
        c1_size = c4_dims // (factor ** 3)

        # if pfr_channels is not specified, default to c1_size (same as original)
        if pfr_channels is None:
            pfr_channels = c1_size

        # semantic head: extract semantic_feat / semantic_mask from x_c1
        self.semantic_head = SemanticHead(in_channels=c1_size, sem_channels=pfr_channels)

        # PFR stage feature channels = pfr_channels (matches semantic_feat)
        feat_channels = pfr_channels
        self.pfr_stages = nn.ModuleList(
            [PFRStage(feat_channels, pfr_channels) for _ in range(pfr_stages)]
        )

    def forward(self, x_c4, x_c3, x_c2, x_c1, return_all=False):
        # semantic branch (high resolution)
        semantic_feat, semantic_mask = self.semantic_head(x_c1)

        # Stage 0: coarse mask (identical to original SimpleDecoding)
        coarse_mask = self.coarse_decoder(x_c4, x_c3, x_c2, x_c1)

        # initial PFR features: use semantic_feat
        feat = semantic_feat
        mask = coarse_mask

        pfr_masks = []  # store outputs for each PFR stage (exclude coarse)

        for stage in self.pfr_stages:
            feat, mask = stage(feat, mask, semantic_feat, semantic_mask)
            pfr_masks.append(mask)

        if len(pfr_masks) > 0:
            final_mask = pfr_masks[-1]
        else:
            final_mask = coarse_mask

        # return a multi-stage dict during training or when return_all=True
        if return_all or self.training:
            return {
                "coarse_mask": coarse_mask,
                "stage_masks": pfr_masks,
                "final_mask": final_mask,
            }
        else:
            # inference/eval returns only final logits, matching SimpleDecoding behavior
            return final_mask


# ---------------------------------------------------------
#  RFD-inspired feature decoupling and dynamic fusion
# ---------------------------------------------------------

class FeatureDecoupler(nn.Module):
    """
    Lightweight feature disentanglement module (ACFD-lite):
      - Input: low-resolution high-semantic features (e.g., x_c4), shape [B, C_in, H, W]
      - Outputs:
          base_feat   : base features (for residual/optional use)
          shared_feat : structure/domain-biased features (Shared-RS)
          private_feat: semantic/target-discriminative features (Private-Ref)
    Uses simple conv + attention without extra losses; provides inputs for MGDF.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Shared branch: conv + spatial attention
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.shared_attn = nn.Conv2d(out_channels, 1, kernel_size=1)

        # Private branch: conv + channel attention (SE-style)
        self.private_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.private_fc1 = nn.Conv2d(out_channels, out_channels // 4, kernel_size=1)
        self.private_fc2 = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1)

    def forward(self, x):
        # base_feat is currently passthrough for future residual use
        base_feat = x

        # shared branch with spatial attention
        s = self.shared_conv(x)
        attn_s = torch.sigmoid(self.shared_attn(s))  # [B,1,H,W]
        shared_feat = s * attn_s

        # private branch with channel attention
        p = self.private_conv(x)
        # global average pooling for channel attention
        gap = F.adaptive_avg_pool2d(p, 1)
        ch = self.private_fc1(gap)
        ch = F.relu(ch, inplace=True)
        ch = self.private_fc2(ch)
        ch = torch.sigmoid(ch)  # [B,C,1,1]
        private_feat = p * ch

        return base_feat, shared_feat, private_feat


class MGDFLite(nn.Module):
    """
    Simplified Matrix-Guided Dynamic Fusion:
      - Inputs: feat (previous stage), shared_feat, private_feat
      - Generates position-aware weights over three features and fuses them.
    """

    def __init__(self, feat_channels: int, shared_channels: int, private_channels: int):
        super().__init__()
        in_channels = feat_channels + shared_channels + private_channels

        # channel reduction
        self.reduce = nn.Conv2d(in_channels, feat_channels, kernel_size=1, bias=False)
        self.reduce_bn = nn.BatchNorm2d(feat_channels)
        self.reduce_relu = nn.ReLU(inplace=True)

        # generate 3-channel weight maps (base/shared/private)
        self.weight_conv = nn.Conv2d(feat_channels, 3, kernel_size=3, padding=1, bias=True)

        # extra residual mapping (optional)
        self.residual_conv = nn.Conv2d(feat_channels, feat_channels, kernel_size=1, bias=False)
        self.residual_bn = nn.BatchNorm2d(feat_channels)

    def forward(self, feat, shared_feat, private_feat):
        # ensure matching spatial sizes
        H, W = feat.shape[-2:]
        if shared_feat.shape[-2:] != (H, W):
            shared_feat = F.interpolate(shared_feat, size=(H, W), mode='bilinear', align_corners=True)
        if private_feat.shape[-2:] != (H, W):
            private_feat = F.interpolate(private_feat, size=(H, W), mode='bilinear', align_corners=True)

        x = torch.cat([feat, shared_feat, private_feat], dim=1)
        x = self.reduce(x)
        x = self.reduce_bn(x)
        x = self.reduce_relu(x)

        # generate position-aware 3-channel weights
        weight_logits = self.weight_conv(x)  # [B,3,H,W]
        weights = torch.softmax(weight_logits, dim=1)
        w_b, w_s, w_p = torch.chunk(weights, 3, dim=1)

        fused = w_b * feat + w_s * shared_feat + w_p * private_feat

        # lightweight residual
        res = self.residual_conv(x)
        res = self.residual_bn(res)
        fused = fused + res

        return fused


class ProgressiveFeatureRecursionModuleRFD(nn.Module):
    """
    Multi-stage PFR decoder enhanced with RFD:

      - Stage 0: use CoarseDecoder to get a coarse mask
      - Introduce FeatureDecoupler to get (shared_rs, private_ref) from x_c4
      - Before each PFR stage, use MGDFLite to fuse
            feat_prev, shared_rs, private_ref
        with dynamic weighting, then feed into PFRStage

    Interface matches ProgressiveFeatureRecursionModule:
        forward(x_c4, x_c3, x_c2, x_c1, return_all=False)
    """

    def __init__(self, c4_dims, factor=2, pfr_stages=2, pfr_channels=None):
        super().__init__()
        self.factor = factor
        self.c4_dims = c4_dims

        # Stage 0: coarse decoder
        self.coarse_decoder = CoarseDecoder(c4_dims, factor=factor)

        # highest-resolution channel count (for x_c1)
        c1_size = c4_dims // (factor ** 3)

        if pfr_channels is None:
            pfr_channels = c1_size

        # semantic head: extract semantic_feat / semantic_mask from x_c1
        self.semantic_head = SemanticHead(in_channels=c1_size, sem_channels=pfr_channels)

        # feature disentanglement: get shared/private from x_c4 (channels match pfr_channels)
        self.decoupler = FeatureDecoupler(in_channels=c4_dims, out_channels=pfr_channels)

        # MGDF-lite: dynamic fusion of feat_prev / shared / private before each PFR stage
        self.mgdf = MGDFLite(
            feat_channels=pfr_channels,
            shared_channels=pfr_channels,
            private_channels=pfr_channels,
        )

        # PFR stages
        feat_channels = pfr_channels
        self.pfr_stages = nn.ModuleList(
            [PFRStage(feat_channels, pfr_channels) for _ in range(pfr_stages)]
        )

    def forward(self, x_c4, x_c3, x_c2, x_c1, return_all=False):
        # semantic branch (high resolution)
        semantic_feat, semantic_mask = self.semantic_head(x_c1)

        # Stage 0: coarse mask
        coarse_mask = self.coarse_decoder(x_c4, x_c3, x_c2, x_c1)

        # feature disentanglement: get shared / private from x_c4
        base_low, shared_low, private_low = self.decoupler(x_c4)

        # upsample shared / private to semantic_feat resolution
        target_size = semantic_feat.shape[-2:]
        shared_feat = F.interpolate(shared_low, size=target_size, mode='bilinear', align_corners=True)
        private_feat = F.interpolate(private_low, size=target_size, mode='bilinear', align_corners=True)

        # initial PFR features: use semantic_feat; initial mask is coarse_mask
        feat = semantic_feat
        mask = coarse_mask

        pfr_masks = []

        for stage in self.pfr_stages:
        # RFD: dynamic fusion of feat / shared / private
            fused_feat = self.mgdf(feat, shared_feat, private_feat)
            # then apply the original PFRStage logic
            feat, mask = stage(fused_feat, mask, semantic_feat, semantic_mask)
            pfr_masks.append(mask)

        if len(pfr_masks) > 0:
            final_mask = pfr_masks[-1]
        else:
            final_mask = coarse_mask

        if return_all or self.training:
            return {
                "coarse_mask": coarse_mask,
                "stage_masks": pfr_masks,
                "final_mask": final_mask,
            }
        else:
            return final_mask
