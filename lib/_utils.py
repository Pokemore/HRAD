from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from bert.modeling_bert import BertModel


# ============================================
# GPG: backbone and decoder provided externally (SimpleDecoding)
# Output a single [B, 2, H, W] logits tensor
# ============================================
class _GPGSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier):
        super(_GPGSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x, l_feats, l_mask, t_feats=None, t_mask=None, p_feats=None, p_mask=None):
        """
        x:       [B, 3, H, W]
        l_feats: [B, C_l, N_l]  (already conv1d-friendly text features)
        l_mask:  [B, N_l, 1]
        """
        input_shape = x.shape[-2:]                         # (H, W)
        if t_feats is not None and p_feats is not None and t_mask is not None and p_mask is not None:
            features = self.backbone(x, l_feats, l_mask, t_feats, t_mask, p_feats, p_mask)
        else:
            features = self.backbone(x, l_feats, l_mask)       # (x_c1, x_c2, x_c3, x_c4)
        x_c1, x_c2, x_c3, x_c4 = features

        x = self.classifier(x_c4, x_c3, x_c2, x_c1)        # [B, 2, H', W']
        x = F.interpolate(x, size=input_shape,
                          mode='bilinear', align_corners=True)
        return x


class GPG(_GPGSimpleDecode):
    """GPG base version (keeps the original interface)."""
    pass


# ============================================
# GPGOne: BERT inside the model
# ============================================
class _GPGOneSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier, args):
        super(_GPGOneSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.text_encoder = BertModel.from_pretrained(args.ck_bert)
        self.text_encoder.pooler = None    # workaround for a DDP bug

    def forward(self, x, text, l_mask, t_mask=None, p_mask=None):
        """
        x:    [B, 3, H, W]
        text: token ids  [B, N_l]
        l_mask: attention mask [B, N_l]
        """
        input_shape = x.shape[-2:]

        # 1) text encoding
        l_feats = self.text_encoder(text, attention_mask=l_mask)[0]  # [B, N_l, C]
        l_feats = l_feats.permute(0, 2, 1)                           # [B, C, N_l]
        l_mask = l_mask.unsqueeze(dim=-1)                            # [B, N_l, 1]

        # 2) fine-grained vision-language features (optional)
        if t_mask is not None and p_mask is not None:
            t_feats = self.text_encoder(text, attention_mask=t_mask)[0]
            t_feats = t_feats.permute(0, 2, 1)
            t_mask = t_mask.unsqueeze(dim=-1)

            p_feats = self.text_encoder(text, attention_mask=p_mask)[0]
            p_feats = p_feats.permute(0, 2, 1)
            p_mask = p_mask.unsqueeze(dim=-1)

            features = self.backbone(x, l_feats, l_mask, t_feats, t_mask, p_feats, p_mask)
        else:
            # 2) vision-language backbone
            features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features

        # 3) decode + upsample
        x = self.classifier(x_c4, x_c3, x_c2, x_c1)
        x = F.interpolate(x, size=input_shape,
                          mode='bilinear', align_corners=True)
        return x


class GPGOne(_GPGOneSimpleDecode):
    """GPGOne version (keeps the original interface)."""
    pass


# ============================================
# GPGPFR: multi-stage PFR decoder (Progressive Feature Recursion Module)
# classifier returns dict: {coarse_mask, stage_masks, final_mask}
# Responsibilities:
#   1) extract multi-scale features from the backbone
#   2) run the PFR decoder
#   3) resize all masks back to input resolution
# ============================================
class _GPGPFRSimpleDecode(nn.Module):
    def __init__(self, backbone, classifier):
        """
        backbone: multi-modal Swin backbone
        classifier: Progressive Feature Recursion Module (in mask_predictor.py)
        """
        super(_GPGPFRSimpleDecode, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def _resize_to_input(self, tensor_or_list, size_hw):
        """Resize mask / mask list to input resolution (H, W)."""
        if tensor_or_list is None:
            return None

        if isinstance(tensor_or_list, torch.Tensor):
            return F.interpolate(tensor_or_list,
                                 size=size_hw,
                                 mode='bilinear',
                                 align_corners=True)
        elif isinstance(tensor_or_list, (list, tuple)):
            out_list = []
            for t in tensor_or_list:
                if t is None:
                    out_list.append(None)
                else:
                    out_list.append(
                        F.interpolate(t,
                                      size=size_hw,
                                      mode='bilinear',
                                      align_corners=True)
                    )
            return out_list
        else:
            return tensor_or_list

    def forward(self, x, l_feats, l_mask, t_feats=None, t_mask=None, p_feats=None, p_mask=None):
        """
        x:       [B, 3, H, W]
        l_feats: [B, C_l, N_l]
        l_mask:  [B, N_l, 1]
        """
        input_shape = x.shape[-2:]                         # (H, W)

        # 1) backbone multi-scale features
        if t_feats is not None and p_feats is not None and t_mask is not None and p_mask is not None:
            features = self.backbone(x, l_feats, l_mask, t_feats, t_mask, p_feats, p_mask)
        else:
            features = self.backbone(x, l_feats, l_mask)
        x_c1, x_c2, x_c3, x_c4 = features

        # 2) run multi-stage PFR decoder
        out = self.classifier(x_c4, x_c3, x_c2, x_c1)

        # 3) if dict (multi-stage), resize all masks to input size
        if isinstance(out, dict):
            coarse = out.get("coarse_mask", None)
            stage_masks = out.get("stage_masks", None)
            final_mask = out.get("final_mask", None)

            coarse_r = self._resize_to_input(coarse, input_shape)
            final_r = self._resize_to_input(final_mask, input_shape)
            stage_r = self._resize_to_input(stage_masks, input_shape)

            return {
                "coarse_mask": coarse_r,
                "stage_masks": stage_r,
                "final_mask": final_r,
            }

        # 4) otherwise fall back to single-stage output (defensive)
        x = self._resize_to_input(out, input_shape)
        return x


class GPGPFR(_GPGPFRSimpleDecode):
    """Wrapper for --model gpg_pfr."""
    pass
