import torch
import torch.nn as nn

from .mask_predictor import SimpleDecoding, ProgressiveFeatureRecursionModule, ProgressiveFeatureRecursionModuleRFD
from .gpg_backbone import MultiModalSwinTransformer as GPGBackbone
from ._utils import GPG, GPGOne, GPGPFR

__all__ = ['gpg', 'gpg_one', 'gpg_pfr', 'gpg_pfr_rfd']


# ------------------------------------------------
# GPG base version
# ------------------------------------------------
def _segm_gpg(pretrained, args):
    # Initialize SwinTransformer backbone for the specified variant
    if args.swin_type == 'tiny':
        embed_dim = 96
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'small':
        embed_dim = 96
        depths = [2, 2, 18, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'base':
        embed_dim = 128
        depths = [2, 2, 18, 2]
        num_heads = [4, 8, 16, 32]
    elif args.swin_type == 'large':
        embed_dim = 192
        depths = [2, 2, 18, 2]
        num_heads = [6, 12, 24, 48]
    else:
        assert False

    # test.py needs window12 because weights are loaded after init
    if ('window12' in pretrained) or getattr(args, 'window12', False):
        print('Window size 12!')
        window_size = 12
    else:
        window_size = 7

    if getattr(args, 'mha', None):
        mha = args.mha.split('-')  # if non-empty, format is ['a', 'b', 'c', 'd']
        mha = [int(a) for a in mha]
    else:
        mha = [1, 1, 1, 1]

    out_indices = (0, 1, 2, 3)

    backbone = GPGBackbone(
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=out_indices,
        use_checkpoint=False,
        num_heads_fusion=mha,
        fusion_drop=args.fusion_drop,
        use_gpg=getattr(args, 'use_gpg', False),
        num_tmem=getattr(args, 'num_tmem', 1),
    )

    if pretrained:
        print('Initializing multi-modal Swin Transformer weights from ' + pretrained)
        backbone.init_weights(pretrained=pretrained)
    else:
        print('Randomly initialize multi-modal Swin Transformer weights.')
        backbone.init_weights()

    classifier = SimpleDecoding(8 * embed_dim)
    model = GPG(backbone, classifier)
    return model


def _load_model_gpg(pretrained, args):
    model = _segm_gpg(pretrained, args)
    return model


def gpg(pretrained='', args=None):
    """Constructor for the GPG base model."""
    return _load_model_gpg(pretrained, args)


# ------------------------------------------------
# GPG-One (BERT inside the model)
# ------------------------------------------------
def _segm_gpg_one(pretrained, args):
    if args.swin_type == 'tiny':
        embed_dim = 96
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'small':
        embed_dim = 96
        depths = [2, 2, 18, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'base':
        embed_dim = 128
        depths = [2, 2, 18, 2]
        num_heads = [4, 8, 16, 32]
    elif args.swin_type == 'large':
        embed_dim = 192
        depths = [2, 2, 18, 2]
        num_heads = [6, 12, 24, 48]
    else:
        assert False

    if ('window12' in pretrained) or getattr(args, 'window12', False):
        print('Window size 12!')
        window_size = 12
    else:
        window_size = 7

    if getattr(args, 'mha', None):
        mha = args.mha.split('-')
        mha = [int(a) for a in mha]
    else:
        mha = [1, 1, 1, 1]

    out_indices = (0, 1, 2, 3)

    backbone = GPGBackbone(
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=out_indices,
        use_checkpoint=False,
        num_heads_fusion=mha,
        fusion_drop=args.fusion_drop,
        use_gpg=getattr(args, 'use_gpg', False),
        num_tmem=getattr(args, 'num_tmem', 1),
    )

    if pretrained:
        print('Initializing multi-modal Swin Transformer weights from ' + pretrained)
        backbone.init_weights(pretrained=pretrained)
    else:
        print('Randomly initialize multi-modal Swin Transformer weights.')
        backbone.init_weights()

    classifier = SimpleDecoding(8 * embed_dim)
    model = GPGOne(backbone, classifier, args)
    return model


def _load_model_gpg_one(pretrained, args):
    model = _segm_gpg_one(pretrained, args)
    return model


def gpg_one(pretrained='', args=None):
    """Constructor for the GPG-One model."""
    return _load_model_gpg_one(pretrained, args)


# ------------------------------------------------
# Multi-stage PFR version (Progressive Feature Recursion Module)
# ------------------------------------------------
def _segm_gpg_pfr(pretrained, args):
    """Multi-stage PFR version (GPG backbone + ProgressiveFeatureRecursionModule)."""
    if args.swin_type == 'tiny':
        embed_dim = 96
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'small':
        embed_dim = 96
        depths = [2, 2, 18, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'base':
        embed_dim = 128
        depths = [2, 2, 18, 2]
        num_heads = [4, 8, 16, 32]
    elif args.swin_type == 'large':
        embed_dim = 192
        depths = [2, 2, 18, 2]
        num_heads = [6, 12, 24, 48]
    else:
        assert False

    if ('window12' in pretrained) or getattr(args, 'window12', False):
        print('Window size 12!')
        window_size = 12
    else:
        window_size = 7

    if getattr(args, 'mha', None):
        mha = args.mha.split('-')
        mha = [int(a) for a in mha]
    else:
        mha = [1, 1, 1, 1]

    out_indices = (0, 1, 2, 3)

    backbone = GPGBackbone(
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=out_indices,
        use_checkpoint=False,
        num_heads_fusion=mha,
        fusion_drop=args.fusion_drop,
        use_gpg=getattr(args, 'use_gpg', False),
        num_tmem=getattr(args, 'num_tmem', 1),
    )

    if pretrained:
        print('Initializing multi-modal Swin Transformer weights from ' + pretrained)
        backbone.init_weights(pretrained=pretrained)
    else:
        print('Randomly initialize multi-modal Swin Transformer weights.')
        backbone.init_weights()

    # c4_dims = 8 * embed_dim, same as the original SimpleDecoding
    c4_dims = 8 * embed_dim

    # number of PFR stages
    pfr_stages = getattr(args, 'pfr_stages', 2)
    pfr_channels = getattr(args, 'pfr_channels', None)
    classifier = ProgressiveFeatureRecursionModule(
        c4_dims=c4_dims,
        factor=2,
        pfr_stages=pfr_stages,
        pfr_channels=pfr_channels,
    )

    model = GPGPFR(backbone, classifier)
    return model


def _load_model_gpg_pfr(pretrained, args):
    model = _segm_gpg_pfr(pretrained, args)
    return model


def gpg_pfr(pretrained='', args=None):
    """Public constructor for the multi-stage PFR model."""
    return _load_model_gpg_pfr(pretrained, args)


# ------------------------------------------------
# RFD-enhanced multi-stage PFR (Region-Adaptive Feature Disentanglement Module)
# ------------------------------------------------
def _segm_gpg_pfr_rfd(pretrained, args):
    """Multi-stage PFR version (RFD enhanced)."""
    if args.swin_type == 'tiny':
        embed_dim = 96
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'small':
        embed_dim = 96
        depths = [2, 2, 18, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'base':
        embed_dim = 128
        depths = [2, 2, 18, 2]
        num_heads = [4, 8, 16, 32]
    elif args.swin_type == 'large':
        embed_dim = 192
        depths = [2, 2, 18, 2]
        num_heads = [6, 12, 24, 48]
    else:
        assert False

    if ('window12' in pretrained) or getattr(args, 'window12', False):
        print('Window size 12!')
        window_size = 12
    else:
        window_size = 7

    if getattr(args, 'mha', None):
        mha = args.mha.split('-')
        mha = [int(a) for a in mha]
    else:
        mha = [1, 1, 1, 1]

    out_indices = (0, 1, 2, 3)

    backbone = GPGBackbone(
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=out_indices,
        use_checkpoint=False,
        num_heads_fusion=mha,
        fusion_drop=args.fusion_drop,
        use_gpg=getattr(args, 'use_gpg', False),
        num_tmem=getattr(args, 'num_tmem', 1),
    )

    if pretrained:
        print('Initializing multi-modal Swin Transformer weights from ' + pretrained)
        backbone.init_weights(pretrained=pretrained)
    else:
        print('Randomly initialize multi-modal Swin Transformer weights.')
        backbone.init_weights()

    c4_dims = 8 * embed_dim

    pfr_stages = getattr(args, 'pfr_stages', 2)
    pfr_channels = getattr(args, 'pfr_channels', None)
    classifier = ProgressiveFeatureRecursionModuleRFD(
        c4_dims=c4_dims,
        factor=2,
        pfr_stages=pfr_stages,
        pfr_channels=pfr_channels,
    )

    model = GPGPFR(backbone, classifier)
    return model


def _load_model_gpg_pfr_rfd(pretrained, args):
    model = _segm_gpg_pfr_rfd(pretrained, args)
    return model


def gpg_pfr_rfd(pretrained: str = '', args=None):
    """RFD-enhanced multi-stage PFR GPG."""
    return _load_model_gpg_pfr_rfd(pretrained, args)


# ------------------------------------------------
# GPG backbone builder
# ------------------------------------------------
def _build_backbone_gpg(args, pretrained):
    # Use existing config to choose Swin variant
    if args.swin_type == 'tiny':
        embed_dim = 96
        depths = [2, 2, 6, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'small':
        embed_dim = 96
        depths = [2, 2, 18, 2]
        num_heads = [3, 6, 12, 24]
    elif args.swin_type == 'base':
        embed_dim = 128
        depths = [2, 2, 18, 2]
        num_heads = [4, 8, 16, 32]
    elif args.swin_type == 'large':
        embed_dim = 192
        depths = [2, 2, 18, 2]
        num_heads = [6, 12, 24, 48]
    else:
        assert False

    if ('window12' in pretrained) or getattr(args, 'window12', False):
        print('Window size 12!')
        window_size = 12
    else:
        window_size = 7

    if getattr(args, 'mha', None):
        mha = args.mha.split('-')
        mha = [int(a) for a in mha]
    else:
        mha = [1, 1, 1, 1]

    out_indices = (0, 1, 2, 3)

    backbone = GPGBackbone(
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=out_indices,
        use_checkpoint=False,
        num_heads_fusion=mha,
        fusion_drop=args.fusion_drop,
        use_gpg=getattr(args, 'use_gpg', False),
        num_tmem=getattr(args, 'num_tmem', 1),
    )

    if pretrained:
        print('Initializing multi-modal Swin Transformer weights from ' + pretrained)
        backbone.init_weights(pretrained=pretrained)
    else:
        print('Randomly initialize multi-modal Swin Transformer weights.')
        backbone.init_weights()

    return backbone, embed_dim


