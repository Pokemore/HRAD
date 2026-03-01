import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn

from functools import reduce
import operator
from bert.modeling_bert import BertModel

import torchvision
from lib import segmentation

import transforms as T
import utils
import numpy as np

import torch.nn.functional as F

import gc
from collections import OrderedDict

def _morphological_dilate(x, kernel_size=3):
    """Approximate binary dilation using max-pooling.

    x: [B, 1, H, W] in [0, 1]
    """
    pad = (kernel_size - 1) // 2
    return F.max_pool2d(x, kernel_size, stride=1, padding=pad)


def _morphological_erode(x, kernel_size=3):
    """Approximate binary erosion using 1 - max_pool(1 - x)."""
    pad = (kernel_size - 1) // 2
    y = 1.0 - x
    y = F.max_pool2d(y, kernel_size, stride=1, padding=pad)
    return 1.0 - y


def compute_boundary_map(x, kernel_size=3):
    """Compute soft boundary map via morphological gradient.

    x: foreground probability map [B, 1, H, W]
    """
    dilated = _morphological_dilate(x, kernel_size)
    eroded = _morphological_erode(x, kernel_size)
    boundary = torch.clamp(dilated - eroded, 0.0, 1.0)
    return boundary


# ------------------------------
# RFD-style auxiliary losses
#   1) contrastive loss (private_feat vs binary mask)
#   2) orthogonality loss (shared_feat vs private_feat)
# ------------------------------
def contrastive_binary_proto_loss(private_feat, target, max_samples=1024, tau=0.1):
    """
    A simple prototype-based contrastive loss for binary segmentation.

    private_feat: [B, C, H, W]
    target     : [B, H, W], {0,1}

    Idea:
      - For each image, sample some foreground pixels and background pixels.
      - Use the mean of foreground/background features as prototypes.
      - Use foreground pixel features as anchors and apply binary cross-entropy:
            sim(anchor, proto_fg) vs sim(anchor, proto_bg)
    """
    B, C, H, W = private_feat.shape
    device = private_feat.device

    # L2 normalize along channel
    feat = F.normalize(private_feat, dim=1)

    if target.shape[-2:] != (H, W):
        tgt = F.interpolate(target.unsqueeze(1).float(),
                            size=(H, W),
                            mode='nearest').squeeze(1).long()
    else:
        tgt = target

    losses = []
    for b in range(B):
        tb = tgt[b]            # [H,W]
        fb = feat[b].view(C, -1).transpose(0, 1)  # [HW, C]

        flat_t = tb.view(-1)
        pos_idx = (flat_t == 1).nonzero(as_tuple=False).view(-1)
        neg_idx = (flat_t == 0).nonzero(as_tuple=False).view(-1)

        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue

        n_pos = min(pos_idx.numel(), max_samples // 2)
        n_neg = min(neg_idx.numel(), max_samples // 2)

        perm_pos = torch.randperm(pos_idx.numel(), device=device)[:n_pos]
        perm_neg = torch.randperm(neg_idx.numel(), device=device)[:n_neg]

        pos = fb[pos_idx[perm_pos]]  # [N_pos, C]
        neg = fb[neg_idx[perm_neg]]  # [N_neg, C]

        proto_pos = pos.mean(dim=0, keepdim=True)  # [1,C]
        proto_neg = neg.mean(dim=0, keepdim=True)  # [1,C]

        anchors = pos  # use foreground pixel features as anchors

        # logits: sim(anchor, proto_pos) vs sim(anchor, proto_neg)
        logit_pos = (anchors * proto_pos).sum(dim=1, keepdim=True) / tau  # [N_pos,1]
        logit_neg = (anchors * proto_neg).sum(dim=1, keepdim=True) / tau  # [N_pos,1]
        logits = torch.cat([logit_pos, logit_neg], dim=1)                 # [N_pos,2]

        labels = torch.zeros(anchors.size(0), dtype=torch.long, device=device)  # 0 corresponds to foreground prototype
        loss_b = F.cross_entropy(logits, labels)
        losses.append(loss_b)

    if len(losses) == 0:
        return torch.tensor(0.0, device=device)

    return sum(losses) / len(losses)


def orthogonality_loss(shared_feat, private_feat, eps=1e-6):
    """
    Encourage shared_feat and private_feat to be decorrelated / orthogonal.

    shared_feat : [B, C, H, W]
    private_feat: [B, C, H, W] (automatically interpolated to the same spatial size)
    """
    B, C, H, W = shared_feat.shape
    device = shared_feat.device

    if private_feat.shape[-2:] != (H, W):
        private_feat = F.interpolate(private_feat,
                                     size=(H, W),
                                     mode='bilinear',
                                     align_corners=True)

    loss_total = 0.0
    for b in range(B):
        s = shared_feat[b].view(C, -1)
        p = private_feat[b].view(C, -1)

        # remove mean
        s = s - s.mean(dim=1, keepdim=True)
        p = p - p.mean(dim=1, keepdim=True)

        # L2 normalize each channel
        s_norm = s / (s.norm(dim=1, keepdim=True) + eps)
        p_norm = p / (p.norm(dim=1, keepdim=True) + eps)

        # cross-correlation matrix
        corr = torch.mm(s_norm, p_norm.t())  # [C,C]

        # expected correlation coefficient to be close to 0
        loss_b = (corr ** 2).mean()
        loss_total += loss_b

    return loss_total / B


def get_dataset(image_set, transform, args):
    from data.dataset_refer_bert import ReferDataset
    ds = ReferDataset(
        args,
        split=image_set,
        image_transforms=transform,
        target_transforms=None,
    )
    num_classes = 2
    return ds, num_classes


# ------------------------------
# IoU calculation for validation
#   only care about "foreground IoU" (class = 1)
# ------------------------------
def IoU(pred, gt):
    """
    pred: logits [B, 2, H, W]
    gt:   label  [B, H, W], {0,1}

    Returns:
        - average foreground IoU within the batch (only class=1)
        - foreground intersection sum (Intersection, scalar)
        - foreground union sum (Union, scalar)
    """
    pred = pred.argmax(1)  # [B, H, W]
    IoU_val = 0.0
    Intersection = 0.0
    Union = 0.0
    eps = 1e-10

    for n in range(pred.shape[0]):
        p = pred[n]
        g = gt[n]

        inter = torch.logical_and(p == 1, g == 1).sum().item()
        uni = torch.logical_or(p == 1, g == 1).sum().item()

        if uni == 0:
            iou_n = 0.0
        else:
            iou_n = inter / (uni + eps)

        IoU_val += iou_n
        Intersection += inter
        Union += uni

    return IoU_val / pred.shape[0], Intersection, Union


# (the following two are not used anymore, but kept for reference)
def meter_iou(target, output, n_class):
    if target.shape != output.shape:
        raise ValueError("Shape mismatch: target and output must have the same shape")

    epsilon = 1e-10
    hist = _fast_hist(target.flatten(), output.flatten(), n_class)

    intersection = np.diag(hist)
    union = np.sum(hist, axis=1) + np.sum(hist, axis=0) - intersection
    iou = np.mean(intersection / (union + epsilon))

    return iou, intersection, union


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask].astype(int),
        minlength=n_class ** 2
    ).reshape(n_class, n_class)
    return hist


def get_transform(args):
    # keep original Resize + ToTensor + Normalize
    transforms = [
        T.Resize(args.img_size, args.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ]
    return T.Compose(transforms)


class DiceLoss:
    def __init__(self, axis: int = 1, smooth: float = 1e-6):
        self.axis = axis
        self.smooth = smooth

    def __call__(self, pred, targ):
        targ = self._one_hot(targ, pred.shape[self.axis])
        pred = F.softmax(pred, dim=self.axis)
        sum_dims = list(range(2, len(pred.shape)))
        inter = torch.sum(pred * targ, dim=sum_dims)
        union = torch.sum(pred + targ, dim=sum_dims)
        dice_score = (2. * inter + self.smooth) / (union + self.smooth)
        loss = 1 - dice_score
        return loss.mean()

    @staticmethod
    def _one_hot(x, classes: int, axis: int = 1):
        return torch.stack([torch.where(x == c, 1, 0) for c in range(classes)], axis=axis)


def _segmentation_loss(logits, target, weight):
    args_global = globals().get("args", None)
    use_gpg_loss = getattr(args_global, "use_gpg_loss", False) if args_global is not None else False
    if use_gpg_loss:
        dice_weight = getattr(args_global, "dice_weight", 0.1)
        dice_loss = DiceLoss()(logits, target)
        ce_loss = F.cross_entropy(logits, target, weight=weight)
        return (1 - dice_weight) * ce_loss + dice_weight * dice_loss
    return F.cross_entropy(logits, target, weight=weight)


def criterion(input, target):
    """Basic segmentation loss (CE or CE+Dice).

    For base/One version: input is [B, 2, H, W] logits.
    For PFR version: multi-stage logic is handled in train_one_epoch.
    """
    weight = torch.FloatTensor([0.9, 1.1]).cuda()
    return _segmentation_loss(input, target, weight)


# ------------------------------
# Evaluation
#   adapt PFR dict output & only calculate foreground IoU
# ------------------------------
def evaluate(model, data_loader, bert_model, args):
    model.eval()
    if bert_model is not None:
        bert_model.eval()
    use_gpg = getattr(args, 'use_gpg', False)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    total_its = 0
    acc_ious = 0.0

    # evaluation variables
    cum_I, cum_U = 0.0, 0.0       # foreground intersection / union sum
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []                 # foreground IoU for each image

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 50, header):
            if len(data) == 4:
                image, target, sentences, attentions = data
                target_masks = None
                position_masks = None
            else:
                image, target, sentences, attentions, target_masks, position_masks = data
            image, target, sentences, attentions = (
                image.cuda(non_blocking=True),
                target.cuda(non_blocking=True),
                sentences.cuda(non_blocking=True),
                attentions.cuda(non_blocking=True),
            )

            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            if target_masks is not None and position_masks is not None:
                target_masks = target_masks.cuda(non_blocking=True).squeeze(1)
                position_masks = position_masks.cuda(non_blocking=True).squeeze(1)

            if bert_model is not None:
                last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]
                embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l)
                attentions_ = attentions.unsqueeze(dim=-1)       # (B, N_l, 1)
                if use_gpg and target_masks is not None and position_masks is not None:
                    t_feats = bert_model(sentences, attention_mask=target_masks)[0].permute(0, 2, 1)
                    p_feats = bert_model(sentences, attention_mask=position_masks)[0].permute(0, 2, 1)
                    t_mask = target_masks.unsqueeze(dim=-1)
                    p_mask = position_masks.unsqueeze(dim=-1)
                    output = model(image, embedding, l_mask=attentions_, t_feats=t_feats, t_mask=t_mask,
                                   p_feats=p_feats, p_mask=p_mask)
                else:
                    output = model(image, embedding, l_mask=attentions_)
            else:
                if use_gpg and target_masks is not None and position_masks is not None:
                    output = model(image, sentences, l_mask=attentions, t_mask=target_masks, p_mask=position_masks)
                else:
                    output = model(image, sentences, l_mask=attentions)

            # adapt PFR dict output
            if isinstance(output, dict):
                logits = output["final_mask"]
            else:
                logits = output

            total_its += 1

            # only calculate foreground IoU
            iou, I, U = IoU(logits, target)
            acc_ious += iou
            mean_IoU.append(iou)
            cum_I += I
            cum_U += U

            for n_eval_iou in range(len(eval_seg_iou_list)):
                eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                seg_correct[n_eval_iou] += (iou >= eval_seg_iou)

            seg_total += 1

        # gather the stats from all processes (IoU related)
        acc_ious = utils.all_reduce_mean(acc_ious)
        total_its = utils.all_reduce_mean(total_its)

        cum_I = utils.all_reduce_mean(cum_I)
        cum_U = utils.all_reduce_mean(cum_U)

        seg_total = utils.all_reduce_mean(seg_total)

        # synchronize seg_correct array
        seg_correct_tensor = torch.tensor(seg_correct, dtype=torch.float32, device='cuda')
        if utils.is_dist_avail_and_initialized():
            torch.distributed.all_reduce(seg_correct_tensor, op=torch.distributed.ReduceOp.SUM)
            seg_correct_tensor = seg_correct_tensor / utils.get_world_size()
        seg_correct = seg_correct_tensor.cpu().numpy().astype(np.int32)

    # --------- scalar mIoU (per-sample avg, foreground IoU) ----------
    iou = acc_ious / (total_its + 1e-10)
    mIoU = float(np.mean(mean_IoU)) if len(mean_IoU) > 0 else 0.0

    # --------- overall IoU (all data foreground IoU) ----------
    overall_iou = float(cum_I / (cum_U + 1e-10)) if cum_U > 0 else 0.0

    # --------- print results ----------
    print('Mean foreground IoU (per-sample avg) is %.2f\n' % (mIoU * 100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % (
            str(eval_seg_iou_list[n_eval_iou]),
            seg_correct[n_eval_iou] * 100. / max(seg_total, 1),
        )
    results_str += '    overall foreground IoU = %.2f\n' % (overall_iou * 100.)
    print(results_str)

    # return two metrics: all_reduce mean IoU, and overall IoU (only foreground)
    return 100 * iou, overall_iou * 100


# ------------------------------
# Train one epoch
#   Deep Supervision + BAR + (optional) RFD losses
# ------------------------------
def train_one_epoch(model, criterion_fn, optimizer, data_loader, lr_scheduler, epoch, print_freq,
                    iterations, bert_model):
    model.train()
    if bert_model is not None:
        bert_model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.8e}'))
    header = 'Epoch: [{}]'.format(epoch)
    train_loss = 0.0
    total_its = 0

    args_global = globals().get("args", None)
    use_gpg = getattr(args_global, "use_gpg", False) if args_global is not None else False
    loss_contrast_weight = getattr(args_global, "loss_contrast_weight", 0.0) if args_global is not None else 0.0
    loss_ortho_weight = getattr(args_global, "loss_ortho_weight", 0.0) if args_global is not None else 0.0
    contrast_max_samples = getattr(args_global, "contrast_max_samples", 1024) if args_global is not None else 1024
    contrast_tau = getattr(args_global, "contrast_tau", 0.1) if args_global is not None else 0.1

    # reduce regular log output frequency
    reduced_print_freq = max(print_freq * 2, 50)  # at least every 50 batches
    for data in metric_logger.log_every(data_loader, reduced_print_freq, header):
        total_its += 1
        if len(data) == 4:
            image, target, sentences, attentions = data
            target_masks = None
            position_masks = None
        else:
            image, target, sentences, attentions, target_masks, position_masks = data
        image, target, sentences, attentions = (
            image.cuda(non_blocking=True),
            target.cuda(non_blocking=True),
            sentences.cuda(non_blocking=True),
            attentions.cuda(non_blocking=True),
        )

        sentences = sentences.squeeze(1)
        attentions = attentions.squeeze(1)
        if target_masks is not None and position_masks is not None:
            target_masks = target_masks.cuda(non_blocking=True).squeeze(1)
            position_masks = position_masks.cuda(non_blocking=True).squeeze(1)

        if bert_model is not None:
            last_hidden_states = bert_model(sentences, attention_mask=attentions)[0]  # (B, N, 768)
            embedding = last_hidden_states.permute(0, 2, 1)  # (B, 768, N_l)
            attentions_ = attentions.unsqueeze(dim=-1)       # (B, N_l, 1)
            if use_gpg and target_masks is not None and position_masks is not None:
                t_feats = bert_model(sentences, attention_mask=target_masks)[0].permute(0, 2, 1)
                p_feats = bert_model(sentences, attention_mask=position_masks)[0].permute(0, 2, 1)
                t_mask = target_masks.unsqueeze(dim=-1)
                p_mask = position_masks.unsqueeze(dim=-1)
                output = model(image, embedding, l_mask=attentions_, t_feats=t_feats, t_mask=t_mask,
                               p_feats=p_feats, p_mask=p_mask)
            else:
                output = model(image, embedding, l_mask=attentions_)
        else:
            if use_gpg and target_masks is not None and position_masks is not None:
                output = model(image, sentences, l_mask=attentions, t_mask=target_masks, p_mask=position_masks)
            else:
                output = model(image, sentences, l_mask=attentions)

        # --------------------------------------------------
        # Deep Supervision + BAR (only when PFR / PFR_RFD)
        # --------------------------------------------------
        weight = torch.FloatTensor([0.9, 1.1]).cuda()

        loss_final = torch.tensor(0.0, device=image.device)
        loss_stage = torch.tensor(0.0, device=image.device)
        loss_coarse = torch.tensor(0.0, device=image.device)
        loss_bdry = torch.tensor(0.0, device=image.device)
        loss_contrast = torch.tensor(0.0, device=image.device)
        loss_ortho = torch.tensor(0.0, device=image.device)

        if isinstance(output, dict):
            # 1) get multi-stage logits
            coarse_mask = output.get("coarse_mask", None)   # [B, 2, H, W]
            stage_masks = output.get("stage_masks", [])     # List[[B, 2, H, W]]
            final_mask = output.get("final_mask", None)     # [B, 2, H, W]

            if final_mask is None:
                raise ValueError("Deep supervision dict output must contain 'final_mask'.")

            # additional: RFD decoupled features (only when PFR_RFD)
            private_feat = output.get("private_feat", None)  # [B,C,H',W']
            shared_feat = output.get("shared_feat", None)    # [B,C,H',W']

            alpha = 0.4   # fine-tuning total weight
            beta = 0.1    # coarse segmentation weight
            lambda_bdry = 0.1  # BAR boundary loss weight (RRSIS-D decreased)
            ksize = 3

            # -------------------
            # (1) multi-stage segmentation loss
            # -------------------
            # final stage main loss
            loss_final = _segmentation_loss(final_mask, target, weight)

            # coarse stage loss (Stage-0)
            if coarse_mask is not None:
                loss_coarse = _segmentation_loss(coarse_mask, target, weight)
            else:
                loss_coarse = torch.tensor(0.0, device=final_mask.device)

            # average loss over all PFR stages
            if stage_masks is not None and len(stage_masks) > 0:
                stage_losses = [
                    _segmentation_loss(m, target, weight) for m in stage_masks
                ]
                loss_stage = sum(stage_losses) / len(stage_losses)
            else:
                loss_stage = torch.tensor(0.0, device=final_mask.device)

            # total segmentation loss
            loss_seg = loss_final + alpha * loss_stage + beta * loss_coarse

            # -------------------
            # (2) BAR boundary supervision
            # -------------------
            prob_pred = torch.softmax(final_mask, dim=1)[:, 1:2, :, :]  # foreground prob
            prob_pred = torch.clamp(prob_pred, min=1e-6, max=1-1e-6)
            bdry_pred = compute_boundary_map(prob_pred, kernel_size=ksize)

            with torch.no_grad():
                gt_fg = (target == 1).float().unsqueeze(1)              # [B,1,H,W]
                bdry_gt = compute_boundary_map(gt_fg, kernel_size=ksize)

            loss_bdry = F.l1_loss(bdry_pred, bdry_gt)

            # -------------------
            # (3) RFD-style auxiliary losses (only when outputs include features and weight > 0)
            # -------------------
            if private_feat is not None and loss_contrast_weight > 0:
                loss_contrast = contrastive_binary_proto_loss(private_feat, target, max_samples=contrast_max_samples, tau=contrast_tau)

            if (shared_feat is not None) and (private_feat is not None) and (loss_ortho_weight > 0):
                loss_ortho = orthogonality_loss(shared_feat, private_feat)

            # total loss = seg + lambda * boundary + RFD regularization
            loss = loss_seg + lambda_bdry * loss_bdry \
                   + loss_contrast_weight * loss_contrast \
                   + loss_ortho_weight * loss_ortho

        else:
            # base / One: single-stage segmentation loss
            loss = criterion_fn(output, target)

        optimizer.zero_grad()  # set_to_none=True is only available in pytorch 1.6+
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"[WARN] Epoch {epoch} iter {total_its}: invalid loss {loss.item()}")
            continue
        loss.backward()
        # gradient clipping for model and bert_model
        parameters_to_clip = list(model.parameters())
        if bert_model is not None:
            parameters_to_clip.extend(list(bert_model.parameters()))
        torch.nn.utils.clip_grad_norm_(parameters_to_clip, max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()

        torch.cuda.synchronize()
        train_loss += loss.item()
        iterations += 1
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        del image, target, sentences, attentions, loss, output, data
        if bert_model is not None:
            del last_hidden_states, embedding

        #gc.collect()
        #torch.cuda.empty_cache()
        #torch.cuda.synchronize()


def main(args):
    dataset, num_classes = get_dataset("train",
                                       get_transform(args=args),
                                       args=args)
    dataset_test, _ = get_dataset("val",
                                  get_transform(args=args),
                                  args=args)

    # batch sampler
    print(f"local rank {args.local_rank} / global rank {utils.get_rank()} successfully built train dataset.")
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=num_tasks, rank=global_rank,
        shuffle=True
    )
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # data loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        pin_memory=args.pin_mem, drop_last=True
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers
    )

    # model initialization
    print(args.model)
    model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights,
                                              args=args)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], find_unused_parameters=True
    )
    single_model = model.module

    if args.model != 'gpg_one':
        model_class = BertModel
        bert_model = model_class.from_pretrained(args.ck_bert)
        bert_model.pooler = None  # workaround for a bug in some Transformers versions
        bert_model.cuda()
        bert_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bert_model)
        bert_model = torch.nn.parallel.DistributedDataParallel(
            bert_model, device_ids=[args.local_rank]
        )
        single_bert_model = bert_model.module
    else:
        bert_model = None
        single_bert_model = None

    # resume training (weights)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        single_model.load_state_dict(checkpoint['model'])
        if args.model != 'gpg_one' and 'bert_model' in checkpoint:
            single_bert_model.load_state_dict(checkpoint['bert_model'])

    # parameters to optimize (keep original param group design)
    backbone_no_decay = list()
    backbone_decay = list()
    for name, m in single_model.backbone.named_parameters():
        if 'norm' in name or 'absolute_pos_embed' in name or 'relative_position_bias_table' in name:
            backbone_no_decay.append(m)
        else:
            backbone_decay.append(m)

    if args.model != 'gpg_one':
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            # the following are the parameters of bert
            {"params": reduce(operator.concat,
                              [[p for p in single_bert_model.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]
    else:
        params_to_optimize = [
            {'params': backbone_no_decay, 'weight_decay': 0.0},
            {'params': backbone_decay},
            {"params": [p for p in single_model.classifier.parameters() if p.requires_grad]},
            # the following are the parameters of bert (inside GPGOne)
            {"params": reduce(operator.concat,
                              [[p for p in single_model.text_encoder.encoder.layer[i].parameters()
                                if p.requires_grad] for i in range(10)])},
        ]

    # optimizer (keep original settings)
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.lr,
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )

    # learning rate scheduler (keep original poly decay)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(data_loader) * args.epochs)) ** 0.9
    )

    # housekeeping
    start_time = time.time()
    iterations = 0
    best_oIoU = -0.1  # now represents the best foreground IoU

    # resume training (optimizer, lr scheduler, and the epoch)
 #   if args.resume:
 #       optimizer.load_state_dict(checkpoint['optimizer'])
 #       lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
 #       resume_epoch = checkpoint['epoch']
 #   else:
 #       resume_epoch = -999

    resume_epoch = -1

    # training loops
    for epoch in range(max(0, resume_epoch + 1), args.epochs):
        data_loader.sampler.set_epoch(epoch)
        train_one_epoch(
            model, criterion, optimizer, data_loader, lr_scheduler,
            epoch, args.print_freq, iterations, bert_model
        )
        iou, overallIoU = evaluate(model, data_loader_test, bert_model, args)

        print('Average foreground IoU {}'.format(iou))
        print('Overall foreground IoU {}'.format(overallIoU))
        save_checkpoint = (best_oIoU < overallIoU)
        if save_checkpoint:
            print('Better epoch: {}\n'.format(epoch))
            if single_bert_model is not None:
                dict_to_save = {
                    'model': single_model.state_dict(),
                    'bert_model': single_bert_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'lr_scheduler': lr_scheduler.state_dict()
                }
            else:
                dict_to_save = {
                    'model': single_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                    'lr_scheduler': lr_scheduler.state_dict()
                }

            utils.save_on_master(
                dict_to_save,
                os.path.join(args.output_dir,
                             'model_best_{}.pth'.format(args.model_id))
            )
            best_oIoU = overallIoU

    # summarize
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    # set up distributed learning
    utils.init_distributed_mode(args)
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
