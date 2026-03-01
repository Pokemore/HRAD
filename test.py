import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn

from bert.modeling_bert import BertModel
import torchvision

from lib import segmentation
import transforms as T
import utils

import numpy as np
from PIL import Image
import torch.nn.functional as F


def get_dataset(image_set, transform, args):
    from data.dataset_refer_bert import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None,
                      eval_mode=True
                      )
    num_classes = 2
    return ds, num_classes


def _get_meta_value(meta, key, index):
    if meta is None or key not in meta:
        return None
    value = meta[key]
    if isinstance(value, (list, tuple)):
        return value[index]
    if torch.is_tensor(value):
        return value[index].item()
    return value


def evaluate(model, data_loader, bert_model, device, save_mask_dir=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    args_global = globals().get("args", None)
    use_gpg = getattr(args_global, "use_gpg", False) if args_global is not None else False

    # evaluation variables
    cum_I, cum_U = 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct = np.zeros(len(eval_seg_iou_list), dtype=np.int32)
    seg_total = 0
    mean_IoU = []
    header = 'Test:'

    with torch.no_grad():
        if save_mask_dir:
            os.makedirs(save_mask_dir, exist_ok=True)

        for data in metric_logger.log_every(data_loader, 100, header):
            if len(data) == 4:
                image, target, sentences, attentions = data
                target_masks = None
                position_masks = None
                meta = None
            elif len(data) == 5:
                image, target, sentences, attentions, meta = data
                target_masks = None
                position_masks = None
            elif len(data) == 6:
                image, target, sentences, attentions, target_masks, position_masks = data
                meta = None
            else:
                image, target, sentences, attentions, target_masks, position_masks, meta = data
            image, target, sentences, attentions = image.to(device), target.to(device), \
                                                   sentences.to(device), attentions.to(device)
            sentences = sentences.squeeze(1)
            attentions = attentions.squeeze(1)
            if target_masks is not None and position_masks is not None:
                target_masks = target_masks.to(device).squeeze(1)
                position_masks = position_masks.to(device).squeeze(1)
            target = target.cpu().data.numpy()
            for j in range(sentences.size(-1)):
                if bert_model is not None:
                    last_hidden_states = bert_model(sentences[:, :, j], attention_mask=attentions[:, :, j])[0]
                    embedding = last_hidden_states.permute(0, 2, 1)
                    if use_gpg and target_masks is not None and position_masks is not None:
                        t_feats = bert_model(sentences[:, :, j], attention_mask=target_masks[:, :, j])[0].permute(0, 2, 1)
                        p_feats = bert_model(sentences[:, :, j], attention_mask=position_masks[:, :, j])[0].permute(0, 2, 1)
                        t_mask = target_masks[:, :, j].unsqueeze(-1)
                        p_mask = position_masks[:, :, j].unsqueeze(-1)
                        output = model(image, embedding, l_mask=attentions[:, :, j].unsqueeze(-1),
                                       t_feats=t_feats, t_mask=t_mask, p_feats=p_feats, p_mask=p_mask)
                    else:
                        output = model(image, embedding, l_mask=attentions[:, :, j].unsqueeze(-1))
                else:
                    if use_gpg and target_masks is not None and position_masks is not None:
                        output = model(image, sentences[:, :, j], l_mask=attentions[:, :, j],
                                       t_mask=target_masks[:, :, j], p_mask=position_masks[:, :, j])
                    else:
                        output = model(image, sentences[:, :, j], l_mask=attentions[:, :, j])

                output = output.cpu()
                output_mask = output.argmax(1).data.numpy()
                if save_mask_dir:
                    for b in range(output_mask.shape[0]):
                        file_name = _get_meta_value(meta, 'file_name', b)
                        ref_id = _get_meta_value(meta, 'ref_id', b)
                        img_id = _get_meta_value(meta, 'img_id', b)
                        if file_name:
                            stem = os.path.splitext(os.path.basename(file_name))[0]
                        else:
                            stem = f'img{img_id}' if img_id is not None else 'img'
                        mask_name = f'{stem}_ref{ref_id}_sent{j}.png'
                        mask_path = os.path.join(save_mask_dir, mask_name)
                        mask_uint8 = (output_mask[b].astype(np.uint8) * 255)
                        Image.fromarray(mask_uint8, mode='L').save(mask_path)
                I, U = computeIoU(output_mask, target)
                if U == 0:
                    this_iou = 0.0
                else:
                    this_iou = I*1.0/U
                mean_IoU.append(this_iou)
                cum_I += I
                cum_U += U
                for n_eval_iou in range(len(eval_seg_iou_list)):
                    eval_seg_iou = eval_seg_iou_list[n_eval_iou]
                    seg_correct[n_eval_iou] += (this_iou >= eval_seg_iou)
                seg_total += 1

            del image, target, sentences, attentions, output, output_mask
            if bert_model is not None:
                del last_hidden_states, embedding

    mean_IoU = np.array(mean_IoU)
    mIoU = np.mean(mean_IoU)
    print('Final results:')
    print('Mean IoU is %.2f\n' % (mIoU*100.))
    results_str = ''
    for n_eval_iou in range(len(eval_seg_iou_list)):
        results_str += '    precision@%s = %.2f\n' % \
                       (str(eval_seg_iou_list[n_eval_iou]), seg_correct[n_eval_iou] * 100. / seg_total)
    results_str += '    overall IoU = %.2f\n' % (cum_I * 100. / cum_U)
    print(results_str)


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                  ]

    return T.Compose(transforms)


def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U


def main(args):
    device = torch.device(args.device)
    dataset_test, _ = get_dataset(args.split, get_transform(args=args), args)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   sampler=test_sampler, num_workers=args.workers)
    print(args.model)
    single_model = segmentation.__dict__[args.model](pretrained='',args=args)
    checkpoint = torch.load(args.resume, map_location='cpu')
    single_model.load_state_dict(checkpoint['model'], strict=args.strict_load)
    model = single_model.to(device)

    if args.model != 'gpg_one':
        model_class = BertModel
        single_bert_model = model_class.from_pretrained(args.ck_bert)
        # work-around for a transformers bug; need to update to a newer version of transformers to remove these two lines
        if args.ddp_trained_weights:
            single_bert_model.pooler = None
        single_bert_model.load_state_dict(checkpoint['bert_model'])
        bert_model = single_bert_model.to(device)
    else:
        bert_model = None

    evaluate(model, data_loader_test, bert_model, device=device, save_mask_dir=args.save_mask_dir)


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print('Image size: {}'.format(str(args.img_size)))
    main(args)
