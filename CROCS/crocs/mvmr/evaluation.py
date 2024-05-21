from terminaltables import AsciiTable
from tqdm import tqdm
import logging
import torch
from crocs.data.datasets.utils import iou, score2d_to_moments_scores
from crocs.utils.comm import is_main_process
from copy import deepcopy
import datetime
import pickle
import numpy as np


def nms_with_score(moments, scores, thresh):
    scores, ranks = scores.sort(descending=True)
    moments = moments[ranks]
    suppressed = ranks.zero_().bool()
    numel = suppressed.numel()
    for i in range(numel - 1):
        if suppressed[i]:
            continue
        mask = iou(moments[i+1:], moments[i]) > thresh
        suppressed[i+1:][mask] = True

    return moments[~suppressed], scores[~suppressed]


def multiple_positive_evaluate(cfg, dataset, predictions, nms_thresh, sample_indices_info, removed_data, recall_metrics=None, additional_name=''):
    print('multiple positive evaluating mode.')
    if cfg.DATASETS.NAME == 'tacos':
        partial_recall_metrics = list(range(5, 25, 5))
    else:
        partial_recall_metrics = list(range(5, 55, 5))

    recall_metrics = [1]+partial_recall_metrics if recall_metrics==None else recall_metrics
    if not is_main_process():
        return
    if cfg.DATASETS.NAME == "tacos":
        iou_metrics = (0.3, 0.5, 0.7)
    elif cfg.DATASETS.NAME == "activitynet":
        iou_metrics = (0.3, 0.5, 0.7)
    elif cfg.DATASETS.NAME == "charades":
        iou_metrics = (0.3, 0.5, 0.7)
    else:
        raise NotImplementedError("No support for %s dataset!" % cfg.DATASETS.NAME)
    
    dataset_name = dataset.__class__.__name__
    logger = logging.getLogger("mmn.inference")
    logger.info("Performing {} MVMR evaluation (Size: {}).".format(dataset_name, len(dataset)))
    num_recall_metrics, num_iou_metrics = len(recall_metrics), len(iou_metrics)
    recall_metrics = torch.tensor(recall_metrics)
    iou_metrics = torch.tensor(iou_metrics)
    num_clips = cfg.MODEL.MMN.NUM_CLIPS
    table = [['R@{},IoU@{:.01f}'.format(i, torch.round(j*100)/100) for i in recall_metrics for j in iou_metrics]]
    recall_x_iou = torch.zeros(num_recall_metrics, num_iou_metrics)
    num_instance = 0

    idx2vid = {idx:dataset.get_vid(idx) for idx in range(len(dataset.annos))}
    vid2idx = {e['vid'].split('.')[0]:i for i, e in enumerate(dataset.annos)}

    for cnt, (origin_idx, origin_result) in enumerate(predictions.items()):   # each positive (origin) video
        if cnt % 500 == 0:
            print(f'[ {datetime.datetime.now()} ] Evaluation {cnt} is over.')
            
        gt = dataset.get_moment(origin_idx) # ground truth: Q x 2

        origin_gt_moments = []
        origin_vid = idx2vid[origin_idx].split('.')[0]

        for i, tmp_pos_moments in enumerate(sample_indices_info[origin_vid]['pos_moments']):
            if len(tmp_pos_moments) != 0:
                tmp_pos_moments = [[vid2idx[e[0]], e[2]] for e in tmp_pos_moments]
                new_gt = [[origin_idx, gt[i]]] + tmp_pos_moments
            else:
                new_gt = [[origin_idx, gt[i]]]

            origin_gt_moments.append(new_gt)

        sample_indices = origin_result['sample_indices'] # indices of samples
        moments = origin_result['moment'] # predictions
        confidences = origin_result['confidence'] # predicted logits

        # compute recall for each query (consider all sample videos)
        query_idx = 0
        for gt_query_idx, gt_moments in enumerate(origin_gt_moments): # gt_moments contain multiple moments
            if gt_query_idx not in removed_data[origin_idx]:
                
                query_samples_moments = []
                query_samples_confidence = []
                moment2sample = []

                # compress predicted moments for one sample
                for sample_idx, query_sample_candidates, query_sample_scores in zip(sample_indices[query_idx], moments[query_idx], confidences[query_idx]):
                    query_sample_moments, query_sample_confidence = nms_with_score(query_sample_candidates, query_sample_scores, nms_thresh)
                    n_preds = len(query_sample_moments)

                    moment2sample += [sample_idx] * n_preds

                    query_samples_moments.append(query_sample_moments)
                    query_samples_confidence.append(query_sample_confidence)

                query_idx += 1

                query_samples_moments_cat = torch.cat(query_samples_moments)
                query_samples_confidence_cat = torch.cat(query_samples_confidence)

                query_samples_confidence_sorted, ranks = query_samples_confidence_cat.sort(descending=True)
                query_samples_moments_sorted = query_samples_moments_cat[ranks]
                moment2sample = torch.tensor(moment2sample)[ranks]
                
                for i, r in enumerate(recall_metrics):
                    multiple_bools = []
                    for gt_info in gt_moments:
                        gt_idx = gt_info[0]
                        gt_moment = gt_info[1]
                        
                        tmp_preds = deepcopy(query_samples_moments_sorted)
                        tmp_preds[moment2sample!=gt_idx] = torch.zeros(2).to(tmp_preds.device)
                        mious = iou(tmp_preds[:r], torch.tensor(gt_moment).to(tmp_preds.device)).to(iou_metrics.device)
                        bools = mious[:, None].expand(r, num_iou_metrics) >= iou_metrics
                        multiple_bools.append(bools)
                    
                    multiple_bools = torch.cat(multiple_bools, dim=0)
                    recall_x_iou[i] += multiple_bools.any(dim=0)
                
                num_instance += 1 


    recall_x_iou /= num_instance
    table.append(['{:.02f}'.format(recall_x_iou[i][j]*100) for i in range(num_recall_metrics) for j in range(num_iou_metrics)])
    table = AsciiTable(table)
    
    for i in range(num_recall_metrics*num_iou_metrics):
        table.justify_columns[i] = 'center'
    logger.info('\n' + table.table)
    result_dict = {}
    
    for i in range(num_recall_metrics):
        for j in range(num_iou_metrics):
            result_dict['R@{},IoU@{:.01f}'.format(recall_metrics[i], torch.round(iou_metrics[j]*100)/100)] = recall_x_iou[i][j]
            
    best_r1 = sum(recall_x_iou[0])/num_iou_metrics
    best_r5 = sum(recall_x_iou[1])/num_iou_metrics
    result_dict['Best_R1'] = best_r1
    result_dict['Best_R5'] = best_r5
    return result_dict

