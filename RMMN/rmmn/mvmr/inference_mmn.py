from torch import nn
from torch.functional import F
from tqdm import tqdm
import numpy as np
import datetime
import pickle

import logging
import torch
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str

from rmmn.mvmr.evaluation import multiple_positive_evaluate
from rmmn.data.datasets.utils import score2d_to_moments_scores



def massive_videos_moment_retrieve(cfg, dataset, model, model_outputs, device, videos_sample_indices, removed_data=None):
    # V: Video, Q: Query, C: Clip, d:dim
    # input) map2d, map2d_iou : (B, d, C, C), sent_feat, sent_feat_iou: B x (Q, d)
    # output) iou_scores, contrastive_scores : (B, B) x (Q) => 1st B: positive (origin) video, 2nd: positive+negative videos
    map2d, map2d_iou, sent_feat, sent_feat_iou = model_outputs['map2d'], model_outputs['map2d_iou'], model_outputs['sent_feat'], model_outputs['sent_feat_iou']
    
    num_videos = len(map2d)
    num_clips = cfg.MODEL.MMN.NUM_CLIPS
    num_dims = cfg.MODEL.MMN.JOINT_SPACE_SIZE
        
    videos_moments = []
    videos_confidences = []

    videos_iou_scores = []
    videos_contrastive_scores = []

    for origin_idx in range(num_videos):
        if origin_idx % 500 == 0:
            print(f'{datetime.datetime.now()} MVMR {origin_idx} is over.')
            
        num_queries = len(sent_feat[origin_idx])

        queries_samples_moments = []
        queries_samples_scores = []

        queries_iou_scores = []
        queries_contrastive_scores = []

        q_i = 0

        for original_dataset_q_i in range(num_queries):
            if original_dataset_q_i not in removed_data[origin_idx]:
                v_q_sample_indices = videos_sample_indices[origin_idx][q_i]

                # mutual matching space part
                vid_feat = map2d[v_q_sample_indices]
                sf = sent_feat[origin_idx][original_dataset_q_i]

                vid_feat_norm = F.normalize(vid_feat, dim=1).permute(0,2,3,1).to(device)
                sf_norm = F.normalize(sf, dim=-1).unsqueeze(-1).to(device)
                
                contrastive_score = torch.mm(vid_feat_norm.reshape(-1,num_dims), sf_norm).reshape(-1, num_clips, num_clips) * model.feat2d.mask2d.to(device)

                # iou space part
                vid_feat_iou = map2d_iou[v_q_sample_indices]
                sf_iou = sent_feat_iou[origin_idx][original_dataset_q_i]

                vid_feat_iou_norm = F.normalize(vid_feat_iou, dim=1).permute(0,2,3,1).to(device)
                sf_iou_norm = F.normalize(sf_iou, dim=-1).unsqueeze(-1).to(device)

                iou_score = torch.mm(vid_feat_iou_norm.reshape(-1,num_dims), sf_iou_norm).reshape(-1, num_clips, num_clips)
                iou_score = (iou_score*10).sigmoid() * model.feat2d.mask2d.to(device)

                queries_iou_scores.append(iou_score)
                queries_contrastive_scores.append(contrastive_score)

                # compute confidence score
                sample_score2d = torch.pow(contrastive_score * 0.5 + 0.5, cfg.TEST.CONTRASTIVE_SCORE_POW) * iou_score

                query_samples_moments = []
                query_samples_scores = []

                for sample_idx, sample_pred_score2d in zip(v_q_sample_indices, sample_score2d):  # each video
                    sample_duration = dataset.get_duration(sample_idx)
                    sample_moments, sample_scores = score2d_to_moments_scores(sample_pred_score2d, num_clips, sample_duration)

                    query_samples_moments.append(sample_moments.detach().to('cpu'))
                    query_samples_scores.append(sample_scores.detach().to('cpu'))

                if len(query_samples_moments) != 0:
                    queries_samples_moments.append(torch.stack(query_samples_moments))
                    queries_samples_scores.append(torch.stack(query_samples_scores))

                q_i += 1
        
        if len(queries_samples_moments) != 0:
            videos_moments.append(torch.stack(queries_samples_moments))
            videos_confidences.append(torch.stack(queries_samples_scores))
            
            videos_iou_scores.append(torch.stack(queries_iou_scores))
            videos_contrastive_scores.append(torch.stack(queries_contrastive_scores))
        
    return videos_iou_scores, videos_contrastive_scores, videos_moments, videos_confidences
    

def compute_on_dataset(cfg, model, data_loader, device, timer=None, is_mvmr=False, num_samples=None, sample_indices=None, removed_data=None):
    model.eval()
    results_dict = {}

    map2d_outputs, map2d_iou_outputs, sent_feat_outputs, sent_feat_iou_outputs = [], [], [], []

    for i, batch in enumerate(tqdm(data_loader)):
        batches, indices = batch

        with torch.no_grad():
            if timer:
                timer.tic()
            map2d, map2d_iou, sent_feat, sent_feat_iou = model(batches.to(device), is_mvmr=is_mvmr)
            sent_feat = [sf.detach().cpu() for sf in sent_feat]
            sent_feat_iou = [sf_iou.detach().cpu() for sf_iou in sent_feat_iou]

            if timer:
                if not device.type == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            
            map2d_outputs.append(map2d.detach().cpu())
            map2d_iou_outputs.append(map2d_iou.detach().cpu())
            sent_feat_outputs += sent_feat
            sent_feat_iou_outputs += sent_feat_iou


    map2d_outputs = torch.cat(map2d_outputs)
    map2d_iou_outputs = torch.cat(map2d_iou_outputs)
    
    model_outputs = {'map2d':map2d_outputs, 'map2d_iou':map2d_iou_outputs, 'sent_feat':sent_feat_outputs, 'sent_feat_iou':sent_feat_iou_outputs}
    _, _, moments, confidences = massive_videos_moment_retrieve(cfg, data_loader.dataset, model, model_outputs, device, sample_indices, removed_data)

    results_dict.update(
        {video_id: {'confidence': result1, 'moment': result2, 'sample_indices': data_sample_indices} for video_id, (data_sample_indices, result1, result2) in enumerate(zip(sample_indices, confidences, moments))}
    )

    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    
    predictions = {}
    for p in all_predictions:
        predictions.update(p)

    idxs = list(sorted(predictions.keys()))
    predictions = {i:predictions[i] for i in idxs}
    return predictions

def inference(
        cfg,
        model,
        data_loader,
        dataset_name,
        nms_thresh,
        device="cuda",
        is_mvmr=False,
        num_samples=50,
        sample_indices=None,
        removed_data=None,
        additional_name='',
        sample_indices_info=None
    ):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("mmn.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset (Size: {}).".format(dataset_name, len(dataset)))
    inference_timer = Timer()

    # should modify this part to evaluate other models
    predictions = compute_on_dataset(cfg, model, data_loader, device, inference_timer, is_mvmr, num_samples, sample_indices, removed_data)

    # wait for all processes to complete before measuring the time
    synchronize()
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({:.03f} s / inference per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)

    multiple_positive_evaluate(cfg, dataset=dataset, predictions=predictions, nms_thresh=nms_thresh, sample_indices_info=sample_indices_info, removed_data=removed_data, additional_name=additional_name)
    