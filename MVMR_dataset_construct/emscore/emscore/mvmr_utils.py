import torch
import clip
from PIL import Image
import json
import cv2
import numpy as np
from tqdm import tqdm
import math
from math import log
from torch.nn.utils.rnn import pad_sequence
import sys
import time
import os
from collections import defaultdict, Counter
from multiprocessing import Pool
from functools import partial
from itertools import chain

### Get moment-wise EMscore


def compute_correlation_uniquehuman(pred, all_human_scores):
    num_workers = 3
    import scipy.stats

    pred = np.around(pred, decimals=4)

    spearman = 0
    for worker_i in range(num_workers):
        tmp, p_value = scipy.stats.spearmanr(pred, all_human_scores[:, worker_i])
        assert p_value < 0.01
        spearman += tmp
    spearman /= num_workers
    spearman = np.around(spearman, decimals=4)

    kendalltau = 0
    for worker_i in range(num_workers):
        tmp, p_value = scipy.stats.kendalltau(pred, all_human_scores[:, worker_i])
        assert p_value < 0.01
        kendalltau += tmp
    kendalltau /= num_workers
    kendalltau = np.around(kendalltau, decimals=4)

    print('kendall: {}, spear: {}'.format(kendalltau, spearman))
    return kendalltau, spearman

def normalize_matrix(A):
    assert len(A.shape) == 2
    A_norm = torch.linalg.norm(A, dim=-1, keepdim=True)
    return A / A_norm

###### kms modification ######
def encode_video_moments(video_file, preprocess, model, batch_size, num_clips, device,  vid_feat_cache=None):

    # image_embed_start_time = time.perf_counter()
    #vid_feat_cache => video emb dictionary
    if vid_feat_cache:
        filename_with_extension = os.path.basename(video_file)
        # Use os.path.splitext to split the filename and the extension
        filename_without_extension = os.path.splitext(filename_with_extension)[0]
        image_features = vid_feat_cache[filename_without_extension]
    else:
        
        cap = cv2.VideoCapture(video_file)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        images = []
        count = 0
        ret = True
        
        while (count < frameCount and ret):
            ret, frame = cap.read()
            if not ret:  # if file is empty break loop
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))
            count += 1

        
        image_input = torch.tensor(np.stack(images)).to(device)
        image_features_list = []
        # bs = 256
        with torch.no_grad():
            n_inter = math.ceil(len(image_input)/batch_size)
            for i in range(n_inter):
                image_features = model.encode_image(image_input[i*batch_size: (i+1)*batch_size]).float()
                image_features_list.append(image_features)
        image_features = torch.cat(image_features_list, dim=0) # frame * D 
        image_features /= image_features.norm(dim=-1, keepdim=True)
        cap.release()

    twod_tan_image_features = []
    twod_tan_video_feature = []
    N, D = image_features.shape # N: number of Frames, D: feature dimension
    n = N//num_clips # clip 당 frame 수
    for length in range(n, N + 1, n):
        for start in range(0, N, n):
            end = start + length
            if end <= n*num_clips:
                sliced = image_features[start:end]
                vid_feature = normalize_matrix(torch.mean(sliced, dim=0, keepdim=True)).squeeze()
                twod_tan_image_features.append(sliced) # list() -> elements: n*D
                twod_tan_video_feature.append(vid_feature) # list() -> elelments: D

    # image_embed_end_time = time.perf_counter()
    # time_diff = image_embed_end_time - image_embed_start_time
    # print(f"image embed done in {time_diff:.2f} seconds")

    return twod_tan_image_features, twod_tan_video_feature, image_features
##############################

def encode_video(video_file, preprocess, model, batch_size, device):

    # cv_start_time = time.perf_counter()
    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    images = []
    count = 0
    ret = True
    
    while (count < frameCount and ret):
        ret, frame = cap.read()
        if not ret:  # if file is empty break loop
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))
        count += 1
    
    # cv_end_time = time.perf_counter()
    # time_diff = cv_end_time-cv_start_time
    # print(f"cv done in {time_diff:.2f} seconds")


    # image_embed_start_time = time.perf_counter()
    image_input = torch.tensor(np.stack(images)).to(device)
    image_features_list = []
    # bs = 256
    with torch.no_grad():
        n_inter = math.ceil(len(image_input)/batch_size)
        for i in range(n_inter):
            image_features = model.encode_image(image_input[i*batch_size: (i+1)*batch_size]).float()
            image_features_list.append(image_features)
    image_features = torch.cat(image_features_list, dim=0)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    cap.release()

    vid_feature = normalize_matrix(torch.mean(image_features, dim=0, keepdim=True)).squeeze()

    # image_embed_end_time = time.perf_counter()
    # time_diff = image_embed_end_time - image_embed_start_time
    # print(f"image embed done in {time_diff:.2f} seconds")

    return image_features, vid_feature

def encode_text(vid_caps, model, tokenizer, idf_dict, device):
    text_input = tokenizer(vid_caps).to(device=device)
    with torch.no_grad():
        text_features = model.encode_text(text_input, local=True).float()
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # For special tokens, use [SOS] and [EOS]
    txt_len = text_input.argmax(dim=-1)
    mask = torch.zeros_like(text_input)
    for i in range(len(mask)):
        mask[i][0:txt_len[i]+1] = 1
    
    # For special tokens, only use [EOS]
    # txt_len = text_input.argmax(dim=-1)
    # mask = torch.zeros_like(text_input)
    # for i in range(len(mask)):
    #     mask[i][1:txt_len[i]+1] = 1

    # # For special tokens, don't use [SOS] and [EOS]
    # txt_len = text_input.argmax(dim=-1)
    # mask = torch.zeros_like(text_input)
    # for i in range(len(mask)):
    #     mask[i][1:txt_len[i]] = 1
    
    idf_weights = torch.tensor([[idf_dict[int(i)] for i in a] for a in text_input.cpu()])

    return text_features, mask, idf_weights


def process(a, tokenizer=None):
    if tokenizer is not None:
        a = tokenizer(a)[0].tolist()
    return set(a)


def get_idf_dict(arr, tokenizer, nthreads=4):
    """
    Returns mapping from word piece index to its inverse document frequency.


    Args:
        - :param: `arr` (list of str) : sentences to process.
        - :param: `tokenizer` : a BERT tokenizer corresponds to `model`.
        - :param: `nthreads` (int) : number of CPU threads to use
    """
    idf_count = Counter()
    num_docs = len(arr)

    process_partial = partial(process, tokenizer=tokenizer)

    with Pool(nthreads) as p:
        idf_count.update(chain.from_iterable(p.map(process_partial, arr)))

    idf_dict = defaultdict(lambda: log((num_docs + 1) / (1)))
    idf_dict.update({idx: log((num_docs + 1) / (c + 1)) for (idx, c) in idf_count.items()})
    return idf_dict


def refs_greedy_cos(ref_embedding, ref_masks, ref_idf, hyp_embedding, hyp_masks, hyp_idf, return_matched_idx):
    """
    Compute greedy matching based on cosine similarity.

    Args:
        - :param: `ref_embedding` (torch.Tensor):
                   embeddings of reference sentences, BxKxd,
                   B: batch size, K: longest length, d: bert dimenison.
        - :param: `ref_masks` (torch.LongTensor): BxKxK, BERT attention mask for
                   reference sentences.
        - :param: `hyp_embedding` (torch.Tensor):
                   embeddings of candidate sentences, BxKxd,
                   B: batch size, K: longest length, d: bert dimenison
        - :param: `hyp_masks` (torch.LongTensor): BxKxK, BERT attention mask for
                   candidate sentences.
    """
    # ref_embedding and hyp_embedding are aleady L2-normalized.

    batch_size = ref_embedding.size(0)
    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
    masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim)
    masks = masks.float().to(sim.device)
    sim = sim * masks
    
    word_precision, matched_indices = sim.max(dim=2)
    word_recall = sim.max(dim=1)[0]

    hyp_idf.div_(hyp_idf.sum(dim=1, keepdim=True))
    ref_idf.div_(ref_idf.sum(dim=1, keepdim=True))
    precision_scale = hyp_idf.to(word_precision.device)
    recall_scale = ref_idf.to(word_recall.device)

    P = (word_precision * precision_scale).sum(dim=1)
    R = (word_recall * recall_scale).sum(dim=1)
    F = 2 * P * R / (P + R)
    
    if return_matched_idx:
        return P, R, F, matched_indices
    else:
        return P, R, F, torch.zeros_like(P)

def vid_greedy_cos(ref_embedding, ref_masks, hyp_embedding, hyp_masks, hyp_idf, return_matched_idx):
    """
    Compute greedy matching based on cosine similarity.

    Args:
        - :param: `ref_embedding` (torch.Tensor):
                   embeddings of reference sentences, BxKxd,
                   B: batch size, K: longest length, d: bert dimenison.
        - :param: `ref_masks` (torch.LongTensor): BxKxK, BERT attention mask for
                   reference sentences.
        - :param: `hyp_embedding` (torch.Tensor):
                   embeddings of candidate sentences, BxKxd,
                   B: batch size, K: longest length, d: bert dimenison
        - :param: `hyp_masks` (torch.LongTensor): BxKxK, BERT attention mask for
                   candidate sentences.
    """
    # ref_embedding and hyp_embedding are aleady L2-normalized.

    batch_size = ref_embedding.size(0)
    sim = torch.bmm(hyp_embedding, ref_embedding.transpose(1, 2))
    masks = torch.bmm(hyp_masks.unsqueeze(2).float(), ref_masks.unsqueeze(1).float())
    masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim)
    masks = masks.float().to(sim.device)
    sim = sim * masks

    word_precision, matched_indices = sim.max(dim=2)
    word_recall = sim.max(dim=1)[0]

    hyp_idf.div_(hyp_idf.sum(dim=1, keepdim=True))
    precision_scale = hyp_idf.to(word_precision.device)
    P = (word_precision * precision_scale).sum(dim=1)
    R = word_recall.sum(dim=1)/ref_masks.sum(dim=1)
    F = 2 * P * R / (P + R)
    
    if return_matched_idx:
        return P, R, F, matched_indices
    else:
        return P, R, F, torch.zeros_like(P)



def em_cos_score(
     num_clips, model, hyps, ori_cands, vids, vid_feat_cache, tokenizer, idf_dict, preprocess, verbose=False, batch_size=64, device="cuda:0", return_matched_idx=False
):
    """
    Compute EMScore.

    Args:
        - :param: `model` : a BERT model in `pytorch_pretrained_bert`
        - :param: `refs` (list of str): reference sentences
        - :param: `hyps` (list of str): candidate sentences
        - :param: `tokenzier` : a BERT tokenizer corresponds to `model`
        - :param: `verbose` (bool): turn on intermediate status update
        - :param: `batch_size` (int): bert score processing batch size
        - :param: `device` (str): device to use, e.g. 'cpu' or 'cuda'
    """

    vid_preds_local = []
    vid_pred_matched_idxs = []
    vid_preds_global = []


    """process text"""
    def dedup_and_sort(l):
        return sorted(list(set(l)), key=lambda x: len(x.split(" ")), reverse=True)

    sentences = dedup_and_sort(hyps)
    embs = []
    iter_range = range(0, len(sentences), batch_size)
    if verbose:
        print("computing text embedding.")
        iter_range = tqdm(iter_range)
    text_local_stats_dict = dict()
    text_global_stats_dict = dict()
    for batch_start in iter_range:
        sen_batch = sentences[batch_start: batch_start + batch_size]
        embs, masks, text_idfs = encode_text(sen_batch, model, tokenizer, idf_dict, device=device)
        embs = embs.cpu()
        masks = masks.cpu()
        # batch 수 만큼 get text embeddings
        for i, sen in enumerate(sen_batch):
            sequence_len = masks[i].sum().item()
            
            # For special tokens, use [SOS] and [EOS]
            local_emb = embs[i, 0:sequence_len]
            global_emb = embs[i, sequence_len-1]
            idf = text_idfs[i, 0:sequence_len]

            # For special tokens, don't use any
            # local_emb = embs[i, 1:sequence_len+1]
            # global_emb = embs[i, sequence_len+1]
            # idf = text_idfs[i, 1:sequence_len+1]

            # For special tokens, only use [EOS] 
            # local_emb = embs[i, 1:sequence_len+1]
            # global_emb = embs[i, sequence_len]
            # idf = text_idfs[i, 1:sequence_len+1]

            text_local_stats_dict[sen] = (local_emb, idf)
            text_global_stats_dict[sen] = global_emb
        

    """process video"""
    if vids:
        ori_vids = vids # video paths list
        unique_vids = list(set(vids))
        if verbose:
            print("computing vid embedding.")
        vid_local_stats_dict = dict()
        vid_global_stats_dict = dict()
        vid_local_momments_stats_dict = dict()
        vid_global_momments_stats_dict = dict()
        for vid_i in range(len(unique_vids)):
            video_file = unique_vids[vid_i]
            twod_tan_image_features, twod_tan_video_feature, image_features= encode_video_moments(video_file, preprocess, model, batch_size=512, num_clips = num_clips, device=device, vid_feat_cache=vid_feat_cache)
            # vid_name = video_file.split('/')[-1][:-4]
            vid_local_momments_stats_dict[video_file] = twod_tan_image_features
            vid_global_momments_stats_dict[video_file] = twod_tan_video_feature
            vid_local_stats_dict[video_file] = image_features


    def pad_local_batch_stats(sen_batch, stats_dict, device):
        stats = [stats_dict[s] for s in sen_batch]
        emb, idf = zip(*stats)
        emb = [e.to(device) for e in emb]
        lens = [e.size(0) for e in emb]
        emb_pad = pad_sequence(emb, batch_first=True, padding_value=0.0)
        idf_pad = pad_sequence(idf, batch_first=True)

        def length_to_mask(lens):
            lens = torch.tensor(lens, dtype=torch.long)
            max_len = max(lens)
            base = torch.arange(max_len, dtype=torch.long).expand(len(lens), max_len)
            return base < lens.unsqueeze(1)

        pad_mask = length_to_mask(lens).to(device)
        return emb_pad, pad_mask, idf_pad

    def pad_vid_local_batch_stats(sen_batch, stats_dict, device):
        stats = [stats_dict[s] for s in sen_batch]
        emb = stats
        emb = [e.to(device) for e in emb]
        lens = [e.size(0) for e in emb]
        emb_pad = pad_sequence(emb, batch_first=True, padding_value=0.0)

        def length_to_mask(lens):
            lens = torch.tensor(lens, dtype=torch.long)
            max_len = max(lens)
            base = torch.arange(max_len, dtype=torch.long).expand(len(lens), max_len)
            return base < lens.unsqueeze(1)

        pad_mask = length_to_mask(lens).to(device)
        return emb_pad, pad_mask
    
    def pad_global_batch_stats(sen_batch, stats_dict, device):
        stats = [stats_dict[s] for s in sen_batch]
        emb = stats
        emb = [e.to(device) for e in emb]
        emb_pad = pad_sequence(emb, batch_first=True, padding_value=0.0)
        return emb_pad

    
    def pad_vid_local_moments_batch(sen_batch, stats_dict, device):
         
        moments_list = [stats_dict[s] for s in sen_batch] #stats_dict[s] -> list

        def length_to_mask(lens):
            lens = torch.tensor(lens, dtype=torch.long)
            max_len = max(lens)
            base = torch.arange(max_len, dtype=torch.long).expand(len(lens), max_len)
            return base < lens.unsqueeze(1)
        
        emb_pad_list = []
        pad_mask_list = []
        for moments in moments_list:
            emb = moments
            emb = [e.to(device) for e in emb]
            lens = [e.size(0) for e in emb]
            emb_pad = pad_sequence(emb, batch_first=True, padding_value=0.0)
            pad_mask = length_to_mask(lens).to(device)
            emb_pad_list.append(emb_pad)
            pad_mask_list.append(pad_mask)
        # num_moment = (num_clips * num_clips + num_clips) / 2 => number of upper triangle moments
        # emb_pad_list : num_moments * max_num_frame * D  emb가 batch 개수 만큼 있음.
        return emb_pad_list, pad_mask_list

    def pad_global_moments_stats(sen_batch, stats_dict, device):
        moments_list = [stats_dict[s] for s in sen_batch]
        emb_pad_list = []
        for moments in moments_list:
            emb = moments
            emb = [e.to(device) for e in emb]
            emb_pad = pad_sequence(emb, batch_first=True, padding_value=0.0)
            emb_pad_list.append(emb_pad)
        return emb_pad_list # emb_pad_list : num_moments * D emb가 batch 개수 만큼 있음.


    


    """ if video used as ground truth """
    if vids:
        if verbose:
            print("computing greedy matching, video as ground truth.")
        iter_range = range(0, len(ori_cands), batch_size)    
        with torch.no_grad():
            for batch_start in iter_range: 
                ## sent
                batch_ori_hyp = ori_cands[batch_start: batch_start + batch_size]
                ori_hyp_stats_local = pad_local_batch_stats(batch_ori_hyp, text_local_stats_dict, device)
                ori_hyp_stats_global = pad_global_batch_stats(batch_ori_hyp, text_global_stats_dict, device)
                ## vid
                batch_ori_vids = ori_vids[batch_start: batch_start + batch_size]
                emb_pad_list, pad_mask_list = pad_vid_local_moments_batch(batch_ori_vids, vid_local_momments_stats_dict, device)
                ori_vids_moments_global = pad_global_moments_stats(batch_ori_vids, vid_global_momments_stats_dict, device)
                for i in range(batch_size):
                    if i >= len(ori_cands):
                        break #수정 가능성 농후
                    ori_hyp_stats_local_rep = (ori_hyp_stats_local[0][i].unsqueeze(0).repeat(int((num_clips * num_clips + num_clips) / 2), 1, 1),
                                               ori_hyp_stats_local[1][i].unsqueeze(0).repeat(int((num_clips * num_clips + num_clips) / 2), 1),
                                               ori_hyp_stats_local[2][i].unsqueeze(0).repeat(int((num_clips * num_clips + num_clips) / 2), 1))
                    emb_pad = emb_pad_list[i]
                    pad_mask = pad_mask_list[i]
                    P, R, F1, matched_indices = vid_greedy_cos(emb_pad, pad_mask, *ori_hyp_stats_local_rep, return_matched_idx)
                    vid_preds_local.append(torch.stack((P, R, F1), dim=-1).cpu())
                    vid_pred_matched_idxs.append(matched_indices)

                    global_vid_emb_pad = ori_vids_moments_global[i]
                    ori_hyp_stats_global_repeat = ori_hyp_stats_global[i].unsqueeze(0).repeat(int((num_clips * num_clips + num_clips) / 2), 1)
                    # print(ori_hyp_stats_global_repeat.unsqueeze(1).shape)
                    # print(global_vid_emb_pad.unsqueeze(1).transpose(1, 2).shape)
                    vid_s_cogr = torch.bmm(ori_hyp_stats_global_repeat.unsqueeze(1), global_vid_emb_pad.unsqueeze(1).transpose(1, 2)).squeeze()
                    vid_preds_global.append(vid_s_cogr.cpu())  


    results = dict()
    """ if video used as ground truth """
    if vids:
        vid_preds_local = torch.cat(vid_preds_local, dim=0).cpu()
        if len(vids) != 1:
            vid_preds_global = torch.cat(vid_preds_global, dim=0).cpu()
        else:
            vid_preds_global = vid_preds_global[0]
        results['vid_result'] = {}
        results['vid_result']['figr'] = vid_preds_local
        results['vid_result']['cogr'] = vid_preds_global
        results['vid_result']['matched_indices'] = torch.cat(vid_pred_matched_idxs)


    return results