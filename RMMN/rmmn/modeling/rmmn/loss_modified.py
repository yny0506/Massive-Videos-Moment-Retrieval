import torch
from torch.functional import F
from rmmn.data.datasets.utils import box_iou
import pickle

class BceLoss(object):
    def __init__(self, min_iou, max_iou, mask2d):
        self.min_iou, self.max_iou = min_iou, max_iou
        self.mask2d = mask2d
        self.bceloss = torch.nn.BCELoss(reduction='none')
        self.hinge_loss = False

    def linear_scale(self, iou):
        return (iou - self.min_iou) / (self.max_iou - self.min_iou)

    def __call__(self, scores2d, ious2d, epoch):
        iou1d = ious2d.masked_select(self.mask2d)
        scores1d = scores2d.masked_select(self.mask2d)
        loss = 0
        iou1d = self.linear_scale(iou1d).clamp(0, 1)
        loss += self.bceloss(scores1d, iou1d).mean()
        return loss


def build_bce_loss(cfg, mask2d):
    min_iou = cfg.MODEL.MMN.LOSS.MIN_IOU 
    max_iou = cfg.MODEL.MMN.LOSS.MAX_IOU
    return BceLoss(min_iou, max_iou, mask2d)


def find_similar_videos(similarity_info):
    num_videos = len(sim_list)
    query_similar_videos = {}

    for vi in range(num_videos):
        mu = torch.mean(sim_list[vi], dim=-1)
        sigma = torch.std(sim_list[vi], dim=-1)
        thres = mu + sigma
        queries_is_similar = sim_list[vi] > thres.reshape(n_queries[vi], 1)
        
        queries_sim_indices = []
        for is_sim in queries_is_similar:
            sim_indices = torch.where(is_sim)[0].tolist()
            queries_sim_indices.append(sim_indices)
            
        query_similar_videos[vi] = queries_sim_indices

class ContrastiveLoss(object):
    def __init__(self, cfg, mask2d):
        self.mask2d = mask2d
        self.T_v = cfg.MODEL.MMN.LOSS.TAU_VIDEO
        self.T_s = cfg.MODEL.MMN.LOSS.TAU_SENT
        self.cri = torch.nn.CrossEntropyLoss()
        self.neg_iou = cfg.MODEL.MMN.LOSS.NEGATIVE_VIDEO_IOU
        self.top_k = cfg.MODEL.MMN.LOSS.NUM_POSTIVE_VIDEO_PROPOSAL
        self.sent_removal_iou = cfg.MODEL.MMN.LOSS.SENT_REMOVAL_IOU
        self.margin = cfg.MODEL.MMN.LOSS.MARGIN
        self.eps = 1e-6
        self.dataset = cfg.DATASETS.NAME

        print('loss modified!')
        with open('./sampling/charades_train_query2query_sim_list.pkl', 'rb') as f:
            self.similar_indices = pickle.load(f)


    def __call__(self, feat2ds, sent_feats, iou2ds, gt_proposals, batches_idx=None):
        """
            feat2ds: B x C x T x T
            sent_feats: list(B) num_sent x C
            iou2ds: list(B) num_sent x T x T
            gt_proposals: list(B) num_sent x 2, with format [start, end], unit being seconds (frame/fps)
        """
        # prepare tensors
        B, C, _, _ = feat2ds.size()
        feat1ds = feat2ds.masked_select(self.mask2d).reshape(B, C, -1)
        feat1ds_norm = F.normalize(feat1ds, dim=1)  # B x C x num_sparse_selected_proposal
        sent_feat_cat = torch.cat(sent_feats, 0)  # sum(num_sent) x C, whole batch
        sum_num_sent = sent_feat_cat.size(0)
        sent_feat_cat_norm = F.normalize(sent_feat_cat, dim=1)  # sum(num_sent) x C, whole batch
        sent_mask = torch.ones(sum_num_sent, sum_num_sent, device=feat2ds.device)

        all_num_sent = [0]
        curr_num_sent = 0
        for idx, i in zip(batches_idx, range(len(sent_feats))):
            curr_num_sent += sent_feats[i].size(0)
            all_num_sent.append(curr_num_sent)
        
        """ 하나의 비디오 내 text 끼리 얼마나 겹치는지 체크. 하나의 비디오 내에서도 겹치지 않는 text는 negative로 보나? 많이 겹치는 query는 0이됨. 즉 나중에 안씀.
            비디오 feature에 대해, 비슷하게 겹치는 쿼리가 있을 수 있잖아 걔네 제거해주는 용도임. 특히 하나의 비디오 내에 비슷한 쿼리가 많다보니 이거로 제거함"""
        for i, gt_per_video in enumerate(gt_proposals):
            iou_map_per_video = box_iou(gt_per_video, gt_per_video)
            iou_mask = iou_map_per_video < self.sent_removal_iou  # remove high iou sentence, keep low iou sentence
            sent_mask[all_num_sent[i]:all_num_sent[i+1], all_num_sent[i]:all_num_sent[i+1]] = iou_mask.float()

        # if batches_idx is not None:
        #     idx2i = {idx:i for i, idx in enumerate(batches_idx)}

        #     batches_sim_indices = {'q':[], 'v':[]}
        #     for idx in batches_idx:
        #         q_sim_indices = self.similar_indices['query_similar_videos_indices'][idx]
        #         # v_sim_indices = self.similar_indices['video_similar_queries_indices'][idx]

        #         q_sim_indices_in_batch = []
        #         for indices in q_sim_indices:
        #             indices_in_batch = torch.tensor(list((set(indices)-{idx}) & set(batches_idx)))
        #             q_sim_indices_in_batch.append(indices_in_batch)

        #         # v_sim_indices_in_batch = [torch.tensor(v_sim_indices.get(i, [])) for i in batches_idx]
                
        #         batches_sim_indices['q'].append(q_sim_indices_in_batch)
        #         # batches_sim_indices['v'].append(v_sim_indices_in_batch)


        #     v_diffcse_mask = torch.ones(sum_num_sent, sum_num_sent, device=feat2ds.device)

        #     for i, queries_sim_indices in enumerate(batches_sim_indices['q']):
        #         for qi, query_sim_indices in enumerate(queries_sim_indices):
        #             masked_query_indices = sum([list(range(all_num_sent[idx2i[idx.item()]], all_num_sent[idx2i[idx.item()]+1])) for idx in query_sim_indices], [])
        #             v_diffcse_mask[all_num_sent[i]+qi, masked_query_indices] = 0

        #     q_diffcse_mask = [torch.ones(len(tmp_sent_feats), len(sent_feats)-1) for tmp_sent_feats in sent_feats] # list(B) x num_sent x (B-1)
        #     # print(batches_sim_indices['q'])
        #     # print('@@@@\n')
        #     for vi, idx in enumerate(batches_idx):
        #         for qi in range(len(sent_feats[vi])):
        #             other_video_indices = torch.arange(B)[torch.arange(B) != vi]

        #             for i, oi in enumerate(other_video_indices):
        #                 if len(batches_sim_indices['q'][vi][qi]) != 0 and batches_idx[oi] in batches_sim_indices['q'][vi][qi]:
        #                     q_diffcse_mask[vi][qi][i] = 0
        
        if batches_idx is not None:
            idx2i = {idx:i for i, idx in enumerate(batches_idx)}
            # mask 1
            v_diffcse_mask = torch.ones(sum_num_sent, sum_num_sent, device=feat2ds.device)

            for i, origin_vi in enumerate(batches_idx):
                v_sim_indices = self.similar_indices[origin_vi]

                for origin_qi in range(len(v_sim_indices)):
                    for j, sample_vi in enumerate(batches_idx):
                        sim_indices = torch.tensor(v_sim_indices[origin_qi][sample_vi])
                        if len(sim_indices) != 0:
                            v_diffcse_mask[all_num_sent[i]+origin_qi, all_num_sent[j]+sim_indices] = 0

            # mask 2
            # q_diffcse_mask = [torch.ones(len(tmp_sent_feats), len(sent_feats)-1) for tmp_sent_feats in sent_feats] # list(B) x num_sent x (B-1)
            # for vi, idx in enumerate(batches_idx):
            #     for qi in range(len(sent_feats[vi])):
            #         other_video_indices = torch.arange(B)[torch.arange(B) != vi]

            #         for i, oi in enumerate(other_video_indices):
            #             if len(batches_sim_indices['q'][vi][qi]) != 0 and batches_idx[oi] in batches_sim_indices['q'][vi][qi]:
            #                 q_diffcse_mask[vi][qi][i] = 0

            # print(all_num_sent)
            # print(idx2i.keys())
            # print()
            # print(batches_sim_indices['q'])
            # print()
            # print(v_diffcse_mask)
            # print()
            #(num_sent, (B-1))

            # print()
            # print()
            # print(q_diffcse_mask)
            # print(1/0)

                # 여기서 mask를 아예 만들고 들어가자. sum(num_sent) X sum(num_sent) 짜리를 vid_neg_exp의 mask로
                # sent_neg_exp에 대해서는 각 비디오 별로 num_sent x (1 + num_same + num_other) 짜리를 mask로

        """@@@@@@@@@@@@@@@@@@@ 난 이제 이부분에 sent_mask에 특정 video와 diffcse로 겹치는 query들에 대해서 마스킹하는 파트만 추가해주면 됨"""
        
        sent_mask += torch.diag(torch.ones(sum_num_sent, device=feat2ds.device)).float()  # add the sentence itself to the denominator in the loss
        margin_mask = torch.diag(torch.ones(sum_num_sent, device=feat2ds.device)) * self.margin

        vid_pos_list = []
        vid_neg_list = []
        sent_pos_list = []
        sent_neg_list = []

        for i, (sent_feat, iou2d) in enumerate(zip(sent_feats, iou2ds)):  # each video in the batch
            # select positive samples
            num_sent_this_batch = sent_feat.size(0)
            feat1d = feat1ds_norm[i, :, :]                                                                          # C x num_sparse_selected_proposal
            sent_feat = F.normalize(sent_feat, dim=1)                                                               # num_sent x C
            iou1d = iou2d.masked_select(self.mask2d).reshape(sent_feat.size(0), -1)                                 # num_sent x num_sparse_selected_proposal

            " i-th video의 topk iou moment feature만을 가져옴. 얘네는 이제 postive moment feature지. chardes는 각 query 마다 단 한개만 pos moment로 씀 "
            topk_index = torch.topk(iou1d, self.top_k, dim=-1)[1]                                                   # num_sent x top_k
            selected_feat = feat1d.index_select(dim=1, index=topk_index.reshape(-1)).reshape(C, -1, self.top_k)     # C x num_sent x top_k
            selected_feat = selected_feat.permute(1, 2, 0)                                                          # num_sent x top_k x C
            
            """ 
            얘는 비디오가 주어졌을때, sent가 neg, pos
            1) vid_pos: 각 query 마다, topk iou의 positive moments feats와 positive query feats 유사도 계산
            2) vid_neg: 각 query 마다, topk iou의 positive moments feats와 all query feats 유사도 계산. 얘는 positive query feats에 대해서 마스킹하지는 않음.
            나중에 margin mask로 positive query feats에 대한 유사도는 감소시킴
            각각의 sent에 대한 positive moment (video)가 있지? 얘와 pos and neg sentennces 사이의 feat 구함

            여기서, vid_neg 부분을 보면, 모든 query가 동일 비디오를 활용하는건데 왜 굳이 비디오를 쿼리 개수만큼 써서 num_sent x sum(num_sent) 개수로 되냐?
            => 각 query별로 대응하는 moment 뽑으므로 동일 비디오 활용한다고 보기 어려움.
            """
            "@@ 결론: 여기서, vid_neg_list 에 similar queries를 제거하든지, similar queries도 분자에 추가하든지?"
            # positive video proposal with pos/neg sentence samples
            vid_pos = torch.bmm(selected_feat,
                                sent_feat.unsqueeze(2)).reshape(-1, self.top_k) - self.margin                       # num_sent x top_k, bmm of (num_sent x top_k x C) and (num_sent x C x 1)
            vid_neg = torch.mm(selected_feat.view(-1, C),
                               sent_feat_cat_norm.t()).reshape(-1, self.top_k, sum_num_sent)                        # num_sent x topk x sum(num_sent), mm of (num_sent*top_k x C) and (C x sum(num_sent))
            
            vid_pos_list.append(vid_pos)
            vid_neg_list.append(vid_neg)
            
            """
            얘는 query가 주어졌을때, video가 neg, pos
            1) sent_pos: 동일
            2) sent_neg: 
             - sent_neg_same_video는 우선 특정 비디오에 대해, moment feats와 비디오에 포함된 query 들의 sent feats 유사도 구하고, iou thres 밑인 애들만 씀.
            동일 비디오 내 에서 너무 유사한 애들은 neg sample로 안쓰겠다는 말.
             - sent_neg_other_video는 걍 다른 모든 비디오에 대해, moment feats와 중심 비디오에 포함된 query 들의 sent feats 유사도 구하여 neg sample로 씀
            """
            "@@ 결론: 여기서, sent_neg_list 에 similar videos를 제거하든지, similar videos도 분자에 추가하든지?"
            # positive sentence with pos/neg video proposals
            sent_pos_list.append(vid_pos.clone())
            sent_neg_same_video = torch.mm(sent_feat, feat1d)                                                   # num_sent x num_sparse_selected_proposal
            iou_neg_mask = (iou1d < self.neg_iou).float()                                                       # only keep the low iou proposals as negative samples in the same video
            sent_neg_same_video = iou_neg_mask * sent_neg_same_video                           # num_sent x num_sparse_selected_proposal

            feat1d_other_video = feat1ds_norm.index_select(dim=0, index=torch.arange(
                B, device=feat2ds.device)[torch.arange(B, device=feat2ds.device) != i])                         # (B-1) x C x num_sparse_selected_proposal
            feat1d_other_video = feat1d_other_video.transpose(1, 0).reshape(C, -1)                              # C x ((B-1) x num_sparse_selected_proposal)
            sent_neg_other_video = torch.mm(sent_feat, feat1d_other_video)                                      # num_sent x ((B-1) x num_sparse_selected_proposal)
            "여기서 sent_neg_other_video에 대해 sent 별로 마스킹해주면됨 (num_sent, (B-1))"
            """(수정핵심"""
            # num_proposal = sent_neg_same_video.shape[-1]
            # sent_neg_other_video = sent_neg_other_video*q_diffcse_mask[i].repeat(1,1,num_proposal).reshape(num_sent_this_batch, -1).to(sent_neg_other_video.device)
            """수정핵심)"""
            sent_neg_all = [vid_pos.clone().unsqueeze(2),
                            sent_neg_same_video.unsqueeze(1).repeat(1, self.top_k, 1),
                            sent_neg_other_video.unsqueeze(1).repeat(1, self.top_k, 1)]
            sent_neg_list.append(torch.cat(sent_neg_all, dim=2))                                # num_sent x topk x (1 + num_same + num_other)

        vid_pos = (torch.cat(vid_pos_list, dim=0).transpose(0, 1)) / self.T_v                   # top_k x num_sent
        vid_neg = torch.cat(vid_neg_list, dim=0).permute(1, 0, 2)                               # top_k x this_cat_to_be_sum(num_sent) x sum(num_sent)
        vid_neg = (vid_neg - margin_mask) / self.T_v                                            # top_k x this_cat_to_be_sum(num_sent) (positive) x sum(num_sent) (negative)

        sent_mask += torch.diag(torch.ones(sum_num_sent, device=feat2ds.device)).float()
        """(수정핵심"""
        vid_neg_exp = torch.exp(vid_neg) * sent_mask.clamp(min=0, max=1) * v_diffcse_mask
        # print(torch.exp(vid_neg))
        # print()
        # print(vid_neg_exp)
        # print(1/0)

        """수정핵심)"""
        # print()
        # print(vid_neg_exp.shape)
        loss_vid = -(vid_pos - torch.log(vid_neg_exp.sum(dim=2, keepdim=False))).mean()

        sent_pos = torch.cat(sent_pos_list, dim=0) / self.T_s
        sent_neg = torch.cat(sent_neg_list, dim=0) / self.T_s
        sent_neg_exp = torch.exp(sent_neg)
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        # print()
        # print(sent_pos.shape)
        # print(sent_neg_exp.shape)
        # print(1/0)
        loss_sent = -(sent_pos - torch.log(sent_neg_exp.sum(dim=2, keepdim=False) + self.eps)).mean()

        return loss_vid, loss_sent


def build_contrastive_loss(cfg, mask2d):
    return ContrastiveLoss(cfg, mask2d)
