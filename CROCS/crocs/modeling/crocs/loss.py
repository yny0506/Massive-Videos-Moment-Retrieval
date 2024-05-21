import torch
from torch.functional import F
from crocs.data.datasets.utils import box_iou


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


class FalseNegBceLoss(object):
    def __init__(self, min_iou, max_iou, mask2d):
        self.min_iou, self.max_iou = min_iou, max_iou
        self.mask2d = mask2d
        self.bceloss = torch.nn.BCELoss(reduction='none')
        self.hinge_loss = False

    def linear_scale(self, iou):
        return (iou - self.min_iou) / (self.max_iou - self.min_iou)

    def __call__(self, scores2d, ious2d, scaler, epoch):
        iou1d = ious2d.masked_select(self.mask2d)
        scores1d = scores2d.masked_select(self.mask2d)
        loss = 0
        iou1d = self.linear_scale(iou1d).clamp(0, 1)
        scaled_bce_loss = (self.bceloss(scores1d, iou1d) * scaler)
        loss += torch.sum(scaled_bce_loss) / torch.sum(scaler.bool()) * 50
        # print(loss)
        return loss


def build_false_neg_bce_loss(cfg, mask2d):
    min_iou = cfg.MODEL.MMN.LOSS.MIN_IOU 
    max_iou = cfg.MODEL.MMN.LOSS.MAX_IOU
    return FalseNegBceLoss(min_iou, max_iou, mask2d)


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

    def __call__(self, feat2ds, sent_feats, iou2ds, gt_proposals):
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
        for i in range(len(sent_feats)):
            curr_num_sent += sent_feats[i].size(0)
            all_num_sent.append(curr_num_sent)

        for i, gt_per_video in enumerate(gt_proposals):
            iou_map_per_video = box_iou(gt_per_video, gt_per_video)
            iou_mask = iou_map_per_video < self.sent_removal_iou  # remove high iou sentence, keep low iou sentence
            sent_mask[all_num_sent[i]:all_num_sent[i+1], all_num_sent[i]:all_num_sent[i+1]] = iou_mask.float()

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
            topk_index = torch.topk(iou1d, self.top_k, dim=-1)[1]                                                   # num_sent x top_k
            selected_feat = feat1d.index_select(dim=1, index=topk_index.reshape(-1)).reshape(C, -1, self.top_k)     # C x num_sent x top_k
            selected_feat = selected_feat.permute(1, 2, 0)                                                          # num_sent x top_k x C
            # positive video proposal with pos/neg sentence samples
            vid_pos = torch.bmm(selected_feat,
                                sent_feat.unsqueeze(2)).reshape(-1, self.top_k) - self.margin                       # num_sent x top_k, bmm of (num_sent x top_k x C) and (num_sent x C x 1)
            vid_neg = torch.mm(selected_feat.view(-1, C),
                               sent_feat_cat_norm.t()).reshape(-1, self.top_k, sum_num_sent)                        # num_sent x topk x sum(num_sent), mm of (num_sent*top_k x C) and (C x sum(num_sent))
            vid_pos_list.append(vid_pos)
            vid_neg_list.append(vid_neg)
            # positive sentence with pos/neg video proposals
            sent_pos_list.append(vid_pos.clone())
            sent_neg_same_video = torch.mm(sent_feat, feat1d)                                                   # num_sent x num_sparse_selected_proposal
            iou_neg_mask = (iou1d < self.neg_iou).float()                                                       # only keep the low iou proposals as negative samples in the same video
            sent_neg_same_video = iou_neg_mask * sent_neg_same_video                           # num_sent x num_sparse_selected_proposal
            feat1d_other_video = feat1ds_norm.index_select(dim=0, index=torch.arange(
                B, device=feat2ds.device)[torch.arange(B, device=feat2ds.device) != i])                         # (B-1) x C x num_sparse_selected_proposal
            feat1d_other_video = feat1d_other_video.transpose(1, 0).reshape(C, -1)                              # C x ((B-1) x num_sparse_selected_proposal)
            sent_neg_other_video = torch.mm(sent_feat, feat1d_other_video)                                      # num_sent x ((B-1) x num_sparse_selected_proposal)
            sent_neg_all = [vid_pos.clone().unsqueeze(2),
                            sent_neg_same_video.unsqueeze(1).repeat(1, self.top_k, 1),
                            sent_neg_other_video.unsqueeze(1).repeat(1, self.top_k, 1)]
            sent_neg_list.append(torch.cat(sent_neg_all, dim=2))                                # num_sent x topk x (1 + num_same + num_other)
        vid_pos = (torch.cat(vid_pos_list, dim=0).transpose(0, 1)) / self.T_v                   # top_k x num_sent
        vid_neg = torch.cat(vid_neg_list, dim=0).permute(1, 0, 2)                               # top_k x this_cat_to_be_sum(num_sent) x sum(num_sent)
        vid_neg = (vid_neg - margin_mask) / self.T_v                                            # top_k x this_cat_to_be_sum(num_sent) (positive) x sum(num_sent) (negative)
        sent_mask += torch.diag(torch.ones(sum_num_sent, device=feat2ds.device)).float()
        vid_neg_exp = torch.exp(vid_neg) * sent_mask.clamp(min=0, max=1)
        loss_vid = -(vid_pos - torch.log(vid_neg_exp.sum(dim=2, keepdim=False))).mean()
        sent_pos = torch.cat(sent_pos_list, dim=0) / self.T_s
        sent_neg = torch.cat(sent_neg_list, dim=0) / self.T_s
        sent_neg_exp = torch.exp(sent_neg)
        loss_sent = -(sent_pos - torch.log(sent_neg_exp.sum(dim=2, keepdim=False) + self.eps)).mean()
        return loss_vid, loss_sent



def conditional_mm(A, B, filter_condition):
    device = a.device
    mm_list = []
    for i, row_A in enumerate(A):
        B[:, not filter_condition[i]] = 0.0
        row_A_mm = torch.mm(row_A.unsqueeze(0), B)
        mm_list.append(row_A_mm)
    return torch.cat(mm_list)



class FalseNegContrastiveLoss(object):
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

    def __call__(self, feat2ds, sent_feats, iou2ds, gt_proposals, simcse_thres, simcse_matrix):
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
        for i in range(len(sent_feats)):
            curr_num_sent += sent_feats[i].size(0)
            all_num_sent.append(curr_num_sent)

        for i, gt_per_video in enumerate(gt_proposals):
            iou_map_per_video = box_iou(gt_per_video, gt_per_video)
            iou_mask = iou_map_per_video < self.sent_removal_iou  # remove high iou sentence, keep low iou sentence
            sent_mask[all_num_sent[i]:all_num_sent[i+1], all_num_sent[i]:all_num_sent[i+1]] = iou_mask.float()

        sent_mask += torch.diag(torch.ones(sum_num_sent, device=feat2ds.device)).float()  # add the sentence itself to the denominator in the loss
        margin_mask = torch.diag(torch.ones(sum_num_sent, device=feat2ds.device)) * self.margin

        n_queries = []
        video_queries_range = []
        cum_n = 0
        for tmp_sent_feats in sent_feats:
            n_q = len(tmp_sent_feats)
            n_queries.append(n_q)
            video_queries_range.append([cum_n, cum_n+n_q])
            cum_n += n_q

        vid_pos_list = []
        vid_neg_list = []
        sent_pos_list = []
        sent_neg_list = []

        v2q_n_false_negs, q2v_n_false_negs = 0, 0
        v2q_n, q2v_n = 0, 0
        is_false_negs = simcse_matrix > simcse_thres

        for i, (sent_feat, iou2d) in enumerate(zip(sent_feats, iou2ds)):  # each video in the batch
            # select positive samples
            num_sent_this_batch = sent_feat.size(0)
            feat1d = feat1ds_norm[i, :, :]                                                                          # C x num_sparse_selected_proposal
            sent_feat = F.normalize(sent_feat, dim=1)                                                               # num_sent x C
            iou1d = iou2d.masked_select(self.mask2d).reshape(sent_feat.size(0), -1)                                 # num_sent x num_sparse_selected_proposal
            topk_index = torch.topk(iou1d, self.top_k, dim=-1)[1]                                                   # num_sent x top_k
            selected_feat = feat1d.index_select(dim=1, index=topk_index.reshape(-1)).reshape(C, -1, self.top_k)     # C x num_sent x top_k
            selected_feat = selected_feat.permute(1, 2, 0)                                                          # num_sent x top_k x C

            # positive video proposal with pos/neg sentence samples
            vid_pos = torch.bmm(selected_feat,
                                sent_feat.unsqueeze(2)).reshape(-1, self.top_k) - self.margin                       # num_sent x top_k, bmm of (num_sent x top_k x C) and (num_sent x C x 1)
            vid_neg = torch.mm(selected_feat.view(-1, C),
                               sent_feat_cat_norm.t()).reshape(-1, self.top_k, sum_num_sent)                        # num_sent x topk x sum(num_sent), mm of (num_sent*top_k x C) and (C x sum(num_sent))
            

            # v2q) False-negative filtering
            tmp_is_false_negs = is_false_negs.T[i] # (sum(num_sent), )
            vid_neg[:, :, tmp_is_false_negs] = 0.0

            v2q_n += len(vid_neg.view(-1))
            v2q_n_false_negs += len(vid_neg[:, :, ~tmp_is_false_negs].view(-1))

            vid_pos_list.append(vid_pos)
            vid_neg_list.append(vid_neg)

            # positive sentence with pos/neg video proposals
            sent_pos_list.append(vid_pos.clone())
            sent_neg_same_video = torch.mm(sent_feat, feat1d)                                                   # num_sent x num_sparse_selected_proposal
            iou_neg_mask = (iou1d < self.neg_iou).float()                                                       # only keep the low iou proposals as negative samples in the same video
            sent_neg_same_video = iou_neg_mask * sent_neg_same_video                           # num_sent x num_sparse_selected_proposal
            feat1d_other_video = feat1ds_norm.index_select(dim=0, index=torch.arange(
                B, device=feat2ds.device)[torch.arange(B, device=feat2ds.device) != i])                         # (B-1) x C x num_sparse_selected_proposal
            feat1d_other_video = feat1d_other_video.transpose(1, 0).reshape(C, -1)                              # C x ((B-1) x num_sparse_selected_proposal)
            sent_neg_other_video = torch.mm(sent_feat, feat1d_other_video).reshape(n_queries[i], B-1, -1)                       # num_sent x ((B-1) x num_sparse_selected_proposal)

            # q2v) False-negative filtering
            tmp_is_false_negs = is_false_negs[torch.arange(*video_queries_range[i])][:, torch.arange(B, device=feat2ds.device) != i] # (num_sent, (B-1))
            sent_neg_other_video[tmp_is_false_negs, :] = 0.0

            q2v_n += len(sent_neg_other_video.view(-1))
            q2v_n_false_negs += len(sent_neg_other_video[~tmp_is_false_negs, :].view(-1))

            sent_neg_other_video = sent_neg_other_video.reshape(n_queries[i], -1)

            sent_neg_all = [vid_pos.clone().unsqueeze(2),
                            sent_neg_same_video.unsqueeze(1).repeat(1, self.top_k, 1),
                            sent_neg_other_video.unsqueeze(1).repeat(1, self.top_k, 1)]
            sent_neg_list.append(torch.cat(sent_neg_all, dim=2))                                # num_sent x topk x (1 + num_same + num_other)

        v2q_weight = v2q_n / v2q_n_false_negs
        q2v_weight = q2v_n / q2v_n_false_negs

        vid_pos = (torch.cat(vid_pos_list, dim=0).transpose(0, 1)) / self.T_v                   # top_k x num_sent
        vid_neg = torch.cat(vid_neg_list, dim=0).permute(1, 0, 2)                               # top_k x this_cat_to_be_sum(num_sent) x sum(num_sent)
        vid_neg = (vid_neg - margin_mask) / self.T_v                                            # top_k x this_cat_to_be_sum(num_sent) (positive) x sum(num_sent) (negative)
        sent_mask += torch.diag(torch.ones(sum_num_sent, device=feat2ds.device)).float()
        vid_neg_exp = torch.exp(vid_neg*v2q_weight) * sent_mask.clamp(min=0, max=1)
        loss_vid = -(vid_pos - torch.log(vid_neg_exp.sum(dim=2, keepdim=False))).mean() 
        sent_pos = torch.cat(sent_pos_list, dim=0) / self.T_s
        sent_neg = torch.cat(sent_neg_list, dim=0) / self.T_s
        sent_neg_exp = torch.exp(sent_neg*q2v_weight)
        loss_sent = -(sent_pos - torch.log(sent_neg_exp.sum(dim=2, keepdim=False) + self.eps)).mean()
        return loss_vid, loss_sent



class SimANSContrastiveLoss(object):
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

    def __call__(self, feat2ds, sent_feats, iou2ds, gt_proposals, simans_mode='q2v'):
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
        for i in range(len(sent_feats)):
            curr_num_sent += sent_feats[i].size(0)
            all_num_sent.append(curr_num_sent)

        for i, gt_per_video in enumerate(gt_proposals):
            iou_map_per_video = box_iou(gt_per_video, gt_per_video)
            iou_mask = iou_map_per_video < self.sent_removal_iou  # remove high iou sentence, keep low iou sentence
            sent_mask[all_num_sent[i]:all_num_sent[i+1], all_num_sent[i]:all_num_sent[i+1]] = iou_mask.float()
            
        sent_mask += torch.diag(torch.ones(sum_num_sent, device=feat2ds.device)).float()  # add the sentence itself to the denominator in the loss
        margin_mask = torch.diag(torch.ones(sum_num_sent, device=feat2ds.device)) * self.margin

        # select positive samples
        pos_index = 0
        iou2d = iou2ds[pos_index]
        sent_feat = sent_feats[pos_index]
        num_sent_this_batch = sent_feat.size(0)
        feat1d = feat1ds_norm[pos_index, :, :]                                                                          # C x num_sparse_selected_proposal
        sent_feat = F.normalize(sent_feat, dim=1)                                                               # num_sent x C
        iou1d = iou2d.masked_select(self.mask2d).reshape(sent_feat.size(0), -1)                                 # num_sent x num_sparse_selected_proposal
        topk_index = torch.topk(iou1d, self.top_k, dim=-1)[1]                                                   # num_sent x top_k
        selected_feat = feat1d.index_select(dim=1, index=topk_index.reshape(-1)).reshape(C, -1, self.top_k)     # C x num_sent x top_k
        selected_feat = selected_feat.permute(1, 2, 0)                                                          # num_sent x top_k x C
        
        if simans_mode == "v2q":
            # positive video proposal with pos/neg sentence samples
            vid_pos = torch.bmm(selected_feat,
                                sent_feat.unsqueeze(2)).reshape(-1, self.top_k) - self.margin                       # num_sent x top_k, bmm of (num_sent x top_k x C) and (num_sent x C x 1)
            vid_neg = torch.mm(selected_feat.view(-1, C),
                            sent_feat_cat_norm.t()).reshape(-1, self.top_k, sum_num_sent)                        # num_sent x topk x sum(num_sent), mm of (num_sent*top_k x C) and (C x sum(num_sent))
            
            vid_pos = (vid_pos.transpose(0, 1)) / self.T_v                   # top_k x num_sent
            vid_neg = vid_neg.permute(1, 0, 2)                               # top_k x this_cat_to_be_sum(num_sent) x sum(num_sent)
            vid_neg = (vid_neg - margin_mask) / self.T_v                                            # top_k x this_cat_to_be_sum(num_sent) (positive) x sum(num_sent) (negative)
            sent_mask += torch.diag(torch.ones(sum_num_sent, device=feat2ds.device)).float()
            vid_neg_exp = torch.exp(vid_neg) * sent_mask.clamp(min=0, max=1)
            loss_vid = -(vid_pos - torch.log(vid_neg_exp.sum(dim=2, keepdim=False))).mean()

            # print('<<v2q simans loss>>')
            # print('num query: ', sum_num_sent)
            # print('selected feat: ', selected_feat.shape)
            # print('sent_feat: ', sent_feat.shape)
            # print('sent_feat_cat_norm: ', sent_feat_cat_norm.t().shape)
            # print('vid pos: ', vid_pos.shape)
            # print('vid_neg: ', vid_neg.shape)
            # print()
            # print('loss_vid: ', loss_vid)
            return loss_vid

        elif simans_mode == "q2v":
            # positive sentence with pos/neg video proposals
            vid_pos = torch.bmm(selected_feat,
                                sent_feat.unsqueeze(2)).reshape(-1, self.top_k) - self.margin                       # num_sent x top_k, bmm of (num_sent x top_k x C) and (num_sent x C x 1)
            sent_neg_same_video = torch.mm(sent_feat, feat1d)                                                   # num_sent x num_sparse_selected_proposal
            iou_neg_mask = (iou1d < self.neg_iou).float()                                                       # only keep the low iou proposals as negative samples in the same video
            sent_neg_same_video = iou_neg_mask * sent_neg_same_video                           # num_sent x num_sparse_selected_proposal
            feat1d_other_video = feat1ds_norm.index_select(dim=0, index=torch.arange(
                B, device=feat2ds.device)[torch.arange(B, device=feat2ds.device) != pos_index])                         # (B-1) x C x num_sparse_selected_proposal
            feat1d_other_video = feat1d_other_video.transpose(1, 0).reshape(C, -1)                              # C x ((B-1) x num_sparse_selected_proposal)
            sent_neg_other_video = torch.mm(sent_feat, feat1d_other_video)                                      # num_sent x ((B-1) x num_sparse_selected_proposal)

            # print('<<q2v simans loss>>')
            # print('mask true sum', torch.sum(iou_neg_mask))
            # print('num query: ', sum_num_sent)
            # print('selected feat: ', selected_feat.shape)
            # print('sent_feat: ', sent_feat.shape)
            # print('feat1d_other_video: ', feat1d_other_video.shape)
            # print('vid pos: ', vid_pos.shape)
            # print('sent neg same vid: ', sent_neg_same_video.shape)
            # print('sent neg other vid: ', sent_neg_other_video.shape)
            # print()
            sent_neg_all = [vid_pos.unsqueeze(2),
                            sent_neg_same_video.unsqueeze(1).repeat(1, self.top_k, 1),
                            sent_neg_other_video.unsqueeze(1).repeat(1, self.top_k, 1)]
            sent_neg_all = torch.cat(sent_neg_all, dim=2)                                # topk x (1 + num_same + num_other)

            sent_pos = (vid_pos.transpose(0, 1)) / self.T_s
            sent_neg = sent_neg_all / self.T_s
            sent_neg_exp = torch.exp(sent_neg)
            loss_sent = -(sent_pos - torch.log(sent_neg_exp.sum(dim=2, keepdim=False) + self.eps)).mean()
            # print('loss_sent: ', loss_sent)
            return loss_sent



def build_contrastive_loss(cfg, mask2d):
    return ContrastiveLoss(cfg, mask2d)


def build_simans_contrastive_loss(cfg, mask2d):
    return SimANSContrastiveLoss(cfg, mask2d)


def build_false_neg_contrastive_loss(cfg, mask2d):
    return FalseNegContrastiveLoss(cfg, mask2d)
    