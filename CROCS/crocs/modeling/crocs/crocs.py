import torch
from torch import nn
from torch.functional import F
from .featpool import build_featpool  # downsample 1d temporal features to desired length
from .feat2d import build_feat2d  # use MaxPool1d/Conv1d to generate 2d proposal-level feature map from 1d temporal features
from .loss import build_contrastive_loss, build_simans_contrastive_loss, build_false_neg_contrastive_loss, build_false_neg_bce_loss
from .loss import build_bce_loss
from .text_encoder import build_text_encoder
from .proposal_conv import build_proposal_conv



class MMN(nn.Module):
    def __init__(self, cfg):
        super(MMN, self).__init__()
        self.only_iou_loss_epoch = cfg.SOLVER.ONLY_IOU
        self.featpool = build_featpool(cfg) 
        self.feat2d = build_feat2d(cfg)
        self.contrastive_loss = build_contrastive_loss(cfg, self.feat2d.mask2d)
        self.iou_score_loss = build_bce_loss(cfg, self.feat2d.mask2d)
        self.text_encoder = build_text_encoder(cfg)
        self.proposal_conv = build_proposal_conv(cfg, self.feat2d.mask2d)
        self.joint_space_size = cfg.MODEL.MMN.JOINT_SPACE_SIZE
        self.encoder_name = cfg.MODEL.MMN.TEXT_ENCODER.NAME

        self.simans_contrastive_loss = build_simans_contrastive_loss(cfg, self.feat2d.mask2d)
        self.false_neg_contrastive_loss = build_false_neg_contrastive_loss(cfg, self.feat2d.mask2d)
        self.false_neg_iou_score_loss = build_false_neg_bce_loss(cfg, self.feat2d.mask2d)

    def forward(self, batches, cur_epoch=1, is_mvmr=False, simans_mode=None, compute_iou_loss=True, compute_neg_iou_loss=False, false_neg_train=False, simcse_threshold=0.9, simcse_matrix=None):
        """
        Arguments:
            batches.all_iou2d: list(B) num_sent x T x T
            feat2ds: B x C x T x T
            sent_feats: list(B) num_sent x C
        """
        # backbone
        batches = batches.to('cuda')
        ious2d = batches.all_iou2d
        # assert len(ious2d) == batches.feats.size(0)
        # for idx, (iou, sent) in enumerate(zip(ious2d, batches.queries)):
        #     assert iou.size(0) == sent.size(0)
        #     assert iou.size(0) == batches.num_sentence[idx]
        
        feats = self.featpool(batches.feats)  # from pre_num_clip to num_clip with overlapped average pooling, e.g., 256 -> 128
        map2d = self.feat2d(feats)  # use MaxPool1d to generate 2d proposal-level feature map from 1d temporal features
        map2d, map2d_iou = self.proposal_conv(map2d)
        sent_feat, sent_feat_iou = self.text_encoder(batches.queries, batches.wordlens)

        # print(map2d_iou)
        # print(sent_feat_iou)
        
        if is_mvmr:
            return map2d, map2d_iou, sent_feat, sent_feat_iou

        # inference
        contrastive_scores = []
        iou_scores = []
        _, T, _ = map2d[0].size()

        sf_iou_list = []
        vid_feat_iou_list = []
        num_pos_data = len(map2d_iou) if len(sent_feat_iou) >= len(map2d_iou) else len(sent_feat_iou)

        for i in range(num_pos_data):  # sent_feat_iou: [num_sent x C] (len=B)
            # iou part
            sf_iou = sent_feat_iou[i]
            vid_feat_iou = map2d_iou[i]  # C x T x T
            vid_feat_iou_norm = F.normalize(vid_feat_iou, dim=0)
            sf_iou_norm = F.normalize(sf_iou, dim=1)
            iou_score = torch.mm(sf_iou_norm, vid_feat_iou_norm.reshape(vid_feat_iou_norm.size(0), -1)).reshape(-1, T, T)  # num_sent x T x T
            iou_scores.append((iou_score*10).sigmoid() * self.feat2d.mask2d)

        if compute_neg_iou_loss:         
            if simans_mode == 'q2v':
                sf_iou = sent_feat_iou[0]
                vid_feat_iou = map2d_iou[1:]  # B X C x T x T
                BN, C, T, _ = vid_feat_iou.shape
                
                vid_feat_iou_norm = F.normalize(vid_feat_iou, dim=1).permute(1,0,2,3).reshape(C, -1) # C X (B*T*T)
                sf_iou_norm = F.normalize(sf_iou, dim=1) # 1 X C

            elif simans_mode == 'v2q':
                vid_feat_iou = map2d_iou[0]  # C x T x T
                sf_iou = torch.cat(sent_feat_iou[1:])
                
                vid_feat_iou_norm = F.normalize(vid_feat_iou, dim=0).reshape(vid_feat_iou_norm.size(0), -1) # C x (T*T)
                sf_iou_norm = F.normalize(sf_iou, dim=1) # Q X C
                    
            neg_iou_scores = torch.mm(sf_iou_norm, vid_feat_iou_norm).reshape(-1, T, T)  # v2q: Q X T X T, q2v: B X T X T
            neg_iou_scores = (neg_iou_scores*10).sigmoid() * self.feat2d.mask2d
            n_neg_samples = len(neg_iou_scores)
            

        # loss
        if self.training:
            if simans_mode is not None:
                loss_iou = self.iou_score_loss(torch.cat(iou_scores, dim=0), torch.cat(ious2d, dim=0), cur_epoch) if compute_iou_loss else 0

                if compute_neg_iou_loss:
                    loss_iou_neg = self.iou_score_loss(neg_iou_scores, torch.zeros_like(neg_iou_scores), cur_epoch) if compute_neg_iou_loss else 0
                    loss_iou_neg = loss_iou_neg / n_neg_samples
                    
                loss_contrast = self.simans_contrastive_loss(map2d, sent_feat, ious2d, batches.moments, simans_mode=simans_mode)
                return loss_contrast, loss_iou, loss_iou_neg

            else:
                loss_iou = self.iou_score_loss(torch.cat(iou_scores, dim=0), torch.cat(ious2d, dim=0), cur_epoch) if compute_iou_loss else 0

                if false_neg_train:
                    loss_vid, loss_sent = self.false_neg_contrastive_loss(map2d, sent_feat, ious2d, batches.moments, simcse_threshold, simcse_matrix)
                else:
                    loss_vid, loss_sent = self.contrastive_loss(map2d, sent_feat, ious2d, batches.moments)

                return loss_vid, loss_sent, loss_iou
        else:
            for i in range(num_pos_data):
                # contrastive part
                sf = sent_feat[i]
                vid_feat = map2d[i, ...]  # C x T x T
                vid_feat_norm = F.normalize(vid_feat, dim=0)
                sf_norm = F.normalize(sf, dim=1)  # num_sent x C
                _, T, _ = vid_feat.size()
                contrastive_score = torch.mm(sf_norm, vid_feat_norm.reshape(vid_feat.size(0), -1)).reshape(-1, T, T) * self.feat2d.mask2d  # num_sent x T x T
                contrastive_scores.append(contrastive_score)

            contrastive_scores = [e.detach().cpu() for e in contrastive_scores]
            iou_scores = [e.detach().cpu() for e in iou_scores]
            sent_feat_iou = [e.detach().cpu() for e in sent_feat_iou]

            return map2d_iou.detach().to('cpu'), sent_feat_iou, contrastive_scores, iou_scores  # first two maps for visualization

        
