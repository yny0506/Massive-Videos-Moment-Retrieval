import os
import json
import logging
import torch
import pickle
from .utils import  moment_to_iou2d, bert_embedding, clip_embedding, get_vid_feat, video2feats
from transformers import DistilBertTokenizer
import numpy as np
from collections import Counter


class CharadesDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, ann_file, feat_file, num_pre_clips, num_clips):
        super(CharadesDataset, self).__init__()
        self.cfg = cfg
        self.feat_file = feat_file
        self.num_pre_clips = num_pre_clips

        with open(ann_file, 'r') as f:
            annos = json.load(f)

        self.annos = []
        if self.cfg.DATASETS.FEATS_TYPE == 'original':
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        
        logger = logging.getLogger("mmn.trainer")
        logger.info("Preparing data, please wait...")
        
        for i, (vid, anno) in enumerate(annos.items()):
            duration = anno['duration']
            # Produce annotations
            moments = []
            all_iou2d = []
            sentences = []
            for i, (timestamp, sentence) in enumerate(zip(anno['timestamps'], anno['sentences'])):
                if timestamp[0] < timestamp[1]:
                    moment = torch.Tensor([max(timestamp[0], 0), min(timestamp[1], duration)])
                    moments.append(moment)
                    iou2d = moment_to_iou2d(moment, num_clips, duration)
                    all_iou2d.append(iou2d)
                    sentences.append(sentence)


            moments = torch.stack(moments)
            all_iou2d = torch.stack(all_iou2d)

            if self.cfg.DATASETS.FEATS_TYPE == 'original':
                queries, word_lens = bert_embedding(cfg, sentences, tokenizer)  # padded query of N*word_len, tensor of size = N
                
            elif self.cfg.DATASETS.FEATS_TYPE == 'CLIP':
                queries, word_lens = clip_embedding(sentences)

            assert moments.size(0) == all_iou2d.size(0)
            assert moments.size(0) == queries.size(0)
            assert moments.size(0) == word_lens.size(0)

            self.annos.append(
                {
                    'vid': vid,
                    'moment': moments,
                    'iou2d': all_iou2d,
                    'sentence': sentences,
                    'query': queries,
                    'wordlen': word_lens,
                    'duration': duration
                }
            )
        
        if self.cfg.DATASETS.FEATS_TYPE == 'CLIP':
            import pickle
            with open('./dataset/Charades_STA/Charades_CLIP_clip_feats.pkl', 'rb') as f:
                self.feats = pickle.load(f)
            
            annos_vids = [e['vid'] for e in self.annos]
            in_annos_indices = [k for k in annos_vids if k not in self.feats.keys()]
            self.annos = [anno for anno in self.annos if anno['vid'] not in in_annos_indices]

    def __getitem__(self, idx):
        if self.cfg.DATASETS.FEATS_TYPE == 'CLIP':
            feat = self.feats[self.annos[idx]['vid']]
        elif self.cfg.DATASETS.FEATS_TYPE == 'original':
            feat = get_vid_feat(self.feat_file, self.annos[idx]['vid'], self.num_pre_clips, dataset_name="charades")
        return feat, self.annos[idx]['query'], self.annos[idx]['wordlen'], self.annos[idx]['iou2d'], self.annos[idx]['moment'], len(self.annos[idx]['sentence']), self.annos[idx]['duration'], idx 
        
    def __len__(self):
        return len(self.annos)

    def get_duration(self, idx):
        return self.annos[idx]['duration']

    def get_sentence(self, idx):
        return self.annos[idx]['sentence']

    def get_moment(self, idx):
        return self.annos[idx]['moment']

    def get_vid(self, idx):
        return self.annos[idx]['vid']

    def get_video_feat(self, idx):
        if self.cfg.DATASETS.FEATS_TYPE == 'CLIP':
            feat = self.feats[self.annos[idx]['vid']]
        elif self.cfg.DATASETS.FEATS_TYPE == 'original':
            feat = get_vid_feat(self.feat_file, self.annos[idx]['vid'], self.num_pre_clips, dataset_name="charades")
        return feat






