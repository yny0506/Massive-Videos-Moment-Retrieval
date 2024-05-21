import json
import logging
import torch
from .utils import moment_to_iou2d, bert_embedding, clip_embedding, get_vid_feat, video2feats
from transformers import DistilBertTokenizer
from torch.functional import F


class TACoSDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, ann_file, feat_file, num_pre_clips, num_clips):
        super(TACoSDataset, self).__init__()
        self.cfg = cfg
        self.feat_file = feat_file
        self.num_pre_clips = num_pre_clips
        with open(ann_file,'r') as f:
            annos = json.load(f)

        if self.cfg.DATASETS.FEATS_TYPE == 'original':
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        self.annos = []
        logger = logging.getLogger("mmn.trainer")
        logger.info("Preparing data, please wait...")
        
        for vid, anno in annos.items():
            duration = anno['num_frames']/anno['fps']  # duration of the video
            # Produce annotations
            moments = []
            all_iou2d = []
            sentences = []
            for timestamp, sentence in zip(anno['timestamps'], anno['sentences']):
                if timestamp[0] < timestamp[1]:
                    moment = torch.Tensor([max(timestamp[0]/anno['fps'], 0), min(timestamp[1]/anno['fps'], duration)])
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
                    'moment': moments,  # N * 2
                    'iou2d': all_iou2d,  # N * 128*128
                    'sentence': sentences,   # list, len=N
                    'query': queries,  # padded query, N*word_len*C for LSTM and N*word_len for BERT
                    'wordlen': word_lens,  # size = N
                    'duration': duration
                }
            )

        if self.cfg.DATASETS.FEATS_TYPE == 'original':
            self.feats = video2feats(feat_file, annos.keys(), num_pre_clips, dataset_name="tacos")

        if self.cfg.DATASETS.FEATS_TYPE == 'CLIP':
            import pickle
            with open('./dataset/TACoS/TACoS_CLIP_clip_feats.pkl', 'rb') as f:
                self.feats = pickle.load(f)
            
            annos_vids = [e['vid'][:-4] for e in self.annos]
            in_annos_indices = [k for k in annos_vids if k not in self.feats.keys()]
            self.annos = [anno for anno in self.annos if anno['vid'][:-4] not in in_annos_indices]

    def __getitem__(self, idx):
        if self.cfg.DATASETS.FEATS_TYPE == 'CLIP':
            feat = self.feats[self.annos[idx]['vid'][:-4]]
        elif self.cfg.DATASETS.FEATS_TYPE == 'original':
            feat = get_vid_feat(self.feat_file, self.annos[idx]['vid'], self.num_pre_clips, dataset_name="tacos")

        query = self.annos[idx]['query']
        wordlen = self.annos[idx]['wordlen']
        iou2d = self.annos[idx]['iou2d']
        moment = self.annos[idx]['moment']
        return feat, query, wordlen, iou2d, moment, len(self.annos[idx]['sentence']), self.annos[idx]['duration'], idx

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
