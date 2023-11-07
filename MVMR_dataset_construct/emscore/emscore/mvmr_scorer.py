import torch
import clip
from PIL import Image
import json
import cv2
import numpy as np
from tqdm import tqdm
import math
import time
from collections import defaultdict


from .mvmr_utils import em_cos_score, get_idf_dict

class EMScorer:
    """
    EMScore for MVMR(Massive Video Moments Retrieval)
    model to get every moments' EMscore

    """
    

    def __init__(self, vid_feat_cache=None, device=None,):

        self.vid_feat_cache = vid_feat_cache

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def score(self, cands, vids=None, verbose=False, batch_size=64, nthreads=4, idf=True, return_matched_idx=False, num_clips = 16):
        """
        Args:
            - :param: `cands` (list of str): candidate sentences
            - :param: `refs` (list of list of str): reference sentences

        Return:
            - :param: `(P, R, F)`: each is of shape (N); N = number of input
                        candidate reference pairs. if returning hashcode, the
                        output will be ((P, R, F), hashcode). If a candidate have 
                        multiple references, the returned score of this candidate is 
                        the *best* score among all references.
        """
        
        model, preprocess = clip.load("ViT-B/32", device=self.device)
        self._model = model
        self._tokenizer = clip.tokenize
        self._image_preprocess = preprocess

    
        ori_cands = cands
        # if reference are avaliable, and there are multiple references for each candidata caption
        # mvmr에서는 안쓸듯.

        if not idf:
            idf_dict = defaultdict(lambda: 1.0)
        elif isinstance(idf, dict):
            if verbose:
                print("using predefined IDF dict...")
            idf_dict = idf
        else:
            if verbose:
                print("preparing IDF dict...")
            start = time.perf_counter()
            idf_corpus = cands
            idf_dict = get_idf_dict(idf_corpus, self._tokenizer, nthreads=nthreads)
            # max token_id are eos token id
            # set idf of eos token are mean idf value
            idf_dict[max(list(idf_dict.keys()))] = sum(list(idf_dict.values()))/len(list(idf_dict.values()))
            if verbose:
                print("done in {:.2f} seconds".format(time.perf_counter() - start))

        
        if verbose:
            print("calculating EMScore scores...")
            time_start = time.perf_counter()

        results = em_cos_score(
            num_clips,
            self._model,
            cands,
            ori_cands,
            vids,
            self.vid_feat_cache,
            self._tokenizer,
            idf_dict,
            self._image_preprocess,
            verbose=verbose,
            device=self.device,
            batch_size=batch_size,
            return_matched_idx=return_matched_idx
        )
        
        final_results = {}
        
        if vids:
            vid_all_local_preds  = results['vid_result']['figr']
            vid_all_global_preds = results['vid_result']['cogr']
            vid_P, vid_R, vid_F  = vid_all_local_preds[..., 0], vid_all_local_preds[..., 1], vid_all_local_preds[..., 2]   # P, R, F

            vid_results = {}
            vid_results['figr_P'] = vid_P
            vid_results['figr_R'] = vid_R
            vid_results['figr_F'] = vid_F
            vid_results['cogr'] = vid_all_global_preds
            vid_results['full_P'] = (vid_results['figr_P'] + vid_results['cogr'])/2
            vid_results['full_R'] = (vid_results['figr_R'] + vid_results['cogr'])/2
            vid_results['full_F'] = (vid_results['figr_F'] + vid_results['cogr'])/2
            # vid_results['vid_matched_indices'] = results['vid_result']['matched_indices']
            final_results['EMScore(X,V)'] = vid_results
        

        if verbose:
            time_diff = time.perf_counter() - time_start
            print(f"done in {time_diff:.2f} seconds, {len(cands) / time_diff:.2f} sentences/sec")

        return final_results


