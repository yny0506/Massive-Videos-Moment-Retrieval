import torch
import json
import pickle
import random
from tqdm import tqdm
from torch import nn

from emscore.emscore.scorer import EMScorer
from emscore.emscore.utils import get_idf_dict
import clip

import yaml
import argparse


def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def parse_args():
    parser = argparse.ArgumentParser(description="Script to process video datasets.")
    parser.add_argument('--config', default='config/config.yaml', type=str, help='Path to the config file.')
    parser.add_argument('--dataset', default='charades', choices=['charades', 'activity_net', 'tacos'], help='Dataset to use.')
    return parser.parse_args()

def main():
 # Setup and configurations
    args = parse_args()
    config = load_config(args.config)
    dataset_config = config['datasets'][args.dataset]
    annos = dataset_config['annos']
    sim_mat = dataset_config['sim_mat']
    clip_dict = dataset_config['clip_dict']
    output_file_name = dataset_config['output_file_name']
    num_clips = dataset_config['num_clips'] #parameter for average pooling
    max_pos_moments = dataset_config['max_pos_moments']
    max_num_retrieval_pool = dataset_config['max_num_retrieval_pool']
    positive_th = config['thresholds']['positive']
    negative_th = config['thresholds']['negative']

    # Load annotation files
    with open(annos, 'r') as file:
        video_dict = json.load(file)
    sentences = []
    moments = []
    video_files = []
    start_indices = {}
    end_indices = {}
    current_index = 0
    for video_file, video_info in video_dict.items():
        start_indices[video_file] = current_index  
        for sentence, moment in zip(video_info['sentences'], video_info['timestamps']):
            sentences.append(sentence)
            moments.append(moment)
            video_files.append(video_file)
            current_index += 1 
        end_indices[video_file] = current_index - 1
    print("Loaded json file!")

    # Load dataset wise simililarity matrix
    similarity_matrix = torch.load(sim_mat).cpu()

    # Load EMScorer
    with open(clip_dict, 'rb') as f:
        tensor_dict = pickle.load(f)
    avgpool = nn.AdaptiveAvgPool1d(num_clips)
    for i in tensor_dict.keys():
        tensor_dict[i] = avgpool(tensor_dict[i].transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)

    metric = EMScorer(tensor_dict)
    idf = get_idf_dict(sentences, clip.tokenize)
    # Compute query-video threshold scores
    vids = []
    for v in video_files:
        vids.append(v)
    results_p = metric.score(cands=sentences, vids=vids,refs=[], idf=idf)
    set_video_files = list(set(video_files))
    q_v_similarity_matrix = torch.zeros(len(sentences), len(set_video_files)) #query * video
    for i, f in enumerate(set_video_files):
        for j in range(len(sentences)):
            s = start_indices[f]
            e = end_indices[f] + 1
            q_v_similarity_matrix[j][i] = torch.max(similarity_matrix[j][s:e], dim=-1)[0]
    neg_vids_idx_list = []

    neg_sentences = []
    for i in range(len(sentences)):
        neg_vids_idx = torch.where(q_v_similarity_matrix[i] < negative_th)[0].tolist()
        if len(neg_vids_idx) < 1:
            continue
        neg_sentences.append(sentences[i])
        neg_vids_idx_list.append(random.sample(neg_vids_idx,1))
    neg_vids_list = [set_video_files[i[0]] for i in neg_vids_idx_list]
    n_vids =[]
    for v in neg_vids_list:
        n_vids.append(v)
    results_n = metric.score(cands=neg_sentences, vids=n_vids,refs=[], idf=idf)

    em_pos_th = results_p['EMScore(X,V)']['full_F'].mean()
    em_neg_th = results_n['EMScore(X,V)']['full_F'].mean()

    for k in tqdm(range(len(sentences))):
        #simCSE filtered
        pos_candi_indices = torch.where(similarity_matrix[k] > positive_th)[0]
        neg_candi_indices = torch.where(q_v_similarity_matrix[k] < negative_th)[0]
        
        #positive filtering with emscore
        p_candi_list = pos_candi_indices.tolist()
        pos_dict = {}
        for i in p_candi_list :
            pos_dict[video_files[i]] = i
        p_candi_list  = list(pos_dict.values())

        random.shuffle(p_candi_list)
        if len(p_candi_list) > 20:
            p_candi_list = p_candi_list[:20]
        p_candi_list = [i for i in p_candi_list if video_files[i] != video_files[k]]
        pos_vids_list = [video_files[i] for i in p_candi_list]
        if len(pos_vids_list) > 0:
            results = metric.score(cands=[sentences[k] for i in range(len(pos_vids_list))], vids=pos_vids_list ,refs=[], idf=False)['EMScore(X,V)']['full_F']
            if not (results > em_pos_th).sum():
                pos_final = []
            else:
                pos_final = torch.tensor(p_candi_list)[(results > em_pos_th).nonzero().squeeze(dim=1)].tolist()
                if len(pos_final) > max_pos_moments:
                    pos_final = random.sample(torch.tensor(p_candi_list)[(results > em_pos_th).nonzero().squeeze(dim=1)].tolist(), 4)
        else:
            pos_final = []
        
        #negative filtering with emsocre
        neg_final = []
        neg_final_vids=[]
        num_neg_final = max_num_retrieval_pool - len(pos_final) - 1
        if num_neg_final > 0:
            neg_candi_list = neg_candi_indices.tolist()
            # Create a unique list of negative candidate videos
            neg_videos = list(set(set_video_files[i] for i in neg_candi_list))
            random.shuffle(neg_videos)
            neg_videos_idx = torch.arange(len(neg_videos))
            start_i = 0
            while len(neg_final) < num_neg_final:
                end_i = start_i + min(256, len(neg_videos) - start_i)       
                # Break the loop if there are no more videos to process
                if start_i >= len(neg_videos):
                    break
                neg_batch_vid_list = neg_videos[start_i:end_i]
                # Score the current batch of negative videos
                repeated_sentences = [sentences[k]] * len(neg_batch_vid_list)
                results = metric.score(cands=repeated_sentences, vids=neg_batch_vid_list, refs=[], idf=idf)['EMScore(X,V)']['full_F']
                # Filter the batch based on scores and extend the final list
                neg_batch_filtered = torch.tensor(neg_videos_idx[start_i:end_i])[(results < em_neg_th).nonzero().squeeze(dim=1)].tolist()
                neg_final.extend(neg_batch_filtered)
                # Update the start index for the next batch
                start_i = end_i
            # Trim neg_final to the desired length
            neg_final = neg_final[:num_neg_final]
            # Issue a warning if not enough negative samples were collected
            if len(neg_final) < num_neg_final:
                print(f"Warning: Only {len(neg_final)}/{num_neg_final} negative samples were collected about query {k}.")
        else:
            print(f"Warning: Query {k} has no negative samples available.")

        retrieval_pool = []
        pos_moments = []
        retrieval_pool.append(video_files[k])
        for i in pos_final:
            pos_moments.append((video_files[i], sentences[i], moments[i]))
            retrieval_pool.append(video_files[i])
        for i in neg_final:
            retrieval_pool.append(neg_videos[i])
        # Save results
        if k == start_indices[video_files[k]]:
            pos_moments_per_video=[]
            retrieval_pool_per_video=[]
        if k <= end_indices[video_files[k]]:
            pos_moments_per_video.append(pos_moments)
            retrieval_pool_per_video.append(retrieval_pool)
        if k == end_indices[video_files[k]]:
            video_dict[video_files[k]]["pos_moments"] = pos_moments_per_video
            video_dict[video_files[k]]["retrieval_pool"] = retrieval_pool_per_video
        if k%50 == 0:
            with open(output_file_name, "w") as file:
                json.dump(video_dict, file) 

    with open(output_file_name, "w") as file:
        json.dump(video_dict, file) 

if __name__ == "__main__":
    main()