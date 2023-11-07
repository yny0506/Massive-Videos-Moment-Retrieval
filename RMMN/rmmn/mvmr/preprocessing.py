import numpy as np

def preprocess_mvmr_dataset(sample_indices_info, data_loaders, cfg):
        num_samples = int(cfg.MVMR.NUM_SAMPLES)
        vid2idx = {data_loaders.dataset.get_vid(idx):idx for idx in range(len(data_loaders.dataset.annos))}
        videos_sample_indices = []
        videos_removed_data = []

        for i, e in enumerate(data_loaders.dataset.annos):
            queries_sample_indices = []
            queries_removed_data = []

            vid_str = e['vid'].split('.')[0] if cfg.DATASETS.NAME=='tacos' else e['vid']

            if e['vid'].split('.')[0] in sample_indices_info.keys():
                queries_samples_vid = sample_indices_info[vid_str]['retrieval_pool']

                for j, query_samples_vid in enumerate(queries_samples_vid):
                    query_sample_indices = []

                    for sample_vid in query_samples_vid:
                        if cfg.DATASETS.NAME == 'tacos':
                            sample_vid = sample_vid+'.avi'
                        query_sample_indices.append(vid2idx[sample_vid])

                    if len(query_sample_indices) != num_samples:
                        queries_removed_data.append(j)
                    else:
                        query_sample_indices = np.sort(query_sample_indices).tolist()
                        queries_sample_indices.append(query_sample_indices)

                videos_sample_indices.append(queries_sample_indices)
                videos_removed_data.append(queries_removed_data)

        return videos_sample_indices, videos_removed_data