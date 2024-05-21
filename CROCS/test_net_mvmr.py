import argparse
import os
import torch
import json
import numpy as np

from crocs.config import cfg
from crocs.data import make_data_loader
from crocs.modeling import build_model
from crocs.utils.checkpoint import MmnCheckpointer
from crocs.utils.comm import synchronize, get_rank
from crocs.utils.logger import setup_logger
from crocs.mvmr.inference_mmn import inference
from crocs.mvmr.preprocessing import preprocess_mvmr_dataset

def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser(description="Mutual Matching Network")
    parser.add_argument(
        "--config_file",
        default="configs/pool_128x128_k5l8_tacos.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument("--additional_name", type=str, default='')
    parser.add_argument("--sample_indices_info", type=str, default='')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    additional_name = args.additional_name
    shuffle = False
    is_mvmr = True
    sample_indices_info = args.sample_indices_info

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("mmn", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)
    
    model = build_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    model.eval()

    output_dir = cfg.OUTPUT_DIR
    checkpointer = MmnCheckpointer(cfg, model, save_dir=output_dir)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)

    dataset_names = cfg.DATASETS.TEST
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed, shuffle=shuffle)[0]
    
    with open(sample_indices_info, 'r') as f:
        sample_indices_info = json.load(f)

    videos_sample_indices, videos_removed_data = preprocess_mvmr_dataset(sample_indices_info, data_loaders_val, cfg)
    
    inference_args = {'cfg':cfg,
                    'model':model,
                    'data_loader':data_loaders_val, 
                    'dataset_name':dataset_names, 
                    'nms_thresh':cfg.TEST.NMS_THRESH, 
                    'device':'cuda', 
                    'is_mvmr':is_mvmr,
                    'num_samples':int(cfg.MVMR.NUM_SAMPLES),
                    'sample_indices':videos_sample_indices,
                    'removed_data':videos_removed_data,
                    'additional_name':additional_name,
                    'sample_indices_info':sample_indices_info}

    _ = inference(**inference_args)
    synchronize()

if __name__ == "__main__":
    main()
