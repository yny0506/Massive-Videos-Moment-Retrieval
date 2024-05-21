#!/bin/bash
#SBATCH --job-name  eval
#SBATCH --time      4-00:00:00
#SBATCH -c          4
#SBATCH --mem       100G
#SBATCH --gpus      1


source activate mvmr
ml cudnn/8.2.0.53-11.3-cuda11.3


dataset_name_idx=$1
master_port=27512
additional_name="mvmr_test"


model_type="crocs"

if [ $dataset_name_idx = 0 ]
then
    dataset_name="charades"
elif [ $dataset_name_idx = 1 ]
then
    dataset_name="activitynet"
elif [ $dataset_name_idx = 2 ]
then
    dataset_name="tacos"
fi

config=$model_type"_original_"$dataset_name
config_file=configs/$config\.yaml
weight_dir=outputs/$config
weight_file=outputs/$config/best_$dataset_name\_$model_type\.pth

sample_indices_info=crocs/mvmr/samples/$dataset_name\_test_mvmr.json

echo $config
echo $additional_name

batch_size=12

# set your gpu id
gpus=0
# number of gpus
gpun=1
# please modify it with different value (e.g., 127.0.0.2, 29502) when you run multi mmn task on the same machine
master_addr=127.0.0.2

CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch \
--nproc_per_node=$gpun --master_addr $master_addr --master_port $master_port \
test_net_mvmr.py --config_file $config_file --ckpt $weight_file --sample_indices_info $sample_indices_info --additional_name $additional_name OUTPUT_DIR $weight_dir TEST.BATCH_SIZE $batch_size

