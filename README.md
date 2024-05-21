# MVMR: A New Framework for Evaluating Faithfulness of Video Moment Retrieval against Multiple Distractors

arXiv Link [arXiv](https://arxiv.org/pdf/2309.16701v1.pdf)

This github page includes (1) MVMR datasets construction code, and (2) MVMR evaluation code (Our CroCs training code will also be added ASAP)

Although this repository includes the MVMR datasets construction code, you do not need to run them since it also contains the constructed three MVMR datasets. 


## Dependencies
```bash
conda create -n mvmr python=3.9
conda activate mvmr
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install git+https://github.com/ShiYaya/CLIP
pip install yacs h5py opencv-python terminaltables transformers==4.29.2 safetensors==0.3.0
```

We recommend creating a new conda env since there may be library conflicts.


# MVMR dataset construction

## Pre-requisites

Before you run the scripts included in this repository, ensure you have the following:

- Pre-computed CLIP features for your dataset.

You can download the pre-computed CLIP features from the following link: [Download CLIP features](#) *(coming soon)*.

After downloading, place the CLIP features in the following directory structure:

```plaintext
- ../MVMR/dataset
  - {dataset_name}
    - {dataset_name}_clip_feats.pkl
```

## Usage

First, you need to compute a sentence similarity matrix:
```bash
python compute_sent_sim.py --dataset_path ../MVMR/dataset/Charades_STA/charades_test.json --dataset_name Charades_STA
```

Then, construct the MVMR dataset
```bash
python make_mvmr_dataset.py
```


# MVMR evaluation

## Pre-requisites

Before you run the scripts included in this repository, ensure you have the following:

1) Datasets
 - The pre-computed off-the-shelf video features for the [Charades-STA](https://prior.allenai.org/projects/charades), [ActivityNet](http://activity-net.org/download.html), [TACoS](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/tacos-multi-level-corpus)

 - You can download the off-the-shelf features from the following link: [Features](https://rochester.app.box.com/s/8znalh6y5e82oml2lr7to8s6ntab6mav) (provided by [2D-TAN](https://github.com/microsoft/2D-TAN))

 - You can find the uploaded ground-truth annotation file of each dataset in the 'dataset' folder. These files are the ground-truth files of [2D-TAN](https://github.com/microsoft/2D-TAN).

2) The pre-trained weights of our CroCs
 - You can download the pre-trained weights for each dataset from the following link:
   
 - (a) The CroCs trained with the Charades-STA: [CroCs_Charades-STA](https://drive.google.com/file/d/16gl68U0M0TR-r-F8mjOlfy8l9ZmmM-G0/view?usp=share_link)

 - (b) The CroCs trained with the ActivityNet: [CroCs_ActivityNet](https://drive.google.com/file/d/1KceHtTq8XOXRrQwamWtJPPaqTZK6LnxY/view?usp=share_link)
 
 - (c) The CroCs trained with the TACoS: [CroCs_TACoS](https://drive.google.com/file/d/1fT6e4uWwLQnv5Irs1InevhLC4X-UFlv9/view?usp=share_link)

 - You should locate the pre-trained weights to the weight folder: 'outputs/crocs_original_$datasetname'.
   

## Quick Start
 - You can start the evaluation code by running the python file 'test_net_mvmr.py'.

 - Please remember that you should download the pre-trained weights and locate them using "--ckpt" parameter.

 - This github page already includes "the ground-truth annotation file of each dataset (./MVMR/dataset)" and "the constructed three MVMR datasets (./MVMR/crocs/mvmr/samples)".

 - You can find the MVMR datasets ("$datasetname_test_mvmr.json") in "./MVMR/crocs/mvmr/samples".

```bash
python test_net_mvmr.py --config-file configs/crocs_original_charades.yaml --ckpt outputs/crocs_original_charades/best_charades_crocs.pth --sample_indices_info crocs/mvmr/samples/charades_test_mvmr.json
```


## Acknowledgments
Parts of the code are modified versions from the [EMScore](https://github.com/ShiYaya/emscore) and [MMN](https://github.com/MCG-NJU/MMN.git). We thank the original authors for their work.








