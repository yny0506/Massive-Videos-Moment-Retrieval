# MVMR
MVMR: Evaluating Natural Language Video Localization Bias over Multiple Reliable Videos Pool


# MVMR dataset construction

## Pre-requisites

Before you run the scripts included in this repository, ensure you have the following:

- Pre-computed CLIP features for your dataset.

You can download the pre-computed CLIP features from the following link: [Download CLIP features](#) *(coming soon)*.

After downloading, place the CLIP features in the following directory structure:

```plaintext
- dataset
  - {dataset_name}
    - {dataset_name}_clip_feats.pkl
```

## Installation

To install the required libraries, run the following command:

```bash
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install torch pyyaml argparse ftfy regex tqdm transformers
pip install git+https://github.com/ShiYaya/CLIP
```

## Usage

First, you need to compute a sentence similarity matrix:
```bash
python compute_sent_sim.py --dataset_path ../dataset/Charades_STA/charades_test.json --dataset_name Charades_STA
```

Then, construct the mvmr dataset
```bash
python make_mvmr_dataset.py
```


# MVMR evaluation

## Pre-requisites

Before you run the scripts included in this repository, ensure you have the following:

1) Datasets
 - The pre-computed off-the-shelf video features for the [Charades-STA](https://prior.allenai.org/projects/charades), [ActivityNet](http://activity-net.org/download.html), [TACoS](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/tacos-multi-level-corpus)

 - You can download the off-the-shelf features from the following link: https://rochester.app.box.com/s/8znalh6y5e82oml2lr7to8s6ntab6mav (provided by [2D-TAN](https://github.com/microsoft/2D-TAN))

 - You can find the uploaded ground-truth annotation file of each dataset in the 'dataset' folder. These files are the ground-truth files of [2D-TAN](https://github.com/microsoft/2D-TAN).

2) The pre-trained weights of our RMMN
 - You can download the pre-trained weights for each dataset from the following link:
   
 (a) The RMMN trained with the Charades-STA: [RMMN_Charades-STA](http://milabfile.snu.ac.kr:16000/detecting-incongruity/pretrained_weights/best_charades_rmmn.pth)

 (b) The RMMN trained with the ActivityNet: [RMMN_ActivityNet](http://milabfile.snu.ac.kr:16000/detecting-incongruity/pretrained_weights/best_tacos_rmmn.pth)
 
 (c) The RMMN trained with the TACoS: [RMMN_TACoS](http://milabfile.snu.ac.kr:16000/detecting-incongruity/pretrained_weights/best_activitynet_rmmn.pth)

 - You should locate the pre-trained weights to the weight folder: 'outputs/rmmn_original_$datasetname'.

## Dependencies
 - Our code is developed on the [MMN](https://github.com/MCG-NJU/MMN.git), thus you should download following libraries:
 - yacs, h5py, terminaltables, tqdm, pytorch, transformers 


## Quick Start
 - You can start the evaluation code by running the bash file 'eval.sh', located in the scripts folder.

```
bash ./scripts/eval.sh 0
```


## Acknowledgments
Parts of the code are modified versions from the [EMScore](https://github.com/ShiYaya/emscore) and [MMN](https://github.com/MCG-NJU/MMN.git). We thank the original authors for their work.








