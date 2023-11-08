# MVMR dataset construction

## Pre-requisites

Before you run the scripts included in this repository, ensure you have the following:

- Pre-computed CLIP features for your dataset.

You can download the pre-computed CLIP features from the following link: [Download CLIP features](#https://drive.google.com/drive/u/0/folders/16BovXpyh7eX7xloQzF0Hy9-_zRUvHVlv) *(now only for charades, coming soon)*.

After downloading, place the CLIP features in the following directory structure:

```plaintext
- ../RMMN/dataset
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
python compute_sent_sim.py --dataset_path ../RMMN/dataset/Charades_STA/charades_test.json --dataset_name Charades_STA
```

Then, construct the mvmr dataset
```bash
python make_mvmr_dataset.py
```

## Acknowledgments

Parts of the code are modified versions from the [EMScore](https://github.com/ShiYaya/emscore). We thank the original authors for their work.