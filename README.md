<div align="center">

# LUNG SEGMENTATION
</div>

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/huynhspm/Lung-Segmentation
cd Lung-Segmentation

# [OPTIONAL] create conda environment
conda create -n lung python=3.10
conda activate lung

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

## Download Data
```bash
cd data
```

[LIDC data](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254&fbclid=IwAR1vDkrpq0IJN8KwPT2Fft1GJ4bFPiMqXp4p08eEfOaUYofS-88pnNF_Z7g)

```bash
# Data Preprocessing with repository
git clone https://github.com/jaeho3690 LIDC-IDRI-Preprocessing
```


## How to run

Train model with default configuration

```bash
# before training, set env WANDB_API_KEY to log with wandb logger
export WANDB_API_KEY = ${WANDB_API_KEY}

# train on one GPU
python src/train.py trainer=gpu logger=wandb

# train on multi GPU
python src/train.py trainer=ddp trainer.devices=4 logger=wandb
