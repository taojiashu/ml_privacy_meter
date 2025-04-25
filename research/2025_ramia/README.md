# Range Membership Inference Code

This repository contains code to reproduce the results in our paper:
"Range Membership Inference Attacks" https://arxiv.org/pdf/2408.05131 by Jiashu Tao and Reza Shokri.
SaTML 2025

## Repository setup
The code provided is refactored to reuse as many existing functions in the open-source [Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter) as possible. However, since the CIFAR-10 models are written in JAX, which is not very compatible to the PyTorch pipeline of Privacy Meter, we provide our script (adapted from [LiRA]((https://github.com/carlini/privacy/tree/better-mi/research/mi_lira_2021))) that computes the model signals in `mi_lira_2021/inference.py`, and the attack in a jupyter notebook `cifar10_wideresnet_ramia.ipynb`.

In our experiments with CelebA, the task is a multi-class classification problem where the output is 40-dimensional. Since there would be an entire seperate logic to handle this dataset, training script, signal computation and aggregation, we also simply provide the demo notebook `celeba_ramia.ipynb` one can use to run the same attack.

## Installation Instructions
For all experiments except CIFAR-10, please refer to the installation instructions of the [Privacy Meter](https://github.com/privacytrustlab/ml_privacy_meter). To run the CIFAR-10 experiments, additional JAX libraries are needed. Please refer to the CIFAR-10 section in the "Training Models" section below for more details.

## Training models
### Purchase-100 and AG News
Privacy Meter automatically handles model training and data splitting. No action is needed for these two experiments.

### CIFAR-10
Since Privacy Meter handle model training automatically, we only provide details on how to train the WideResNets we used in our CIFAR-10 experiments. This is identical to LiRA. Note that JAX and objax need to be installed to train the JAX models. Please refer to `mi_lira_2021/env.yaml` or the [original LiRA repo](https://github.com/carlini/privacy/tree/better-mi/research/mi_lira_2021) for the environment details and installation instructions.
```bash
cd mi_lira_2021
bash scripts/train_demo.sh
```
This script will train 16 WideResNets. To compute the signals on the range queries, do
```bash
python3 inference.py --logdir=exp/cifar10/
```

The `mi_lira_2021/inference.py` is modified to apply transformations to all point queries before computing signals from all transformed samples. The subsequent steps are in `cifar10_wideresnet_ramia.ipynb`.

### CelebA
For CelebA, model training is included in `celeba_ramia.ipynb`. The model is defined in `facial_attribute_cnn.py`.

## Running RaMIA
### Purchase-100
For Purchase-100 experiments with missing data, we use the Privacy Meter's auditing with RaMIA:
```bash
# clone the Privacy Meter repo to local if you have not done so
git clone https://github.com/privacytrustlab/ml_privacy_meter.git
cd ml_privacy_meter
python run_range_mia.py --cf configs/ramia/purchase_missingvalues.yaml
```

### CelebA
For CelebA experiments, we also compute all loss values in `celeba_ramia.ipynb` first, before computing the RMIA signals. Note that in multi-class classification, the loss is averaged over all 40 dimensions when we compute the RMIA signal for each input image. This is in line with the common evaluation metric: averaged accuracy over all attributes for all data.

### CIFAR-10
For CIFAR-10 experiments, the attack can be found in `cifar10_wideresnet_ramia.ipynb`. In particular, we first compute the loss values for all samples before we compute the RMIA signals. We then perform the trimming as mentioned in the paper.

### AG News
For AG News experiments with word replacement, we use the Privacy Meter's auditing with RaMIA:
```bash
# clone the Privacy Meter repo to local if you have not done so
git clone https://github.com/privacytrustlab/ml_privacy_meter.git
cd ml_privacy_meter
python run_range_mia.py --cf configs/ramia/agnews.yaml
```