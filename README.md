# Simple_VPR_codebase

This repository serves as a starting point to implement a VPR pipeline. It allows you to train a simple
ResNet-18 on the GSV dataset. It relies on the [pytorch_metric_learning](https://kevinmusgrave.github.io/pytorch-metric-learning/)
library.

## Download datasets
NB: if you are using Colab, skip this section

The following script:

> python download_datasets.py

allows you to download GSV_xs, SF_xs, tokyo_xs, which are reduced version of the GSVCities, SF-XL, Tokyo247 datasets respectively.

## Install dependencies
NB: if you are using Colab, skip this section

You can install the required packages by running
> pip install -r requirements.txt


## Run an experiment
You can choose to validate/test on sf_xs or tokyo_xs.


>python main.py --train_path /path/to/datasets/gsv_xs --val_path /path/to/datasets/tokyo_xs/test --test_path /path/to/datasets/tokyo_xs/test


## Usage on Colab

We provide you with the notebook `colab_example.ipynb`.
It shows you how to attach your GDrive file system to Colab, unzip the datasets, install packages and run your first experiment.

NB: BEFORE running this notebook, you must copy the datasets zip into your GDrive. You can use the [link](https://drive.google.com/drive/folders/1Ucy9JONT26EjDAjIJFhuL9qeLxgSZKmf?usp=sharing) that we provided and simply click 'create a copy'. Make sure that you have enough space (roughly 8 GBs)

NB^2: you can ignore the dataset `robotcar_one_every_2m`.