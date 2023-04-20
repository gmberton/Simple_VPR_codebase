# Simple_VPR_codebase

This repository serves as a starting point to implement a VPR pipeline. It allows you to train a simple
ResNet-18 on the GSV dataset. It relies on the [pytorch_metric_learning](https://kevinmusgrave.github.io/pytorch-metric-learning/)
library.

## Run an experiment
You can choose to validate/test on sf_xs or tokyo_xs.

```bash
python main.py --train_path /path/to/datasets/gsv_xs --val_path /path/to/datasets/tokyo_xs/test --test_path /path/to/datasets/tokyo_xs/test
```
## Test on a pretrained model
At every training the best 3 models, chosen by evaluating the R@1, are saved in the **./LOGS/lightning_logs/version_x/checkpoints/** folder in 3 distinct **.ckpt** files.  

We can use these checkpoint files to test the model against a new test dataset.  

To test the pretrained model on new test dataset you can simply add the **--ckpt_path** arg, indicating the path to a .ckpt file, when executing the main.  
This will **not** trigger the training and the validation phases but will only test the model on the dataset indicated in the --test_path arg.  

```bash
python main.py --ckpt_path ./LOGS/lightning_logs/version_x/checkpoints/_epoch\(xx\)_step\(xxxxx\)_R@1[0.0000]_R@5[0.0000].ckpt \
  --train_path /content/gsv_xs --val_path /content/sf_xs/val --test_path /content/tokyo_xs/test
```

## Select Aggregator Layer and Loss
It is possible to pass specific args to select the Aggregator Layer and the Loss we want to use to train the model.  

```bash
python main.py --train_path /content/gsv_xs --val_path /content/sf_xs/val --test_path /content/sf_xs/test \
  --num_workers 2 --loss_name MultiSimilarityLoss --agg_arch ConvAP
```
It is possible to choose among multiple types of Aggregator Layers and Losses, for a complete list check:

- the file [./utils/losses.py](https://github.com/danielemansillo/Simple_VPR_codebase/blob/main/utils/losses.py) for the Losses
- the folder [./models/aggregators/](https://github.com/danielemansillo/Simple_VPR_codebase/tree/main/models/aggregators) for the Aggregator Layers


## Usage on Colab

We provide the notebook `colab_example.ipynb`.
It shows you how to attach your GDrive file system to Colab, unzip the datasets, install packages and run your first experiment.
