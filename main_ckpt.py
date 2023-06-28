
import torch
import numpy as np
import torchvision.models
import pytorch_lightning as pl
from torchvision import transforms as tfm
from pytorch_metric_learning import losses
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import logging
from os.path import join

import utils
import parser
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset

# NEW
from models import helper

class LightningModel(pl.LightningModule):
    def __init__(self,
                #---- Datasets
                val_dataset,
                test_dataset,

                #---- Aggregator
                agg_arch='ConvAP', #CosPlace, NetVLAD, GeM
                agg_config={},

                #---- Other options (?)
                descriptors_dim=512,
                num_preds_to_save=0,
                save_only_wrong_preds=True,
                
                #---- Loss
                loss_name='MultiSimilarityLoss', 
                miner_name='MultiSimilarityMiner', 
                miner_margin=0.1
                # faiss_gpu=False
                ):
        super().__init__()
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.num_preds_to_save = num_preds_to_save
        self.save_only_wrong_preds = save_only_wrong_preds

        # Use a pretrained model
        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

        # Change the output of the FC layer to the desired descriptors dimension
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, descriptors_dim)

        # OLD self.loss_fn = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
        # OLD self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        # NEW
        # Set the loss function
        self.loss_name = loss_name

        self.loss_fn = utils.get_loss(loss_name)
        
        # Set the miner function
        self.miner_name = miner_name
        self.miner_margin = miner_margin

        self.miner = utils.get_miner(miner_name, miner_margin) if miner_name.lower() != "nominer" else None

        self.batch_acc = [] # we will keep track of the % of trivial pairs/triplets at the loss level 

        self.agg_name = agg_arch

        # Change the pooling layer
        if agg_arch.lower() != "avgpool":
            self.model.avgpool = helper.get_aggregator(agg_arch, agg_config)
            self.model.fc = None

        if agg_arch.lower() == "mixvpr":
          self.model.layer3 = None
          self.model.layer4 = None
        
        # TODO implement code for the margin

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        if self.agg_name.lower() != "mixvpr":
          x = self.model.layer3(x)
          x = self.model.layer4(x)
        x = self.model.avgpool(x)
        if self.agg_name.lower() == "avgpool":
          x = self.model.fc(x)
        return x

    def configure_optimizers(self):
        optimizers = torch.optim.SGD(self.parameters(), lr=0.001, weight_decay=0.001, momentum=0.9)
        return optimizers

    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        loss = self.loss_fn(descriptors, labels)
        return loss
    
    # NEW
    def loss_function(self, descriptors, labels):
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)
            
            # calculate the % of trivial pairs/triplets 
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined/nb_samples)

        else: # no online mining
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple: 
                # somes losses do the online mining inside (they don't need a miner objet), 
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class, 
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        # log it
        self.log('b_acc', sum(self.batch_acc) /
                len(self.batch_acc), prog_bar=True, logger=True)
        return loss

    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        images, labels = batch
        num_places, num_images_per_place, C, H, W = images.shape
        images = images.view(num_places * num_images_per_place, C, H, W)
        labels = labels.view(num_places * num_images_per_place)

        # Feed forward the batch to the model
        descriptors = self(images)                      # Here we are calling the method forward that we defined above
        loss = self.loss_function(descriptors, labels)  # Call the loss_function we defined above
        
        self.log('loss', loss.item(), logger=True)
        return {'loss': loss}

    # For validation and test, we iterate step by step over the validation set
    def inference_step(self, batch):
        images, _ = batch
        descriptors = self(images)
        return descriptors.cpu().numpy().astype(np.float32)

    def validation_step(self, batch, batch_idx):
        return self.inference_step(batch)

    def test_step(self, batch, batch_idx):
        return self.inference_step(batch)

    def validation_epoch_end(self, all_descriptors):
        return self.inference_epoch_end(all_descriptors, self.val_dataset, 'val')

    def test_epoch_end(self, all_descriptors):
        return self.inference_epoch_end(all_descriptors, self.test_dataset, 'test', self.num_preds_to_save)

    def inference_epoch_end(self, all_descriptors, inference_dataset, split, num_preds_to_save=0):
        """all_descriptors contains database then queries descriptors"""
        all_descriptors = np.concatenate(all_descriptors)
        queries_descriptors = all_descriptors[inference_dataset.database_num : ]
        database_descriptors = all_descriptors[ : inference_dataset.database_num]

        recalls, recalls_str = utils.compute_recalls(
            inference_dataset, queries_descriptors, database_descriptors,
            self.logger.log_dir, num_preds_to_save, self.save_only_wrong_preds
        )
        # print(recalls_str)
        logging.info(f"Epoch[{self.current_epoch:02d}]): " +
                      f"recalls: {recalls_str}")

        self.log(f'{split}/R@01', recalls[0], prog_bar=False, logger=True)
        self.log(f'{split}/R@05', recalls[1], prog_bar=False, logger=True)
        self.log(f'{split}/R@10', recalls[2], prog_bar=False, logger=True)
        self.log(f'{split}/R@20', recalls[3], prog_bar=False, logger=True)

def get_datasets_and_dataloaders(args):
    train_transform = tfm.Compose([
        tfm.RandAugment(num_ops=3),
        tfm.ToTensor(),
        tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = TrainDataset(
        dataset_folder=args.train_path,
        img_per_place=args.img_per_place,
        min_img_per_place=args.min_img_per_place,
        transform=train_transform
    )
    val_dataset = TestDataset(dataset_folder=args.val_path)
    test_dataset = TestDataset(dataset_folder=args.test_path)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader

# NEW METHOD

def default_agg_config(agg_arch='ConvAP'):
    agg_config = {}
    if 'cosplace' in agg_arch.lower():
        agg_config={'in_dim': 512,          # OLD 2048
                    'out_dim': 512}         # OLD 2048
    elif 'gem' in agg_arch.lower():
        agg_config={'p': 3}
    elif 'convap' in agg_arch.lower():
        agg_config={'in_channels': 512,     # OLD 2048
                    'out_channels': 512}    # OLD 2048
    elif 'mixvpr' in agg_arch.lower():
        agg_config={'in_channels' : 128,
                    'in_h' : 28,
                    'in_w' : 28,
                    'out_channels' : 128,
                    'mix_depth' : 4,
                    'mlp_ratio' : 1,
                    'out_rows' : 4} # the output dim will be (out_rows * out_channels)
    return agg_config

if __name__ == '__main__':
    args = parser.parse_arguments()
    utils.setup_logging(join('logs', 'lightning_logs', args.exp_name), console='info')

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = get_datasets_and_dataloaders(args)

    # Define a personalized agg_config if you want
    agg_config = default_agg_config(args.agg_arch)

    model = LightningModel(
            val_dataset, test_dataset,                          # Datasets
            args.agg_arch, agg_config,                          # Aggregator layer
            args.descriptors_dim,                               # Architecture
            args.num_preds_to_save, args.save_only_wrong_preds, # Visualizations parameters
            args.loss_name,                                     # Loss
            args.miner_name, args.miner_margin                  # Miner
        )
    
    # Model params saving using Pytorch Lightning. Save the best 3 models according to Recall@1
    checkpoint_cb = ModelCheckpoint(
        monitor='val/R@01',
        filename='_epoch({epoch:02d})_R@1[{val/R@01:.4f}]_R@5[{val/R@05:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=False,
        save_top_k=1,
        save_last=True,
        mode='max'
    )

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/", version=args.exp_name)

    # Instantiate a trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        default_root_dir='./logs',              # Tensorflow can be used to viz
        num_sanity_val_steps=0,                 # runs a validation step before stating training
        precision=16,                           # we use half precision to reduce  memory usage
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,              # run validation every epoch
        logger=tb_logger,                       # log through tensorboard
        callbacks=[checkpoint_cb],              # we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1,    # we reload the dataset to shuffle the order
        log_every_n_steps=20,
    )

    trainer.validate(model=model, dataloaders=val_loader, ckpt_path=args.ckpt_path)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.ckpt_path)
    trainer.test(model=model, dataloaders=test_loader, ckpt_path="best")

