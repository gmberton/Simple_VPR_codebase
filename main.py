
import torch
import numpy as np
import torchvision.models
import pytorch_lightning as pl
from torchvision import transforms as tfm
from pytorch_metric_learning import losses
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

import utils
import parser
from datasets.test_dataset import TestDataset
from datasets.train_dataset import TrainDataset


class LightningModel(pl.LightningModule):
    def __init__(self, val_dataset, test_dataset, descriptors_dim=512, num_preds_to_save=0, save_only_wrong_preds=True):
        super().__init__()
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.num_preds_to_save = num_preds_to_save
        self.save_only_wrong_preds = save_only_wrong_preds
        # Use a pretrained model
        self.model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        # Change the output of the FC layer to the desired descriptors dimension
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, descriptors_dim)
        # Set the loss function
        self.loss_fn = losses.ContrastiveLoss(pos_margin=0, neg_margin=1)

    def forward(self, images):
        descriptors = self.model(images)
        return descriptors

    def configure_optimizers(self):
        optimizers = torch.optim.SGD(self.parameters(), lr=0.001, weight_decay=0.001, momentum=0.9)
        return optimizers

    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        loss = self.loss_fn(descriptors, labels)
        return loss

    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        images, labels = batch
        num_places, num_images_per_place, C, H, W = images.shape
        images = images.view(num_places * num_images_per_place, C, H, W)
        labels = labels.view(num_places * num_images_per_place)

        # Feed forward the batch to the model
        descriptors = self(images)  # Here we are calling the method forward that we defined above
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
        return self.inference_epoch_end(all_descriptors, self.val_dataset)

    def test_epoch_end(self, all_descriptors):
        return self.inference_epoch_end(all_descriptors, self.test_dataset, self.num_preds_to_save)

    def inference_epoch_end(self, all_descriptors, inference_dataset, num_preds_to_save=0):
        """all_descriptors contains database then queries descriptors"""
        all_descriptors = np.concatenate(all_descriptors)
        queries_descriptors = all_descriptors[inference_dataset.database_num : ]
        database_descriptors = all_descriptors[ : inference_dataset.database_num]

        recalls, recalls_str = utils.compute_recalls(
            inference_dataset, queries_descriptors, database_descriptors,
            trainer.logger.log_dir, num_preds_to_save, self.save_only_wrong_preds
        )
        print(recalls_str)
        self.log('R@1', recalls[0], prog_bar=False, logger=True)
        self.log('R@5', recalls[1], prog_bar=False, logger=True)

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


if __name__ == '__main__':
    args = parser.parse_arguments()

    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = get_datasets_and_dataloaders(args)
    model = LightningModel(val_dataset, test_dataset, args.descriptors_dim, args.num_preds_to_save, args.save_only_wrong_preds)
    
    # Model params saving using Pytorch Lightning. Save the best 3 models according to Recall@1
    checkpoint_cb = ModelCheckpoint(
        monitor='R@1',
        filename='_epoch({epoch:02d})_step({step:04d})_R@1[{val/R@1:.4f}]_R@5[{val/R@5:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        mode='max'
    )

    # Instantiate a trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        default_root_dir='./LOGS',  # Tensorflow can be used to viz
        num_sanity_val_steps=0,  # runs a validation step before stating training
        precision=16,  # we use half precision to reduce  memory usage
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,  # run validation every epoch
        callbacks=[checkpoint_cb],  # we only run the checkpointing callback (you can add more)
        reload_dataloaders_every_n_epochs=1,  # we reload the dataset to shuffle the order
        log_every_n_steps=20,
    )
    trainer.validate(model=model, dataloaders=val_loader)
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model=model, dataloaders=test_loader)

