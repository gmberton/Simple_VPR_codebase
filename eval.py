
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning import loggers as pl_loggers
from os.path import join

import utils
import parser
from datasets.test_dataset import TestDataset
from main import LightningModel

def get_datasets_and_dataloaders(args):
    val_dataset = TestDataset(dataset_folder=args.val_path)
    test_dataset = TestDataset(dataset_folder=args.test_path)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)
    return val_dataset, test_dataset, val_loader, test_loader


if __name__ == '__main__':
    args = parser.parse_arguments()
    utils.setup_logging(join('logs', args.exp_name), console='info')

    val_dataset, test_dataset, val_loader, test_loader = get_datasets_and_dataloaders(args)
    model = LightningModel(val_dataset, test_dataset, args.descriptors_dim, args.num_preds_to_save, args.save_only_wrong_preds)
    
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/", version=args.exp_name)
    # Instantiate a trainer
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=[0],
        default_root_dir='./logs',  # Tensorflow can be used to viz
        num_sanity_val_steps=0,  # runs a validation step before stating training
        precision=16,  # we use half precision to reduce  memory usage
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,  # run validation every epoch
        logger=tb_logger, # log through tensorboard
        reload_dataloaders_every_n_epochs=1,  # we reload the dataset to shuffle the order
        log_every_n_steps=20,
    )
    trainer.test(model=model, dataloaders=test_loader, ckpt_path=args.checkpoint)
