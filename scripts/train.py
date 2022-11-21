import mlflow
import pytorch_lightning as pl
import sys
import configparser
from torch.cuda import is_available as cuda_available
from pytorch_lightning.loggers import MLFlowLogger

from project.DataModule import DataModule
from project.Network import Network


def train(config):
    # config
    num_workers = 4
    gpus = 4
    test_batch = -1  # useful to test a small subset of data to ensure things are working, set to 0 to use all data

    # initialise network and datamodule
    net = Network(data_dir=data_dir)
    dm = DataModule(data_dir=data_dir, batch_size=4, test_batch=test_batch, num_workers=num_workers)

    mlf_logger = MLFlowLogger(
        experiment_name=config['project']['name'],
    )

    #  start logged run-
    mlflow.pytorch.autolog()
    with mlflow.start_run() as run:
        trainer = pl.Trainer(
            logger=mlf_logger,
            auto_select_gpus=True,
            precision=16 if cuda_available() else 32,
            devices=gpus if cuda_available() else None,
            accelerator='gpu' if cuda_available() else None,
            log_every_n_steps=1,
            max_epochs=epochs,
            strategy="ddp" if cuda_available() else None,
            num_sanity_val_steps=0,
        )
        trainer.fit(net, dm)

    logger.info('Training complete')

if __name__ == '__main__':

    if len(sys.argv) > 0:
        config_path = sys.argv[1]
    else:
        config_path = 'config/config.cfg'

    config = configparser.ConfigParser()
    config.read(config_path)
    train(config)
