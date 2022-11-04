import mlflow
import pytorch_lightning as pl
import sys
import configparser
from torch.cuda import is_available as cuda_available
from pytorch_lightning.loggers import MLFlowLogger

from project.DataModule import DataModule
from project.Network import Network


def train(config):

    # initialise network and datamodule
    net = Network(data_dir=data_dir)
    dm = DataModule(data_dir=data_dir, batch_size=4)

    mlf_logger = MLFlowLogger(
        experiment_name=config['project']['name'],
    )

    #  start logged run-
    mlflow.pytorch.autolog()
    with mlflow.start_run() as run:
        trainer = pl.Trainer(logger=mlf_logger,
                             gpus=-1 if cuda_available() else None,
                             accelerator='ddp'
                             )
        trainer.fit(net, dm)


if __name__ == '__main__':

    if len(sys.argv) > 0:
        config_path = sys.argv[1]
    else:
        config_path = 'config/config.cfg'

    config = configparser.ConfigParser()
    config.read(config_path)
    train(config)
