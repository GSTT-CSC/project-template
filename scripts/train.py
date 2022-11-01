import pytorch_lightning as pl
from project.Network import Network
from project.DataModule import DataModule
import mlflow
from torch.cuda import is_available as cuda_available


def train(data_dir):

    # initialise network and datamodule
    net = Network(data_dir=data_dir)
    dm = DataModule(data_dir=data_dir, batch_size=4)

    #  start logged run-
    mlflow.pytorch.autolog()
    with mlflow.start_run() as run:
        trainer = pl.Trainer(logger=True,
                             gpus=-1 if cuda_available() else None,
                             accelerator='ddp'
                             )
        trainer.fit(net, dm)
        trainer.test(datamodule=dm)


if __name__ == '__main__':
    data_dir = '/DATA'
    train(data_dir)
