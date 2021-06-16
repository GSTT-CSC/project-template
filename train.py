import pytorch_lightning as pl
from project.Network import Network
from project.DataModule import DataModule
import mlflow
from torch.cuda import is_available as cuda_available


def train(data_dir):

    # initialise network and datamodule
    print('Creating Network and DataModule')
    dm = DataModule(data_dir=data_dir)
    net = Network()

    #  start logged run
    print('Starting logged run')
    mlflow.pytorch.autolog()
    with mlflow.start_run() as run:
        trainer = pl.Trainer(logger=True,
                             gpus=-1 if cuda_available() else None,
                             accelerator='ddp'
                             )
        print('Starting training')
        trainer.fit(net, dm)


if __name__ == '__main__':
    data_dir = '/Users/lj16/code/Other/DATA/carnax/Carnax_Images_V2'
    train(data_dir)