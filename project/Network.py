import pytorch_lightning
import torch


class Network(pytorch_lightning.LightningModule):

    """
    Network object defines the model architecture, inherits from LightningModule.

    https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._model = None  # object that describes the model layers - should inherit from torch.nn.Module
        self.loss_function = None  # Define loss function

    def forward(self, x):
        """
        Forward pass
        :param x:
        :return:
        """
        return self._model(x)

    def training_step(self, batch, batch_idx):
        """
        Training step
        :param batch:
        :param batch_idx:
        :return: loss
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_funciton(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step
        :param batch:
        :param batch_idx:
        :return: loss
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        return loss

    def test_step(self, batch, batch_idx):
        """
        test step
        :param batch:
        :param batch_idx:
        :return: loss
        """
        x, y = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        return loss

    def configure_optimizers(self):
        """
        Setup optimiser
        :return: Optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=0.02)
