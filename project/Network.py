import pytorch_lightning
from monai.networks.nets import Classifier
from torch.nn import CrossEntropyLoss
from monai.inferers import sliding_window_inference
from torch.optim import Adam


class Network(pytorch_lightning.LightningModule):

    def __init__(self, **kwargs):
        super(Network, self).__init__()
        self.in_shape = kwargs.get('in_shape', (1, 1024, 1024))
        self.classes = kwargs.get('classes', 2)
        self.channels = kwargs.get('channels', (8, 16, 32, 64))
        self.strides = kwargs.get('strides', (1, 1, 1, 1))
        self._learning_rate = kwargs.get('lr', 1e-3)
        self.loss_function = CrossEntropyLoss()

        self._model = Classifier(self.in_shape,
                                 self.classes,
                                 self.channels,
                                 self.strides,
                                 kernel_size=3,
                                 num_res_units=2,
                                 act='PRELU',
                                 norm='INSTANCE',
                                 dropout=None,
                                 bias=True,
                                 last_act=None)

    def forward(self, x):
        return self._model(x)

    def configure_optimizers(self):
        optimizer = Adam(self._model.parameters(), self._learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        self.log('train_loss', loss)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     images, labels = batch["image"], batch["label"]
    #     output = self.forward(images)
    #     loss = self.loss_function(output, labels)
    #     self.log('val_loss', loss)
