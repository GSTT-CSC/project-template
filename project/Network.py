import pytorch_lightning


class Network(pytorch_lightning.LightningModule):
    def __init__(self, **kwargs):
        super(Network, self).__init__()
        # self._model = monai.network.nets.UNet()

    def forward(self, x):
        return self._model(x)
