import pytorch_lightning
from typing import Optional


class SpleenDataModule(pytorch_lightning.LightningDataModule):

    def __init__(self, data_dir: str = './', batch_size: int = 1, num_workers: int = 6):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        pass

    def prepare_data(self, *args, **kwargs):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass
