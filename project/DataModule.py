import pytorch_lightning
from typing import Optional
from monai.data import CacheDataset, Dataset
from torch.utils.data import random_split, DataLoader
from torch.cuda import is_available
from monai.data import list_data_collate


class DataModule(pytorch_lightning.LightningDataModule):

    def __init__(self, data_dir: str = './', batch_size: int = 1, num_workers: int = 4, test_fraction: float = 0.1,
                 train_val_ratio: float = 0.2):
        super().__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_val_ratio = train_val_ratio
        self.test_fraction = test_fraction

    def setup(self, stage: Optional[str] = None):
        """
        Use the setup method to setup your data and define your Dataset objects
        :param stage:
        :return:
        """
        data_samples = {}  # Dataset containing all samples - see pytorch definition of Dataset
        self.train_samples, self.valid_samples, self.test_samples = random_split(data_samples,
                                                                                 self.data_split(len(data_samples)))

    def prepare_data(self, *args, **kwargs):
        pass

    def train_dataloader(self):
        """
        Define train dataloader
        :return:
        """
        train_loader = DataLoader(self.train_samples, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, collate_fn=list_data_collate,
                                  pin_memory=is_available())

        return train_loader

    def val_dataloader(self):
        """
        Define validation dataloader
        :return:
        """
        val_loader = DataLoader(self.valid_samples, batch_size=self.batch_size, num_workers=self.num_workers,
                                pin_memory=is_available())
        return val_loader

    def test_dataloader(self):
        """
        Define test dataloader
        :return:
        """
        test_loader = DataLoader(self.test_samples, batch_size=self.batch_size, num_workers=self.num_workers,
                                 pin_memory=is_available())
        return test_loader

    def data_split(self, total_count):
        test_count = int(self.test_fraction * total_count)
        train_count = int((1 - self.train_val_ratio) * (total_count - test_count))
        valid_count = int(self.train_val_ratio * (total_count - test_count))
        split = (train_count, valid_count, test_count)
        print('Number of samples (Train, validation, test) = {0}'.format(split))
        return split
