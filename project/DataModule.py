import pytorch_lightning
from typing import Optional
from monai.data import CacheDataset, Dataset
from torch.utils.data import random_split, DataLoader
from torch.cuda import is_available
from monai.data import list_data_collate
from mlops.data.tools.tools import xnat_build_dataset
from xnat.mixin import ImageScanData, SubjectData
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union


class DataModule(pytorch_lightning.LightningDataModule):

    def __init__(self, data_dir: str = './', xnat_configuration: dict = None, batch_size: int = 1, num_workers: int = 4,
                 test_fraction: float = 0.1,
                 train_val_ratio: float = 0.2):
        super().__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_val_ratio = train_val_ratio
        self.test_fraction = test_fraction
        self.xnat_configuration = xnat_configuration

    def setup(self, stage: Optional[str] = None):
        """
        Use the setup method to setup your data and define your Dataset objects
        :param stage:
        :return:
        """
        actions = [()]  # list of tuples defining action functions and their data keys

        self.xnat_data_list = xnat_build_dataset(self.xnat_configuration, actions=actions)

        self.train_samples, self.valid_samples, self.test_samples = random_split(data_samples,
                                                                                 self.data_split(len(data_samples)))

    def prepare_data(self, *args, **kwargs):
        pass

    def train_dataloader(self):
        """
        Define train dataloader
        :return:
        """
        return DataLoader(self.train_samples, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, collate_fn=list_data_collate,
                          pin_memory=is_available())

    def val_dataloader(self):
        """
        Define validation dataloader
        :return:
        """
        return DataLoader(self.valid_samples, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=is_available())

    def test_dataloader(self):
        """
        Define test dataloader
        :return:
        """
        return DataLoader(self.test_samples, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=is_available())

    def data_split(self, total_count):
        test_count = int(self.test_fraction * total_count)
        train_count = int((1 - self.train_val_ratio) * (total_count - test_count))
        valid_count = int(self.train_val_ratio * (total_count - test_count))
        split = (train_count, valid_count, test_count)
        print('Number of samples (Train, validation, test) = {0}'.format(split))
        return split
