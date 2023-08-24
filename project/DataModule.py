import pytorch_lightning
from typing import Optional
from monai.data import CacheDataset, Dataset
from torch.utils.data import random_split, DataLoader
from torch.cuda import is_available
from monai.data import list_data_collate
from mlops.data.tools.tools import xnat_build_dataset
from xnat.mixin import ImageScanData, SubjectData
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
import mlflow

class DataModule(pytorch_lightning.LightningDataModule):

    def __init__(self, data_dir: str = './', xnat_configuration: dict = None, batch_size: int = 1, num_workers: int = 4,
                 test_fraction: float = 0.1, test_batch: int = 0,
                 train_val_ratio: float = 0.2):
        super().__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_val_ratio = train_val_ratio
        self.test_fraction = test_fraction
        self.xnat_configuration = xnat_configuration
        self.test_batch = test_batch

    def setup(self, stage: Optional[str] = None):
        """
        Use the setup method to setup your data and define your Dataset objects
        :param stage:
        :return:
        """
        actions = [()]  # list of tuples defining action functions and their data keys

        self.xnat_data_list = xnat_build_dataset(self.xnat_configuration, actions=actions, test_batch=self.test_batch)

        self.raw_data = Dataset(self.xnat_data_list)

        # Split data
        val_size = int(self.train_val_ratio * len(self.raw_data))
        train_size = len(self.raw_data) - val_size
        self.train_data, self.validation_data = random_split(self.raw_data, [train_size, val_size])

        mlflow.log_params({'N_train': len(self.train_data),
                           'N_validation': len(self.validation_data)})

        self.val_dataset = CacheDataset(data=self.validation_data, transform=self.val_transforms, copy_cache=False, num_workers=self.num_workers)
        self.train_dataset = CacheDataset(data=self.train_data, transform=self.train_transforms, copy_cache=False, num_workers=self.num_workers)

    def prepare_data(self, *args, **kwargs):
        pass

    def train_dataloader(self):
        """
        Define train dataloader
        :return:
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, collate_fn=list_data_collate,
                          pin_memory=is_available())

    def val_dataloader(self):
        """
        Define validation dataloader
        :return:
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          pin_memory=is_available())

