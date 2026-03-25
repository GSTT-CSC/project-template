import logging
from collections import Counter
from typing import List, Optional
import mlflow
import numpy as np
import pytorch_lightning
import torch

from monai.data import CacheDataset, Dataset
from monai.data import pad_list_data_collate
from monai.transforms import Compose

from sklearn.model_selection import train_test_split
from torch.cuda import is_available
from torch.utils.data import DataLoader

from src.transforms import normalise, train_augment, output 
from src.transforms.SafeWrapper import SafeWrapperTransform

logger = logging.getLogger(__name__)

label_dict = {
    'NEGATIVE': 0,
    'POSITIVE': 1,
}

class DataModule(pytorch_lightning.LightningDataModule):

    def __init__(self, data, dm_batch_size: int = 1, num_workers: int = 16,
                test_fraction: float = 0.2, cache_dataset=False,
                random_seed: int = 42, image_size: int = 224):
        super().__init__()
        self.data = data
        self.xnat_data_list = None
        self.num_workers = num_workers
        self.batch_size = dm_batch_size
        self.test_fraction = test_fraction
        self.cache_dataset = cache_dataset
        self.random_seed = random_seed
        self.image_size = image_size

        self.torch_device = self.trainer.model.device if self.trainer else torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.train_transforms = Compose(
            normalise(self.image_size)
            + train_augment(self.image_size)
            + output(self.image_size)
        )
        
        self.val_transforms = Compose(
            normalise(self.image_size)
            + output(self.image_size)
        )

    def calculate_class_weights(self, data) -> list:
        """Calculate class weights for unbalanced datasets

        :param data: input data
        :type data: list
        :return: class weights
        :rtype: list
        """
        labels = self.get_labels(data)
        return [
            len(labels) / (len(set(labels)) * labels.count(l))
            for l in sorted(set(labels))
        ]
    
    def get_labels(self, data) -> list:
        """Get list of sample labels

        :param data: input data
        :type data: list
        :return: list of labels
        :rtype: list
        """
        return [
            int(sample["action_data"])
            for subject in data
            for sample in subject["data"]
            if sample["data_label"] == "label"
        ]

    def setup(self, stage: Optional[str] = None):
        """
        Use the setup method to set up your data and define your Dataset objects
        :param stage:
        :return:
        """
        data = self.data

        self.labels = [int(sample['action_data']) for subject in data for sample in subject['data'] if
                       sample['data_label'] == 'label']

        self.train_data, self.validation_data = train_test_split(data,
                                                                 test_size=self.test_fraction,
                                                                 stratify=self.labels,
                                                                 random_state=self.random_seed,
                                                                 )

        self.data_manifest = {
            'N_total': len(data),
            'train': self.dataset_stats(self.train_data),
            'validation': self.dataset_stats(self.validation_data),
            'labels': label_dict
        }

        mlflow.log_dict(self.data_manifest, "data_manifest.json")

        if self.cache_dataset:
            self.val_dataset = CacheDataset(data=self.validation_data,
                                            transform=SafeWrapperTransform(transform = self.val_transforms,
                                                                           log_file = "val_transform_failures.csv",
                                                                           image_size = self.image_size),
                                            num_workers=self.num_workers)
            self.train_dataset = CacheDataset(data=self.train_data,
                                            transform=SafeWrapperTransform(transform = self.train_transforms,
                                                                           log_file = "train_transform_failures.csv",
                                                                           image_size = self.image_size),
                                            num_workers=self.num_workers)
        else:
            self.val_dataset = Dataset(data=self.validation_data, transform=SafeWrapperTransform(transform = self.val_transforms,
                                                                                log_file = "val_transform_failures.csv",
                                                                                image_size = self.image_size)
                                                                                )
            self.train_dataset = Dataset(data=self.train_data, transform=SafeWrapperTransform(transform = self.train_transforms,
                                                                                log_file = "train_transform_failures.csv",
                                                                                image_size = self.image_size)
                                                                                )

    def train_dataloader(self):
        """
        Define train dataloader
        :return:
        """
        
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=pad_list_data_collate,
                          pin_memory=is_available())

    def val_dataloader(self):
        """
        Define validation dataloader
        :return:
        """
        return DataLoader(self.val_dataset, batch_size=1, num_workers=self.num_workers,
                          pin_memory=is_available(), collate_fn=pad_list_data_collate, )

    def dataset_stats(self, dataset):
        stats = {'n_samples': len(dataset)}
        stats["class_weights"] = self.calculate_class_weights(dataset)
        return stats