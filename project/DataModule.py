import logging
import random
import copy
from collections import Counter
from typing import List, Dict

import mlflow
import pytorch_lightning
from monai.data import CacheDataset, Dataset, DataLoader, list_data_collate
from monai.transforms import Compose
from torch.cuda import is_available
from torchvision.utils import make_grid
from sklearn.model_selection import train_test_split

from mlops.data.tools.tools import DataBuilderXNAT
from xnat.mixin import ImageScanData, SubjectData
import matplotlib.pyplot as plt

from project.transforms import (
    load_xnat,
    train_augmentation,
    normalisation,
    output_transforms,
)

# Update this with your labels
label_dict = {
    'Normal': 0,
    'Positive': 1,
}

class DataModule(pytorch_lightning.LightningDataModule):

    def __init__(self, xnat_configuration: dict = None, batch_size: int = 1, num_workers: int = 4,
                visualise_training_data = True,):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.xnat_configuration = xnat_configuration
        self.visualise_training_data = visualise_training_data

        self.train_transforms = Compose(
            load_xnat(self.xnat_configuration, self.image_series_option)
            + normalisation()
            + train_augmentation()
            + output_transforms()
        )

        self.val_test_transforms = Compose(
            load_xnat(self.xnat_configuration, self.image_series_option) + normalisation() + output_transforms()
        )

    def get_data(self) -> None:
        """
        Fetches raw XNAT data and stores in raw_data attribute
        """
        actions = [
            (self.fetch_xr, "image"),
            (self.fetch_label, "label"),
        ]

        data_builder = DataBuilderXNAT(self.xnat_configuration,
                                       actions=actions,
                                       num_workers=self.num_workers)

        data_builder.fetch_data()
        self.raw_data = data_builder.dataset
    
    def validate_data(self) -> None:
        """
        Remove samples that do not have both an image and a label
        """
        temp = []
        n_invalid = 0
        for sample in self.raw_data:
            logging.debug(f"Processing sample: {sample['subject_id']}")
            image_objects = [
                im for im in sample["data"] if im["source_action"] == "fetch_xr"
            ]
            label_object = [
                im for im in sample["data"] if im["source_action"] == "fetch_label"
            ]
            if len(image_objects) >= 1 and len(label_object) >= 1:
                temp.append(sample)
            else:
                n_invalid = n_invalid + 1

        logging.info(f"Removed {n_invalid} invalid samples.")
        self.raw_data = temp

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

    def prepare_data(self, *args, **kwargs):
        
        self.get_data()
        logging.info("Validating data")
        self.validate_data()

        # Creates 50-25-25 split for train, val, test sets (update as you wish)
        self.train_data, val_test = train_test_split(self.raw_data, test_size=0.5, random_state=42)
        self.val_data, self.test_data = train_test_split(val_test, test_size=0.5, random_state=42)

        mlflow.log_text(
            str([sample["subject_id"] for sample in self.test_data]),
            "test_ids.txt",
        )

        self.data_manifest = {
            "N_patients": len(self.raw_data),
            "train": self.dataset_stats(self.train_data),
            "validation": self.dataset_stats(self.val_data),
            "test": self.dataset_stats(self.test_data),
            "labels": label_dict,
        }

        mlflow.log_dict(self.data_manifest, "data_manifest.json")
        
        if self.visualise_training_data:
            self.visualise_data(self.train_data, self.train_transforms, n_samples=24)
        
        self.train_dataset = Dataset(data=self.train_data, transform=self.train_transforms)
        self.val_dataset = Dataset(data=self.val_data, transform=self.val_test_transforms)
        self.test_dataset = Dataset(data=self.test_data, transform=self.val_test_transforms)

    def visualise_data(self, data, transforms: Compose, n_samples) -> None:
        """Visualise a subset of random images from training dataloader after transforms

        :param data: input data
        :type data: list
        :param transforms: transforms
        :type transforms: Compose
        :param n_samples: number of samples to visualise
        :type n_samples: int
        """
        dl = DataLoader(
            Dataset(
                data=random.sample(data, min(n_samples, len(data))),
                transform=transforms,
            ),
            num_workers=self.num_workers,
            batch_size=n_samples,
            collate_fn=list_data_collate,
        )
        inputs = next(iter(dl))
        grid = make_grid(inputs["image"], nrow=4)
        plt.gcf().set_dpi(200)  # default size is a bit blurry
        plt.imshow(grid[0, :, :].rot90(k=3))
        mlflow.log_figure(plt.gcf(), "sample_images.png")
        mlflow.log_text(", ".join(inputs["subject_id"]), "sample_images_labels.txt")


    def train_dataloader(self):
        """
        Define train dataloader
        :return:
        """
        if not self.train_dataset:
            self.train_dataset = CacheDataset(
                                    data = self.train_data,
                                    transform = self.train_transforms,
                                    num_workers = self.num_workers,
            )

        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, collate_fn=list_data_collate,
                          pin_memory=is_available())

    def val_dataloader(self):
        """
        Define validation dataloader
        :return:
        """
        if not self.val_dataset:
            self.val_dataset = CacheDataset(
                                    data = self.val_data,
                                    transform = self.val_test_transforms,
                                    num_workers = self.num_workers,
            )

        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                            num_workers=self.num_workers, collate_fn=list_data_collate,
                            pin_memory=is_available())
    
    def test_dataloader(self):
        """
        Define validation dataloader
        :return:
        """
        if not self.test_dataset:
            self.test_dataset = CacheDataset(
                                    data = self.test_data,
                                    transform = self.val_test_transforms,
                                    num_workers = self.num_workers,
            )

        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                            num_workers=self.num_workers, collate_fn=list_data_collate,
                            pin_memory=is_available())


    @staticmethod
    def fetch_xr(subject_data: SubjectData = None) -> List[ImageScanData]:
        """
        Function that identifies and returns the required xnat ImageData object from a xnat SubjectData object
        along with the 'key' that it will be used to access it.
        """

        scan_objects = []

        for exp in subject_data.experiments:
            if "CR" in subject_data.experiments[exp].modality or "DX" in subject_data.experiments[exp].modality:
                for scan in subject_data.experiments[exp].scans:
                    scan_objects.append(subject_data.experiments[exp].scans[scan])
        return scan_objects
    
    @staticmethod
    def fetch_label(subject_data: SubjectData = None):
        """
        Function that identifies and returns the required label from a XNAT SubjectData object.
        """
        label = None
        for exp in subject_data.experiments:
            if "CR" in subject_data.experiments[exp].modality or "DX" in subject_data.experiments[exp].modality:
                temp_label = subject_data.experiments[exp].label
                x = temp_label.split("_")
                label = int(x[1])

        return label
    
    def dataset_stats(self, dataset: List[Dict], fields=["label"]) -> dict:
        """Calculate dataset statistics

        :param dataset: input dataset
        :type dataset: list
        :param fields: fields to calculate statistics for, defaults to ["label"]
        :type fields: list, optional
        :return: dataset statistics
        :rtype: dict
        """
        logging.debug(f"Fetching stats for {dataset}")

        stats = {}
        stats["n_samples"] = len(dataset)

        all_labels = self.get_labels(dataset)

        count_dict = dict(Counter(all_labels))

        for k, v in count_dict.items():
            stats[f"N_{str(k)}"] = v

        stats["class_weights"] = self.calculate_class_weights(dataset)

        return stats
