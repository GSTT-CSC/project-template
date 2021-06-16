import pytorch_lightning
from typing import Optional
from monai.data import CacheDataset, list_data_collate
from monai.utils import set_determinism
from torch.utils.data import random_split, DataLoader
import glob
import os
import numpy as np
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensityd,
    ToTensord,
    ToTensor,
    Resized
)
import csv


class DataModule(pytorch_lightning.LightningDataModule):

    def __init__(self, data_dir: str = './', batch_size: int = 1, num_workers: int = 6, train_val_ratio: float = 0.1):
        super().__init__()
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_val_ratio = train_val_ratio

    def setup(self, stage: Optional[str] = None):
        train_images = sorted(glob.glob(os.path.join(self.data_dir, "*.dcm")))
        # train_labels = sorted(glob.glob(os.path.join(self.data_dir, "labelsTr", "*.nii.gz")))

        with open(os.path.join(self.data_dir, 'Data_Csv_File.csv'), mode='r') as infile:
            label_dict = dict(csv.reader(infile))

        train_labels = []
        for i, img_path in enumerate(train_images):
            train_labels.append(int(label_dict[os.path.basename(img_path)]))

        # set deterministic training for reproducibility
        set_determinism(seed=0)

        data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in
                      zip(train_images, train_labels)]

        # Split into train and validation data
        indices = np.arange(len(data_dicts))
        np.random.shuffle(indices)
        val_split = int(self.train_val_ratio * len(data_dicts))
        val_indices = indices[:val_split]
        train_indices = indices[val_split:]

        self.train_files = [data_dicts[i] for i in train_indices]
        self.val_files = [data_dicts[i] for i in val_indices]

        self.train_transforms = Compose(
            [
                LoadImaged(keys=['image']),
                Resized(keys=['image'], spatial_size=[1024, 1024]),
                # Canny(),
                ScaleIntensityd(keys=['image']),
                # AddChanneld(keys=['image']),
                ToTensord(keys=['image']),
            ]
        )

        self.train_ds = CacheDataset(data=self.train_files, transform=self.train_transforms, cache_rate=1.0, num_workers=4)
        self.val_ds = CacheDataset(data=self.val_files, transform=self.train_transforms, cache_rate=1.0, num_workers=4)

    def cropper(self):
        # CAnny fitler
        pass

    def prepare_data(self, *args, **kwargs):
        """ Steps to perform before data setup"""
        pass

    def train_dataloader(self):
        train_loader = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                                  collate_fn=list_data_collate)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers)
        return val_loader
