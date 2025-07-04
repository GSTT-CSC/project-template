import torch
from monai.transforms import (
    LoadImage,
    SqueezeDimd,
    EnsureChannelFirstd,
    CropForegroundd,
    Resized,
    ScaleIntensityd,
    CastToTyped,
    RandFlipd,
    RandZoomd,
    RandRotated,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandAdjustContrastd,
    RandCoarseDropoutd,
    ResizeWithPadOrCropd,
    ToTensord,
    SelectItemsd,
    EnsureTyped,
    Spacingd,
)

from src.transforms.LoadImageXNATd import LoadImageXNATd

def load_xnat(xnat_configuration: dict):
    """
    This transform is used by the DataModule to load images from XNAT
    """
    return [
        LoadImageXNATd(
            keys=['data'],
            xnat_configuration=xnat_configuration, 
            expected_filetype_ext='.dcm',
            image_loader=LoadImage(image_only=True, prune_meta_pattern="^0|^2")
        ),
    ]

def normalise(image_size):
    """
    This transform list is used to prepare tensors for training or inference
    """
    return [
        SqueezeDimd(keys=['image'], dim=2),
        EnsureChannelFirstd(keys=['image']),
        CropForegroundd(keys=['image'], source_key='image'),
        Resized(keys=['image'], size_mode='longest', spatial_size=image_size+20),
        #Maybe limit top intensity in case of big spikes?
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=255.0),
        CastToTyped(keys=["image"], dtype=torch.uint8),
    ]

def train_augment(image_size):
    """
    This transform list is used to augment images for training.
    Aim here is to improve generalisation.
    """
    return [
        RandFlipd(keys=['image'], spatial_axis=0, prob=0.5),
        RandZoomd(keys=['image'], prob=0.2, min_zoom=1.05,max_zoom=1.1),
        RandRotated(keys=['image'], prob=0.2, range_x=0.4),
        RandAffined(keys=['image'], prob=0.2, padding_mode='zeros'),
        RandGaussianNoised(keys=['image'], prob=0.1, mean=0.0, std=0.1),
        RandGaussianSmoothd(keys=['image'], prob=0.2, sigma_x=(0.5,1.0)),
        RandScaleIntensityd(keys=['image'], prob=0.15, factors=(0.75,1.25)),
        RandAdjustContrastd(keys=['image'], prob=0.1, gamma=(0.5,2), retain_stats=True, invert_image=True),
        RandAdjustContrastd(keys=['image'], prob=0.3, gamma=(0.5,2), retain_stats=True, invert_image=False),
        ResizeWithPadOrCropd(
            keys=["image"],
            spatial_size=(image_size,image_size),
            mode='replicate'
        ),
        RandCoarseDropoutd(keys=['image'], prob=0.5, fill_value=0, holes=8, max_holes=16, spatial_size=(10,10), max_spatial_size=(36,36)),
    ]

def output(image_size):
    """
    This transform list is used for final normalisation and feature selection.
    """
    return [
        ResizeWithPadOrCropd(
            keys=["image"],
            spatial_size=(image_size,image_size),
            mode='replicate'
        ),
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1),
        ToTensord(keys=['image', 'label']),
        SelectItemsd(keys=['subject_id', 'image', 'label']),
        EnsureTyped(keys=['image', 'label'], track_meta=False),
    ]