import torch
from monai.transforms import (
    CastToTyped,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImage,
    NormalizeIntensityd,
    RandAdjustContrastd,
    RandAffined,
    RandCoarseDropoutd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotated,
    RandScaleIntensityd,
    RandZoomd,
    ResizeWithPadOrCropd,
    Resized,
    ScaleIntensityd,
    ScaleIntensityRangePercentilesd,
    SelectItemsd,
    Spacingd,
    SqueezeDimd,
    ToTensord,
)

from src.transforms.load_image_xnatd import LoadImageXNATd

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
        ScaleIntensityRangePercentilesd(keys=["image"], lower=0, upper=99, b_min=0.0, b_max=255.0, clip=True),
        CastToTyped(keys=["image"], dtype=torch.float32),
    ]

def train_augment(image_size):
    """
    This transform list is used to augment images for training.
    Aim here is to improve generalisation.
    """
    return [
        RandFlipd(keys=['image'], spatial_axis=0, prob=0.5),
        RandZoomd(keys=['image'], prob=0.4, min_zoom=1.05,max_zoom=1.1),
        RandRotated(keys=['image'], prob=0.4, range_x=0.4),
        RandAffined(keys=['image'], prob=0.3, padding_mode='zeros'),
        RandGaussianNoised(keys=['image'], prob=0.3, mean=0.0, std=10.0),
        RandGaussianSmoothd(keys=['image'], prob=0.35, sigma_x=(0.5,1.0), sigma_y=(0.5,1.0)),
        RandScaleIntensityd(keys=['image'], prob=0.3, factors=(0.75,1.25)),
        RandAdjustContrastd(keys=['image'], prob=0.2, gamma=(0.5,2), retain_stats=True, invert_image=True),
        RandAdjustContrastd(keys=['image'], prob=0.4, gamma=(0.5,2), retain_stats=True, invert_image=False),
        ResizeWithPadOrCropd(
            keys=["image"],
            spatial_size=(image_size,image_size),
            mode='replicate'
        ),
        RandCoarseDropoutd(keys=['image'], prob=0.35, fill_value=0, holes=8, max_holes=16, spatial_size=(10,10), max_spatial_size=(15,15)),
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
        # Normalize with grayscale-averaged ImageNet stats (mean=0.449, std=0.226)
        # Required for pretrained ImageNet models
        NormalizeIntensityd(keys=["image"], subtrahend=0.449, divisor=0.226),
        ToTensord(keys=['image', 'label']),
        SelectItemsd(keys=['subject_id', 'image', 'label']),
        EnsureTyped(keys=['image', 'label'], track_meta=False),
    ]