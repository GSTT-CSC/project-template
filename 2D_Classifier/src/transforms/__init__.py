import torch
from transforms.LoadImageXNATd import LoadImageXNATd

from monai.transforms import (
    CastToTyped,
    CropForegroundd,
    DivisiblePadd,
    EnsureChannelFirstd,
    LoadImage,
    LoadImaged,
    RandFlipd,
    RandRotate90d,
    Resized,
    ResizeWithPadOrCropd,
    ScaleIntensityd,
    SelectItemsd,
    SqueezeDimd,
    TorchVisiond,
    ToTensord,
)


def zero_or_max(x):
    """
    Example function that can be used with cropforegroundd
    """
    return (x > 0) & (x < x.max())


def load_xnat(xnat_configuration: dict, image_series_option: str):
    """
    This transform is used by the DataModule to load images from XNAT
    """
    return [
        LoadImageXNATd(
            keys=["data"],
            xnat_configuration=xnat_configuration,
            expected_filetype_ext=".dcm",
            image_loader=LoadImage(image_only=True),
            image_series_option=image_series_option,
        ),
    ]


def load_dicom():
    """
    This transform is used to load images from dicom file
    """
    return [LoadImaged(keys=["image"], reader=None, image_only=True)]


# normalisation
def normalisation():
    """
    This transform list is used to prepare tensors for training or inference
    """
    return [
        EnsureChannelFirstd(keys=["image"]),
        SqueezeDimd(keys=["image"], dim=3),
        CropForegroundd(keys=["image"], source_key="image", select_fn=zero_or_max),
        Resized(keys=["image"], size_mode="longest", spatial_size=256),
        ResizeWithPadOrCropd(keys=["image"], spatial_size=(256, 256)),
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=255.0),
        CastToTyped(keys=["image"], dtype=torch.uint8),
    ]


# training augmentations
def train_augmentation():
    """
    This transform list is used for training augmentation
    """
    return [
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        # RandRotate90d(keys=['image'], prob=0.5),
        TorchVisiond(keys=["image"], name="AugMix", severity=2, fill=0),
    ]


# final transforms
def output_transforms():
    """
    This transform list is used for final checks on tensors before use
    """
    return [
        CastToTyped(keys=["image"], dtype=torch.float32),
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        DivisiblePadd(keys=["image"], k=32),
        SelectItemsd(keys=["image", "label", "subject_id"], allow_missing_keys=True),
        ToTensord(keys=["image", "label"], track_meta=False),
    ]
