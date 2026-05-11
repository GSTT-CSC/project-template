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
    This transform is used by the DataModule to load images from XNAT.

    Args:
        xnat_configuration (dict): XNAT connection settings passed to LoadImageXNATd,
            typically containing keys such as server, user, password, and project.

    Returns:
        list[MapTransform]: A single-element list containing LoadImageXNATd, which reads
            DICOM files from XNAT and stores the resulting image array under the 'data' key.
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
    This transform list is used to prepare tensors for training or inference.

    Squeezes the redundant depth dimension, ensures a channel-first layout, crops
    to the foreground, resizes to a slightly larger intermediate size, then clips
    and scales intensity to [0, 255] and casts to float32.

    Args:
        image_size (int): Target spatial size for the longest side of the image.
            The longest side of the image is resized to image_size + 20, 
            with the shorter size scaled proportionally, to leave a small margin
            before the final crop applied by downstream transforms.

    Returns:
        list[MapTransform]: Ordered list of MONAI transforms operating on the
            'image' key. Output tensor has shape (1, H, W) where 
            max(H,W) = image_size + 20 with dtype float32 and intensity
            values in [0.0, 255.0].
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

    Applies a randomised sequence of spatial and intensity augmentations before
    padding or cropping to the final square size. All augmentations are
    probabilistic and operate on the 'image' key only (labels are unaffected).

    Spatial augmentations: horizontal flip, zoom (×1.05–1.10), rotation
        (±0.4 rad), and random affine with zero-padding.
    Intensity augmentations: Gaussian noise (σ≤10), Gaussian blur (σ 0.5–1.0),
        intensity scaling (×0.75–1.25), and contrast adjustment (γ 0.5–2.0,
        applied twice — once inverted, once normal).
    Dropout: random coarse dropout of 8–16 rectangular patches of 10–15 px.

    Args:
        image_size (int): Final spatial size (pixels) for both height and width.
            The image is padded or cropped to (image_size, image_size) at the
            end of the augmentation pipeline.

    Returns:
        list[MapTransform]: Ordered list of MONAI random transforms. Output
            tensor has shape (1, image_size, image_size) with the same dtype
            and intensity range as the input.
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

    Pads or crops to the target square size, scales intensity to [0, 1], then
    applies ImageNet-style normalisation (mean=0.449, std=0.226, the grayscale
    average of the per-channel ImageNet statistics). Required when using
    pretrained ImageNet backbones. Finally converts to tensors and selects
    only the keys needed downstream.

    Args:
        image_size (int): Final spatial size (pixels) for both height and width.
            The image is padded or cropped to (image_size, image_size).

    Returns:
        list[MapTransform]: Ordered list of MONAI transforms. After the last
            step, the batch dict contains only 'subject_id', 'image', and
            'label'. The image tensor has shape (1, image_size, image_size),
            dtype float32, and is ImageNet-normalised. The label tensor has
            meta tracking disabled (track_meta=False).
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