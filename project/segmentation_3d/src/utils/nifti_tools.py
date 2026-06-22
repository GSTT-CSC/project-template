import os
import logging
from typing import List
import numpy as np
import nibabel as nib

logger = logging.getLogger(__name__)


def create_empty_nifti_like(
        reference_nifti_uri: str = None,
        output_nifti_uri: str = None
        ):
    """
    Create an empty NIfTI mask with the same geometry as a reference NIfTI.

    Inputs:
        reference_nifti_uri
        output_nifti_uri

    Returns:
        output_nifti_uri
    """
    reference_nifti = nib.load(reference_nifti_uri)
    empty_mask = np.zeros(reference_nifti.shape, dtype=np.uint8)
    empty_nifti = nib.nifti1.Nifti1Image(
        empty_mask,
        reference_nifti.affine,
        header=reference_nifti.header.copy(),
    )
    nib.save(empty_nifti, output_nifti_uri)
    logger.info(f"Created empty contour NIfTI at: {output_nifti_uri}")
    return output_nifti_uri


def nifti_contour_combiner(
        input_nifti_uris: List[str] = None,
        output_nifti_uri: str = None
        ):
    """
    Combine contours from multiple NIfTI files into a single NIfTI file.

    Contour mask voxel values assigned corresponding integer values, e.g.:
    Background voxel values = 0
    Contour 1 voxel values = 1
    Contour 2 voxel values = 2
    Contour 3 voxel values = 3
    etc.

    Inputs:
        input_nifti_uris
        output_nifti_uri

    Returns:
        output_nifti_uri
    """
    if len(input_nifti_uris) == 0:
        logger.warning("Zero NIfTI files found. Skipping nifti_contour_combiner.")
        return
    if len(input_nifti_uris) == 1:
        logger.warning(f"Only one NIfTI file found: {input_nifti_uris}. Skipping nifti_contour_combiner.")
        return

    logger.info(f"Combining NIfTI files: {[os.path.basename(uri) for uri in input_nifti_uris]}")

    # load
    nifti_objects = [nib.load(uri) for uri in input_nifti_uris]

    # check contour file dimensions match
    nifti_dims = [nifti_obj.header["dim"] for nifti_obj in nifti_objects]
    nifti_dims_bool = [np.array_equal(nii, nifti_dims[0]) for nii in nifti_dims]
    if not all(nifti_dims_bool):
        logger.warning("Inconsistent NIfTI dimensions. Skipping nifti_contour_combiner.")
        return

    # set mask voxel values according to idx+1 in list
    # np.round(...).astype(int) ensures integer label values
    nifti_imgs = [
        np.round(np.multiply(nii.get_fdata(), idx+1)).astype(int)
        for idx, nii in enumerate(nifti_objects)
    ]
    nifti_combined_img = np.rollaxis(np.stack(nifti_imgs, axis=3), axis=3)

    logger.info(f"Combined NIfTI array label values: {np.unique(nifti_combined_img)}")

    priority_mask = np.copy(nifti_combined_img[0])
    secondary_masks = np.copy(nifti_combined_img[1::])
    combined_mask = np.copy(priority_mask)
    for mask in secondary_masks:
        combined_mask = mask_array_combiner(combined_mask, mask)

    # write NIfTI file
    nifti_combined_object = nib.nifti1.Nifti1Image(combined_mask, None, header=nifti_objects[0].header.copy())
    nib.save(nifti_combined_object, output_nifti_uri)
    logger.info(f"Combined contours NIfTI file saved to: {output_nifti_uri}")
    return output_nifti_uri


def mask_array_combiner(priority_mask: np.ndarray, secondary_mask: np.ndarray) -> np.ndarray:
    """
    Combine mask arrays. priority_mask values supersede secondary_mask values.

    Finds background values (0) in priority_mask and overwrites if nonzero value at
    corresponding index in secondary_mask.

    Inputs:
        priority_mask - target mask
        secondary_mask - mask applied to priority_mask (priority_mask values overrule secondary_mask values)

    Returns:
        priority_mask - updated version with additional secondary_mask

    Example:
        m1 = np.array([0, 1, 1, 0, 1])
        m2 = np.array([2, 2, 0, 0, 0])
        m3 = np.array([3, 3, 3, 3, 3])

        M = mask_array_combiner(m1, m2)
        print(M)
        >> array([2, 1, 1, 0, 1])

        M = mask_array_combiner(M, m3)
        print(M)
        >> array([2, 1, 1, 3, 1])
    """

    truth_table = (priority_mask == 0) & (secondary_mask != 0)
    np.putmask(priority_mask, truth_table, np.max(secondary_mask))
    return priority_mask
