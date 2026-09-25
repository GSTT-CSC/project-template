import os
import logging
from typing import List
import numpy as np
import nibabel as nib

logger = logging.getLogger(__name__)


def nifti_contour_combiner(
        input_nifti_uris: List[str] = None,
        output_nifti_uri: str = None
        ):
    """
    Combine contours from one or more NIfTI files into a single NIfTI file.

    Contour mask voxel values assigned corresponding integer values, e.g.:
    Background voxel values = 0
    Contour 1 voxel values = 1
    Contour 2 voxel values = 2
    Contour 3 voxel values = 3
    etc.

    A single contour is valid and gives a binary {0, 1} mask. Callers copy
    output_nifti_uri afterwards, so this always writes it or raises.

    Inputs:
        input_nifti_uris
        output_nifti_uri

    Returns:
        output_nifti_uri
    """
    if len(input_nifti_uris) == 0:
        raise ValueError("No NIfTI files to combine.")

    logger.info(f"Combining NIfTI files: {[os.path.basename(uri) for uri in input_nifti_uris]}")

    nifti_objects = [nib.load(uri) for uri in input_nifti_uris]

    # check contour file dimensions match
    nifti_dims = [nifti_obj.header["dim"] for nifti_obj in nifti_objects]
    if not all(np.array_equal(dim, nifti_dims[0]) for dim in nifti_dims):
        raise ValueError(
            f"Inconsistent NIfTI dimensions across contours: "
            f"{dict(zip([os.path.basename(uri) for uri in input_nifti_uris], [tuple(d[1:4]) for d in nifti_dims]))}"
        )

    # Build the multi-label mask incrementally as uint8: read each contour one at a time and combine
    combined_mask = None
    for idx, nifti_obj in enumerate(nifti_objects):
        # this contour's label mask: value idx+1 where present, 0 elsewhere (uint8)
        contour_label_mask = (np.asanyarray(nifti_obj.dataobj) > 0).astype(np.uint8) * (idx + 1)
        if combined_mask is None:
            combined_mask = np.zeros(contour_label_mask.shape, dtype=np.uint8)
        combined_mask = mask_array_combiner(combined_mask, contour_label_mask)

    logger.info(f"Combined NIfTI array label values: {np.unique(combined_mask)}")

    combined_object = nib.nifti1.Nifti1Image(
        combined_mask, affine=None, header=nifti_objects[0].header.copy()
    )
    combined_object.set_data_dtype(np.uint8)
    nib.save(combined_object, output_nifti_uri)
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
