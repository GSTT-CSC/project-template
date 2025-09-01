import os
import logging
from typing import List
import numpy as np
import nibabel as nib

logger = logging.getLogger(__name__)

def nifti_contour_combiner(input_nifti_uris: List[str], output_nifti_uri: str):
    """
    Combine multiple NIfTI masks into a single NIfTI file.
    
    Args:
        input_nifti_uris: list of NIfTI file paths
        output_nifti_uri: path to save combined NIfTI mask
    
    Returns:
        Path to saved combined NIfTI file
    """
    if not input_nifti_uris:
        logger.warning("No NIfTI files provided. Skipping combination.")
        return
    
    if len(input_nifti_uris) == 1:
        logger.warning(f"Only one NIfTI file found: {input_nifti_uris[0]}. Skipping combination.")
        return
    
    nifti_objs = [nib.load(f) for f in input_nifti_uris]
    
    # Verify dimensions match
    shapes = [nii.shape for nii in nifti_objs]
    if not all(s == shapes[0] for s in shapes):
        logger.warning("NIfTI dimensions do not match. Skipping combination.")
        return
    
    # Combine masks, assigning unique integer per mask
    combined = np.zeros(shapes[0], dtype=int)
    for idx, nii in enumerate(nifti_objs):
        mask = np.round(nii.get_fdata()).astype(int)
        combined = mask_array_combiner(combined, mask, idx+1)
    
    combined_nifti = nib.Nifti1Image(combined, affine=nifti_objs[0].affine, header=nifti_objs[0].header)
    nib.save(combined_nifti, output_nifti_uri)
    logger.info(f"Saved combined NIfTI to {output_nifti_uri}")
    return output_nifti_uri


def mask_array_combiner(priority_mask: np.ndarray, secondary_mask: np.ndarray, label_value: int) -> np.ndarray:
    """
    Combine two mask arrays. Priority_mask values take precedence unless zero.
    Assigns `label_value` to secondary_mask voxels.
    """
    combined = priority_mask.copy()
    combined[(priority_mask == 0) & (secondary_mask != 0)] = label_value
    return combined
