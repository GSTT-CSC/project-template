"""This is an example operator to implement an image segmentator"""

from typing import List, Optional
import numpy as np
import torch
import monai.deploy.core as md
from monai.deploy.core.domain.dicom_series_selection import StudySelectedSeries

from monai.deploy.core import (
    ExecutionContext,
    InputContext,
    IOType,
    Operator,
    OutputContext,
)

from monai.data import Dataset
from monai.transforms import Compose
from transforms import load_and_norm

from torch.utils.data import DataLoader
from transforms.Detector import Detectord


@md.input("study_selected_series_list", List[StudySelectedSeries], IOType.IN_MEMORY)
@md.output("bbox_array", np.ndarray, IOType.DISK)
class DetectorOperator(Operator):
    """Apply detector model on the selected image and return bbox coordinates."""

    def __init__(
        self,
        detector_model: Optional[str] = "",
        mmdet_config: Optional[str] = "",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.detector_model = detector_model
        self.mmdet_config = mmdet_config

    def compute(
        self,
        op_input: InputContext,
        op_output: OutputContext,
        context: ExecutionContext,
    ):
        """
        Applies transforms to pixel data necessary for use of detector.
        Set operator output to bbox coordinates
        """

        # get image from list
        study_selected_series_list = op_input.get("study_selected_series_list")

        if not study_selected_series_list or len(study_selected_series_list) < 1:
            raise ValueError("Missing expected input 'study_selected_series_list'")

        selected_series = study_selected_series_list[0].selected_series[0].series
        dicom_list = [sop._sop.filename for sop in selected_series._sop_instances]

        transforms = Compose(
            load_and_norm()
            + [
                Detectord(
                    keys=["image"],
                    config_file=self.mmdet_config,
                    checkpoint_file=self.detector_model,
                    device=torch.device("cpu"),
                )
            ]
        )

        dataset = Dataset(
            data=[{"image": dicom_list, "label": -1}],
            transform=transforms,
        )

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        for d in dataloader:
            bbox = d["roi"].numpy()

        op_output.set(bbox, "bbox_array")
