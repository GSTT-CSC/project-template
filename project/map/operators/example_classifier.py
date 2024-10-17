"""This is an example operator to implement an image classifier"""

import monai.deploy.core as md
import numpy as np
from monai.deploy.core.domain.dicom_series_selection import StudySelectedSeries
from typing import List, Optional
from collections import UserDict
from monai.deploy.core import (
    ExecutionContext,
    InputContext,
    IOType,
    Operator,
    OutputContext,
    DataPath,
)

from monai.data import DataLoader, Dataset
from monai.transforms import Compose
from transforms import load_and_norm, crop_and_output
import torch
import datetime


@md.input("study_selected_series_list", List[StudySelectedSeries], IOType.IN_MEMORY)
@md.input("bbox_array", np.ndarray, IOType.DISK)
@md.output("output_udict", UserDict, IOType.DISK)
@md.env(pip_packages=["monai"])
class ClassifierOperator(Operator):
    """Classifies the given image and returns the class name."""

    def __init__(self, classifier_model: Optional[str] = "", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classifier_model = classifier_model

    def compute(
        self,
        op_input: InputContext,
        op_output: OutputContext,
        context: ExecutionContext,
    ):

        # get image from list
        study_selected_series_list = op_input.get("study_selected_series_list")

        if not study_selected_series_list or len(study_selected_series_list) < 1:
            raise ValueError("Missing expected input 'study_selected_series_list'")

        selected_series = study_selected_series_list[0].selected_series[0].series
        dicom_list = [sop._sop.filename for sop in selected_series._sop_instances]

        transforms = Compose(load_and_norm() + crop_and_output())

        bbox_array = op_input.get("bbox_array").tolist()[0]

        if bbox_array[-1] < 0.9:
            # Update as appropriate to given project
            output_dict = {
                "result": "Indeterminate result",
                "error_message": "Unable to identify the scaphoid bone",
                "advice": "ScaphX is unable to provide a reliable assessment for this case. Please proceed with standard clinical evaluation, including X-ray interpretation and additional imaging as needed.",
                "start_time": f"{begin}",
                "end_time": f"{end}",
                "bounding_box": bbox_array,
            }
        else:
            dataset = Dataset(
                data=[{"image": dicom_list, "label": -1, "roi": bbox_array}],
                transform=transforms,
            )

            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

            model = torch.jit.load(self.classifier_model)
            model.eval()

            begin = str(datetime.datetime.now())

            with torch.inference_mode():
                for d in dataloader:
                    pred = model(d["image"].to("cpu"))
                    confidence = torch.nn.functional.softmax(pred, dim=1)

            result = confidence[0].detach().numpy()[0]

            end = str(datetime.datetime.now())

            # Update as appropriate to specific project
            if result >= 0.9:
                output_dict = {
                    "result": "Scaphoid Fracture Detected",
                    "error_message": "None",
                    "advice": "Apply a below elbow POP backslab and refer the patient to fracture clinic as per scaphoid pathway.",
                    "start_time": f"{begin}",
                    "end_time": f"{end}",
                    "bounding_box": bbox_array,
                }
            elif result >= 0.9:  # and result < 0.8:
                output_dict = {
                    "result": "Indeterminate result",
                    "error_message": "Unable to classify image",
                    "advice": "ScaphX is unable to provide a reliable assessment for this case. Please proceed with standard clinical evaluation, including X-ray interpretation and additional imaging as needed.",
                    "start_time": f"{begin}",
                    "end_time": f"{end}",
                    "bounding_box": bbox_array,
                }
            else:
                output_dict = {
                    "result": "No Scaphoid Fracture Detected",
                    "error_message": "None",
                    "advice": "Provide patient with Futura splint and refer the patient as per scaphoid pathway.",
                    "start_time": f"{begin}",
                    "end_time": f"{end}",
                    "bounding_box": bbox_array,
                }

        output_udict = UserDict(output_dict)
        op_output.set(output_udict, "output_udict")
