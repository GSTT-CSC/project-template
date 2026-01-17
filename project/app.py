"""
This file contains the application class for MAP building.
"""

import monai.deploy.core as md
import logging
from monai.deploy.core import Application
from monai.deploy.operators.dicom_utils import ModelInfo
from monai.deploy.operators import (
    DICOMEncapsulatedPDFWriterOperator,
    DICOMSeriesSelectorOperator,
)
from monai.deploy.operators import DICOMDataLoaderOperator

# TODO: Import project-specific operators from map.operators, eg

from map.operators.example_detector import DetectorOperator
from map.operators.example_classifier import ClassifierOperator
from map.operators.generate_pdf import GeneratePDFOperator

import os
import json
from pathlib import Path

os.environ["root_dir"] = os.path.dirname(os.path.realpath(__file__))

# These files are expected to exist and correctly filled in
config = json.load(open(Path(__file__).resolve().parent / "map" / "app_config.json"))
requirements_file = Path(__file__).resolve().parent / "map" / "app_requirements.txt"


@md.resource(
    cpu=config["resources"]["cpu"],
    gpu=config["resources"]["gpu"],
    memory=config["resources"]["memory"],
)
@md.env(pip_packages=requirements_file.as_posix())
class myApplication(Application):
    """Classifies the given image and returns the class name."""

    def __init__(self, *args, **kwargs):
        """Creates an application instance."""
        self._logger = logging.getLogger("{}.{}".format(__name__, type(self).__name__))
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        # This method calls the base class to run. Can be omitted if simply calling through.
        self._logger.debug(f"Begin {self.run.__name__}")
        super().run(*args, **kwargs)
        self._logger.debug(f"End {self.run.__name__}")

    def compose(self):
        """Creates the app specific operators and chain them up in the processing DAG."""

        self._logger.debug(f"Begin {self.compose.__name__}")

        app_directory = os.path.dirname(os.path.realpath(__file__))
        model_info = ModelInfo(creator="", name="", version="", uid="")

        # TODO: Define your models and optionally additional configs
        # Model weights are the output of the training run saved as .pth
        detector_model = app_directory + "/map/operators/models/detector_version3.pth"
        mmdet_config = app_directory + "/detector/mmdetector.py"
        classifier_model = (
            app_directory + "/map/operators/models/classifier_version2.pth"
        )

        # Build operators:
        loader = DICOMDataLoaderOperator()
        selector = DICOMSeriesSelectorOperator(
            rules=json.dumps(config["input"]["selection_rules"]), all_matched=True
        )

        # TODO: Define project-sepcific operators, eg
        example_detector = DetectorOperator(
            detector_model=detector_model, mmdet_config=mmdet_config
        )
        example_classifier = ClassifierOperator(classifier_model=classifier_model)
        generate_pdf = GeneratePDFOperator(detector_model=detector_model)

        dicom_encapsulation = DICOMEncapsulatedPDFWriterOperator(
            copy_tags=True, model_info=model_info
        )

        # Create flow - stringing together the operators, specifying inputs and outputs
        # TODO: update flow as appropriate with project specific operators
        self.add_flow(loader, selector, {"dicom_study_list": "dicom_study_list"})
        self.add_flow(
            selector,
            example_detector,
            {"study_selected_series_list": "study_selected_series_list"},
        )

        # EXAMPLE Classifier requires 2 inputs
        self.add_flow(
            selector,
            example_classifier,
            {"study_selected_series_list": "study_selected_series_list"},
        )
        self.add_flow(
            example_detector, example_classifier, {"bbox_array": "bbox_array"}
        )

        # EXAMPLE PDF Generator requires 2 inputs
        self.add_flow(
            selector,
            generate_pdf,
            {"study_selected_series_list": "study_selected_series_list"},
        )
        self.add_flow(
            example_classifier, generate_pdf, {"output_udict": "output_udict"}
        )

        # EXAMPLE dicom_encapsulation requires 2 inputs
        self.add_flow(
            selector,
            dicom_encapsulation,
            {"study_selected_series_list": "study_selected_series_list"},
        )
        self.add_flow(generate_pdf, dicom_encapsulation, {"pdf_file": "pdf_file"})

        self._logger.debug(f"End {self.compose.__name__}")


if __name__ == "__main__":
    myApplication(do_run=True)
