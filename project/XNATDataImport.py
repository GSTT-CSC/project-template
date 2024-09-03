import os
import logging
import pandas as pd
import numpy as np

from utils.tools import DataBuilderXNAT
from xnat.mixin import ImageScanData, SubjectData
from typing import List, Dict

logger = logging.getLogger(__name__)

class XNATDataImport():
    
    def __init__(self, xnat_configuration: dict = None, num_workers: int = 4):
        self.xnat_configuration = xnat_configuration
        self.num_workers = num_workers

    def import_xnat_data(self):
        actions = [
            (self.fetch_xr, "image"),
            (self.fetch_label, "label"),
        ]

        data_builder = DataBuilderXNAT(
            self.xnat_configuration, actions=actions, num_workers=self.num_workers
        )

        data_builder.fetch_data()
        return(data_builder.dataset)

    @staticmethod
    def fetch_xr(subject_data: SubjectData = None) -> List[ImageScanData]:
        """
        Function that identifies and returns the required xnat ImageData object from a xnat SubjectData object
        along with the 'key' that it will be used to access it.
        """

        scan_objects = []

        for exp in subject_data.experiments:
            if (
                "CR" in subject_data.experiments[exp].modality
                or "DX" in subject_data.experiments[exp].modality
            ):
                for scan in subject_data.experiments[exp].scans:
                    scan_objects.append(subject_data.experiments[exp].scans[scan])
        return scan_objects

    @staticmethod
    def fetch_label(subject_data: SubjectData = None):
        """
        Function that identifies and returns the required label from a XNAT SubjectData object.
        """
        label = None
        for exp in subject_data.experiments:
            if (
                "CR" in subject_data.experiments[exp].modality
                or "DX" in subject_data.experiments[exp].modality
            ):
                temp_label = subject_data.experiments[exp].label
                x = temp_label.split("_")
                label = int(x[1])

        return label
