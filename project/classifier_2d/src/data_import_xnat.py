import logging
from tqdm import tqdm
from typing import List

from monai.data import Dataset
from torch.utils.data import DataLoader
from xnat.mixin import ImageScanData, SubjectData

from src.transforms import load_xnat
from shared.xnat_tools import DataBuilderXNAT

logger = logging.getLogger(__name__)

class DataImportXNAT():
    
    def __init__(self, xnat_configuration: dict = None, num_workers: int = 4, test_batch: int = 0,
                 n_month_data_window=9999, run_type: str='train'):
        self.xnat_configuration = xnat_configuration
        self.num_workers = 4
        self.test_batch = test_batch
        self.run_type = run_type

    def import_xnat_data(self):
        
        if self.run_type == 'train':
            actions = [
                (self.fetch_xr, 'image'),
                (self.fetch_label, 'label'),
            ]  # list of tuples defining action functions and their data keys

        data_builder = DataBuilderXNAT(self.xnat_configuration,
                                       actions=actions,
                                       test_batch=self.test_batch,
                                       num_workers=self.num_workers)

        data_builder.fetch_data()
        return(data_builder.dataset)
    
    def xnat_image_download(self, data):

        raw_dataset = Dataset(
                        data=data,
                        transform=load_xnat(self.xnat_configuration)
                        )
        
        raw_dataloader = DataLoader(raw_dataset, batch_size=None,
                          num_workers=self.num_workers,shuffle=False, pin_memory=False)

        extracted_data = []
        for sample in tqdm(raw_dataloader, "Downloading Image Data"):
            extracted_data.append(sample)

        return(extracted_data)

    @staticmethod
    def fetch_xr(subject_data: SubjectData = None) -> List[ImageScanData]:
        """
        Function that identifies and returns the required xnat ImageData object from a xnat SubjectData object
        along with the 'key' that it will be used to access it.
        """
        
        scan_objects = []

        for exp in subject_data.experiments:
            for scan in exp.scans:
                if 'cr' in scan.modality.lower() or 'dx' in scan.modality.lower():
                    scan_objects.append(scan.uri)

        return scan_objects
    
    @staticmethod
    def fetch_label(subject_data: SubjectData = None):
        """
        Function that identifies and returns the label from a SubjectData object
        """
        for exp in subject_data.experiments:
            for scan in exp.scans:
                if 'cr' in scan.modality.lower() or 'dx' in scan.modality.lower():
                    try:
                        return exp.label
                    except Exception as e:
                        logger.warning(f"Unable to fetch {exp}'s label due to exception: {e}")