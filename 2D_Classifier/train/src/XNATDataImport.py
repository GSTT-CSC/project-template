import logging
from tqdm import tqdm
from typing import List

from utils.tools import DataBuilderXNAT
from xnat.mixin import ImageScanData, SubjectData

from src.transforms import load_xnat
from monai.data import Dataset
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class XNATDataImport():
    
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
                for scan in subject_data.experiments[exp].scans:
                    if 'cr' in subject_data.experiments[exp].scans[scan].modality.lower() or 'dx' in subject_data.experiments[exp].scans[scan].modality.lower():
                        scan_objects.append(subject_data.experiments[exp].scans[scan].uri)
       
        return scan_objects
    
    @staticmethod
    def fetch_label(subject_data: SubjectData = None):
        """
        Function that identifies and returns the label from a SubjectData object
        """
        for exp in subject_data.experiments:
            for scan in subject_data.experiments[exp].scans:
                    if 'cr' in subject_data.experiments[exp].scans[scan].modality.lower() or 'dx' in subject_data.experiments[exp].scans[scan].modality.lower():
                        try:
                            full_label = subject_data.experiments[exp].label
                            return full_label
                        except Exception as e:
                            logger.warning(f"Unable to fetch {subject_data.experiments[exp]}'s label due to exception: {e}")