import os
import json
import shutil
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import nibabel as nib
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Try to import nnUNet's dataset JSON generator
try:
    from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
    NNUNET_AVAILABLE = True
except ImportError:
    NNUNET_AVAILABLE = False

logger = logging.getLogger(__name__)


class DataModule_nnUNetV2:
    """
    DataModule for 3D medical image segmentation
    Handles data import from XNAT and preparation for nnUNetV2
    """
    
    def __init__(
        self,
        xnat_configuration: Dict,
        train_fraction: float = 0.9,
        test_fraction: float = 0.1,
        num_workers: int = 4,
        tmp_dirs_configuration: Optional[Dict] = None,
        regions_json_path: Optional[str] = None,
        random_seed: int = 42
    ):
        """
        initialise DataModule

        Args:
            xnat_configuration: XNAT connection parameters
            train_fraction: Fraction of data for training
            test_fraction: Fraction of data for testing
            num_workers: Number of parallel workers
            tmp_dirs_configuration: Temporary directory configuration
            regions_json_path: Path to regions definition JSON
            random_seed: Random seed for reproducibility
        """
        self.xnat_config = xnat_configuration
        self.train_fraction = train_fraction
        self.test_fraction = test_fraction
        self.num_workers = num_workers
        self.random_seed = random_seed
        
        # setup directories
        self._setup_directories(tmp_dirs_configuration)
        
        # load regions configuration if provided
        self.regions = self._load_regions(regions_json_path)
        
        # initialise data containers
        self.data_manifest = {
            "train": [],
            "validation": [],
            "test": []
        }
        self.dataset_info = {}
        self.DF = None  # dataFrame for tracking data
        
        # nnUNet dataset naming
        self.dataset_id = self._get_next_dataset_id()
        self.dataset_dir_name = f"Dataset{self.dataset_id:03d}_{xnat_configuration['project']}"
        
        logger.info(f"Initialised DataModule for dataset: {self.dataset_dir_name}")
        
    def _setup_directories(self, tmp_dirs_config: Optional[Dict]):
        """Setup working directories"""
        if tmp_dirs_config:
            self.nnunet_raw_dir = Path(tmp_dirs_config.get("nnunet_raw_dir", "nnUNet_raw"))
            self.nnunet_results_dir = Path(tmp_dirs_config.get("nnunet_results_dir", "nnUNet_results"))
            self.nnunet_preprocessed_dir = Path(tmp_dirs_config.get("nnunet_preprocessed_dir", "nnUNet_preprocessed"))
            self.tmp_dir = Path(tmp_dirs_config.get("os_tmp_dir", "/tmp")) / tmp_dirs_config.get("tmp_working_dir", "nnunet_tmp")
        else:
            self.nnunet_raw_dir = Path("nnUNet_raw")
            self.nnunet_results_dir = Path("nnUNet_results")
            self.nnunet_preprocessed_dir = Path("nnUNet_preprocessed")
            self.tmp_dir = Path("/tmp/nnunet_tmp")
            
        # create directories
        for dir_path in [self.nnunet_raw_dir, self.nnunet_results_dir, 
                         self.nnunet_preprocessed_dir, self.tmp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def _load_regions(self, regions_json_path: Optional[str]) -> Dict:
        """Load regions configuration from JSON"""
        if regions_json_path and os.path.exists(regions_json_path):
            with open(regions_json_path, 'r') as f:
                regions = json.load(f)
                logger.info(f"Loaded {len(regions)} regions from {regions_json_path}")
                return regions
        else:
            logger.warning("No regions JSON provided or file not found. Using default configuration.")
            # default single-class segmentation
            return {
                "background": 0,
                "foreground": 1
            }
            
    def _get_next_dataset_id(self) -> int:
        """Get next available dataset ID for nnUNet"""
        existing_datasets = []
        if self.nnunet_raw_dir.exists():
            for item in self.nnunet_raw_dir.iterdir():
                if item.is_dir() and item.name.startswith("Dataset"):
                    try:
                        dataset_id = int(item.name[7:10])
                        existing_datasets.append(dataset_id)
                    except (ValueError, IndexError):
                        continue
                        
        return max(existing_datasets, default=0) + 1
        
    def setup(self):
        """Main setup method - imports data and prepares nnUNet structure"""
        logger.info("Starting data setup...")
        
        # import data from XNAT
        self._import_xnat_data()
        
        # split data into train/test
        self._split_data()
        
        # create nnUNet dataset structure
        self._create_nnunet_dataset()
        
        # generate dataset.json for nnUNet
        self._create_dataset_json()
        
        logger.info("Data setup completed successfully")
        
    def _import_xnat_data(self):
        """Import data from XNAT"""
        try:
            from XNATDataImport import XNATDataImport
            
            importer = XNATDataImport(
                xnat_configuration=self.xnat_config,
                num_workers=self.num_workers
            )
            
            # import metadata
            logger.info("Importing data from XNAT...")
            raw_data = importer.import_xnat_data()
            
            # download images
            logger.info("Downloading images from XNAT...")
            self.data = importer.xnat_image_download(raw_data)
            
            # create DataFrame for tracking
            self.DF = pd.DataFrame(self.data)
            logger.info(f"Imported {len(self.data)} subjects from XNAT")
            
        except ImportError:
            logger.warning("XNATDataImport not available. Using mock data for testing.")
            self._create_mock_data()
            
    def _create_mock_data(self):
        """Create mock data for testing without XNAT"""
        logger.info("Creating mock data for testing...")
        
        mock_data = []
        for i in range(10):  # create 10 mock subjects
            mock_data.append({
                'subject_id': f'MOCK_{i:03d}',
                'image_path': None,  # Would be actual path in real scenario
                'label_path': None,  # Would be actual path in real scenario
                'metadata': {
                    'modality': 'CT',
                    'spacing': [1.0, 1.0, 1.0]
                }
            })
            
        self.data = mock_data
        self.DF = pd.DataFrame(mock_data)
        
    def _split_data(self):
        """Split data into train, validation, and test sets"""
        np.random.seed(self.random_seed)
        
        n_total = len(self.data)
        n_train = int(n_total * self.train_fraction)
        n_test = int(n_total * self.test_fraction)
        n_val = n_total - n_train - n_test
        
        # shuffle indices
        indices = np.random.permutation(n_total)
        
        # split indices
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        # assign data to splits
        for idx in train_idx:
            self.data_manifest["train"].append(self.data[idx])
        for idx in val_idx:
            self.data_manifest["validation"].append(self.data[idx])
        for idx in test_idx:
            self.data_manifest["test"].append(self.data[idx])
            
        logger.info(f"Data split - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
    def _create_nnunet_dataset(self):
        """Create nnUNet dataset directory structure"""
        dataset_path = self.nnunet_raw_dir / self.dataset_dir_name
        
        # create subdirectories
        (dataset_path / "imagesTr").mkdir(parents=True, exist_ok=True)
        (dataset_path / "labelsTr").mkdir(parents=True, exist_ok=True)
        (dataset_path / "imagesTs").mkdir(parents=True, exist_ok=True)
        (dataset_path / "labelsTs").mkdir(parents=True, exist_ok=True)  # Optional for test labels
        
        logger.info(f"Created nnUNet dataset structure at: {dataset_path}")
        
        # copy/convert images to nnUNet format
        self._prepare_nnunet_data(dataset_path)
        
    def _prepare_nnunet_data(self, dataset_path: Path):
        """Prepare and copy data to nnUNet format"""
        logger.info("Preparing data in nnUNet format...")
        
        # process training data
        train_val_data = self.data_manifest["train"] + self.data_manifest["validation"]
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            
            # Submit training data tasks
            for idx, item in enumerate(train_val_data):
                future = executor.submit(
                    self._process_single_case,
                    item,
                    dataset_path / "imagesTr",
                    dataset_path / "labelsTr",
                    f"case_{idx:04d}"
                )
                futures.append(future)
                
            # submit test data tasks
            for idx, item in enumerate(self.data_manifest["test"]):
                future = executor.submit(
                    self._process_single_case,
                    item,
                    dataset_path / "imagesTs",
                    dataset_path / "labelsTs",
                    f"case_{idx:04d}"
                )
                futures.append(future)
                
            # wait for completion with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing cases"):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing case: {e}")
                    
    def _process_single_case(
        self,
        item: Dict,
        image_dir: Path,
        label_dir: Path,
        case_name: str
    ):
        """Process a single case for nnUNet format"""
        # nnUNet naming convention: {case_name}_0000.nii.gz for images
        # {case_name}.nii.gz for labels
        
        # this is a placeholder - actual implementation would:
        # 1. Load the image from item['image_path']
        # 2. Convert to NIfTI format if needed
        # 3. Ensure correct spacing and orientation
        # 4. Save with nnUNet naming convention
        
        # For mock data, create empty files
        image_path = image_dir / f"{case_name}_0000.nii.gz"
        label_path = label_dir / f"{case_name}.nii.gz"
        
        # in real implementation, you would copy/convert actual data
        image_path.touch()
        label_path.touch()
        
    def _create_dataset_json(self):
        """Create dataset.json file for nnUNet"""
        dataset_path = self.nnunet_raw_dir / self.dataset_dir_name
        
        # determine number of training cases
        n_training = len(list((dataset_path / "imagesTr").glob("*.nii.gz")))
        
        # prepare labels dict (name -> index)
        labels_dict = {name: int(i) for name, i in self.regions.items()}
        
        if NNUNET_AVAILABLE:
            # Use nnUNet's official dataset.json generator (preferred)
            generate_dataset_json(
                str(dataset_path),
                channel_names={0: "CT"},
                labels=labels_dict,
                file_ending=".nii.gz",
                num_training_cases=n_training
            )
            logger.info(f"Created dataset.json using nnUNet generator with {n_training} training cases")
        else:
            # Fallback to manual creation
            dataset_json = {
                "channel_names": {
                    "0": "CT"  # or "MRI" depending on modality
                },
                "labels": labels_dict,
                "numTraining": n_training,
                "file_ending": ".nii.gz",
                "overwrite_image_reader_writer": "NibabelIOWithReorient"
            }
            
            # save dataset.json
            dataset_json_path = dataset_path / "dataset.json"
            with open(dataset_json_path, 'w') as f:
                json.dump(dataset_json, f, indent=2)
                
            logger.info(f"Created dataset.json manually with {n_training} training cases")
            self.dataset_info = dataset_json
        
    def get_data_statistics(self) -> Dict:
        """Get statistics about the dataset"""
        stats = {
            "total_cases": len(self.data),
            "train_cases": len(self.data_manifest["train"]),
            "validation_cases": len(self.data_manifest["validation"]),
            "test_cases": len(self.data_manifest["test"]),
            "num_classes": len(self.regions),
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_dir_name
        }
        
        return stats
        
    def cleanup(self):
        """Clean up temporary files"""
        if self.tmp_dir.exists():
            shutil.rmtree(self.tmp_dir)
            logger.info("Cleaned up temporary files")


# additional utility class for XNAT interaction (placeholder)
class XNATDataImport:
    """
    Placeholder for XNAT data import functionality
    In production, this would handle actual XNAT API calls
    """
    
    def __init__(self, xnat_configuration: Dict, num_workers: int = 4):
        self.xnat_config = xnat_configuration
        self.num_workers = num_workers
        
    def import_xnat_data(self) -> List[Dict]:
        """Import metadata from XNAT"""
        # placeholder - would query XNAT API
        logger.info("Importing data from XNAT (mock)")
        return []
        
    def xnat_image_download(self, raw_data: List[Dict]) -> List[Dict]:
        """Download images from XNAT"""
        # placeholder - would download actual images
        logger.info("Downloading images from XNAT (mock)")
        return raw_data


def get_training_region_filenames(regions_json_path: str = None) -> List:
    """
    Fetches list of filenames corresponding to regions for model training
    
    Inputs:
        regions_json_path: path to regions.json file
    
    Outputs:
        regions_to_train: List of region filenames to use for model training
    """
    if os.path.isfile(regions_json_path) and regions_json_path.endswith(".json"):
        with open(regions_json_path) as jsonfile:
            json_dict = json.load(jsonfile)
            regions_to_train = json_dict["training_models"]
            return regions_to_train
    else:
        raise TypeError("Training regions file not found or not specified.")


def get_image_data_filenames(regions_json_path: str = None) -> List:
    """
    Fetches list of filenames corresponding to image data for model training
    
    Inputs:
        regions_json_path: path to regions.json file
    
    Outputs:
        image_data_to_train: List of image data filenames to use for model training
    """
    if os.path.isfile(regions_json_path) and regions_json_path.endswith(".json"):
        with open(regions_json_path) as jsonfile:
            json_dict = json.load(jsonfile)
            image_data_to_train = json_dict["image_filenames"]
            return image_data_to_train
    else:
        raise TypeError("Image data file not found or not specified.")