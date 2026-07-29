import os
import shutil
import glob
import tempfile
import uuid
import logging
import json
import xnat
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List

from sklearn.model_selection import train_test_split
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from xnat.mixin import ImageScanData, SubjectData

from shared.xnat_tools import DataBuilderXNAT
from src.utils.nifti_tools import nifti_contour_combiner

pd.set_option('display.max_columns', None, 'display.max_colwidth', None)

logger = logging.getLogger(__name__)


class DataModule_nnUNetV2():
    """From XNAT data, create an nnU-Net v2 compatible dataset.

    This is not a lightning ``DataModule``!

    nnU-Net performs its own data loading/augmentation, so this class only creates the
    nnU-Net v2 directory schema on disk (``DatasetXXX_<model>/{imagesTr,labelsTr,imagesTs}``
    plus ``dataset.json``) from data held on XNAT.

    XNAT expected structure: each subject has, attached to an RTSTRUCT scan, a NIFTI resource
    that contains a pre-converted image volume NIfTI (.nii.gz) and one .nii.gz per labelled
    structure.

    Per-structure NIfTIs are combined into a single multi-label mask via
    ``nifti_contour_combiner``. Which image/contour files to use is defined in ``regions.json``.
    """

    def __init__(self, xnat_configuration: dict,
                 tmp_dirs_configuration: dict,
                 regions_json_path: str,
                 xnat_download_num_workers: int = 4,
                 train_fraction: float = 0.9,
                 test_fraction: float = 0.1,
                 random_seed: int = 42,
                 ):
        self.xnat_configuration = xnat_configuration
        self.tmp_dirs_configuration = tmp_dirs_configuration
        self.regions_json_path = regions_json_path
        self.xnat_download_num_workers = xnat_download_num_workers
        self.train_fraction = train_fraction
        self.test_fraction = test_fraction
        self.random_seed = random_seed

    def make_tmp_dir(self) -> None:
        """
        Make temporary working directory in OS tmp folder in the format:
        <folder_prefix>_<datetime>_<uuid>
        """

        if not os.path.isdir(self.tmp_dirs_configuration["os_tmp_dir"]):
            raise Exception("Operating system tmp directory not found.")

        self.tmp_working_dir = os.path.join(
            self.tmp_dirs_configuration["os_tmp_dir"],
            self.tmp_dirs_configuration["tmp_working_dir"]
            + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_" + uuid.uuid4().hex[:8]
            )

        os.makedirs(self.tmp_working_dir, exist_ok=True)
        logger.debug(f'Created temporary working directory: {self.tmp_working_dir}')

    def make_nnunet_base_dirs(self) -> None:
        """
        Make nnU-Net base folder structure (raw / results / preprocessed).
        """

        if not os.path.isdir(self.tmp_working_dir):
            raise Exception("Working directory not found or does not exist.")

        self.nnunet_raw_dir = os.path.join(self.tmp_working_dir, self.tmp_dirs_configuration["nnunet_raw_dir"])
        self.nnunet_results_dir = os.path.join(self.tmp_working_dir, self.tmp_dirs_configuration["nnunet_results_dir"])
        self.nnunet_preprocessed_dir = os.path.join(self.tmp_working_dir, self.tmp_dirs_configuration["nnunet_preprocessed_dir"])

        for folder in (self.nnunet_raw_dir, self.nnunet_results_dir, self.nnunet_preprocessed_dir):
            os.makedirs(folder, exist_ok=True)

        logger.debug(f'Created nnUNet directory structure: {os.listdir(self.tmp_working_dir)}')

    def make_nnunet_dataset_dirs(self, modelnames: List = None) -> None:
        """
        Make nnU-Net DatasetXXX sub-directory structure
            e.g.
                Dataset001_Lungs/
                    imagesTr/
                    imagesTs/
                    labelsTr/

        Inputs:
            modelnames: List of modelnames used to populate DatasetXXX folder names
        """

        if not os.path.isdir(self.tmp_working_dir):
            raise Exception("Working directory not found or does not exist.")
        if not os.path.isdir(self.nnunet_raw_dir):
            raise Exception(f"{self.nnunet_raw_dir} not found or does not exist.")
        if isinstance(modelnames, str):
            modelnames = [modelnames]

        ctr = 1
        for modelname in modelnames:  # note: loop currently redundant; workflow deals with one DatasetXXX folder
            self.dataset_dir_name = "Dataset" + "{0:03}".format(ctr) + "_" + modelname
            for folder in [
                "imagesTr",
                "imagesTs",
                "labelsTr",
                "labelsTs",
            ]:
                os.makedirs(os.path.join(self.nnunet_raw_dir, self.dataset_dir_name, folder), exist_ok=True)

            ctr += 1

    def get_xnat_data(self) -> None:
        """
        Fetches raw XNAT data and stores in raw_data attribute
        """

        actions = [(self.fetch_resource_nifti, "nifti")]

        data_builder = DataBuilderXNAT(self.xnat_configuration,
                                       actions=actions,
                                       num_workers=self.xnat_download_num_workers)

        data_builder.fetch_data()
        self.raw_data = data_builder.dataset

    def validate_data(self, files_to_validate: List = None) -> None:
        """
        Validate XNAT object contains necessary files for training

        Returns:
            file_validation_dict: Nested Dict containing subject IDs and boolean depending if file exists in XNAT object,
                e.g.:
                    file_validation_dict = {
                        SUBJECT_1
                            CONTOUR1.nii.gz: True
                            CONTOUR2.nii.gz: True
                            CONTOUR3.nii.gz: True
                        SUBJECT_2
                            CONTOUR1.nii.gz: True
                            CONTOUR2.nii.gz: False
                            CONTOUR3.nii.gz: False
                        etc.
                    }
        """

        if not files_to_validate:
            raise Warning("List of regions to validate not found or missing.")
        if not self.raw_data:
            raise Exception("raw_data not found.")
        else:
            xnat_data_obj = self.raw_data
        if type(files_to_validate) == str:
            files_to_validate = [files_to_validate]

        # validate files for training exist in XNAT data object
        logger.info("Validating specified files exist in XNAT object:")
        for fl in files_to_validate:
            logger.info(f"{fl}")

        # initialise empty file validation dictionary
        file_validation_dict = {}
        for subject in xnat_data_obj:
            file_validation_dict[subject['subject_id']] = {}
            for fl in files_to_validate:
                file_validation_dict[subject['subject_id']][fl] = None

        # check each subject in XNAT object contains specified files, update validation dictionary accordingly
        for subject in xnat_data_obj:
            logger.info(f"Validating subject: {subject['subject_id']} ...")
            subject_files = subject['data'][0]['resource_files']
            for fl in files_to_validate:
                if fl in subject_files:
                    logger.info(f"{fl} exists.")
                    file_validation_dict[subject['subject_id']][fl] = True
                elif fl not in subject_files:
                    logger.info(f"{fl} not found.")
                    file_validation_dict[subject['subject_id']][fl] = False

        # add some metadata
        file_validation_dict['num_files_per_subject'] = len(files_to_validate)

        return file_validation_dict

    def xnat_open_connection(self):
        logger.info("Opening XNAT connection ...")
        xnat_connection = xnat.connect(
                server=self.xnat_configuration['server'],
                user=self.xnat_configuration['user'],
                password=self.xnat_configuration['password'],
                verify=self.xnat_configuration['verify'],
                loglevel='ERROR',
            )
        return xnat_connection

    def xnat_download_session_object(
            self,
            xnat_connection_obj=None,
            subject_xnat_uri: str = None,
            output_dir: str = None,
            expected_filetype: str = ".nii.gz"
    ):
        """
        Open the XNAT Session object for a single subject and download its contents to a
        directory (e.g. all NIfTI files within one subject's Session object).
        """

        session_obj = xnat_connection_obj.create_object(subject_xnat_uri)

        session_obj.download_dir(output_dir, verbose=False)
        file_paths = glob.glob(
            os.path.join(output_dir, '**/*' + expected_filetype), recursive=True)

        return file_paths

    def process_subject(self, row, xnat_connection, contour_column_names):
        """
        Download a subject's XNAT session and write its nnU-Net files.
        Train subject -> combined multi-label mask (labelsTr) + image (imagesTr);
        Test subject  -> combined multi-label mask (labelsTs) + image (imagesTs).

        Test labels are ground truth used only for evaluation - nnU-Net never trains on them.
        """

        with tempfile.TemporaryDirectory() as session_tmp_holding_dir:
            session_file_uris = self.xnat_download_session_object(
                xnat_connection_obj=xnat_connection,
                subject_xnat_uri=row["XNAT_NIFTI_DATA_URI"],
                output_dir=session_tmp_holding_dir,
            )

            def find(filename):
                match = next((uri for uri in session_file_uris if filename in uri), None)
                if match is None:
                    raise FileNotFoundError(
                        f"Expected file '{filename}' not found in the XNAT session download."
                    )
                return match

            if row["IS_TRAIN_SUBJECT"]:
                image_dest, label_dest, kind = row["TRAIN_IMAGE_DEST_URI"], row["TRAIN_LABEL_DEST_URI"], "training"
            elif row["IS_TEST_SUBJECT"]:
                image_dest, label_dest, kind = row["TEST_IMAGE_DEST_URI"], row["TEST_LABEL_DEST_URI"], "test"
            else:
                return

            # Label: combine the subject's contour NIfTIs into one multi-label mask.
            # Filename order sets the label order (contour 1 -> 1, contour 2 -> 2, ...).
            contour_uris = [find(fl) for fl in row[contour_column_names].tolist()]
            combined_uri = os.path.join(session_tmp_holding_dir, "tmp_nifti.nii.gz")
            nifti_contour_combiner(input_nifti_uris=contour_uris, output_nifti_uri=combined_uri)
            shutil.copy(combined_uri, label_dest)
            shutil.copy(find(row["IMAGE_DATA_FILE_NAME"]), image_dest)
            logger.info(f"Wrote {kind} label + image for subject index {row['SUBJECT_INDEX']}")

    def _dataset_dest_uri(self, idx, subfolder, suffix):
        """Build a Struct_NNN destination path inside the dataset's nnU-Net subfolder."""

        return os.path.join(
            self.nnunet_raw_dir,
            self.dataset_dir_name,
            subfolder,
            "Struct_" + "{0:03}".format(idx) + suffix,
        )

    def setup(self):
        """
        Prepare the nnU-Net dataset: pull from XNAT, validate, split, download, and write
        ``dataset.json``.
        """
        
        # 1. initialise dir structure
        self.make_tmp_dir()
        self.make_nnunet_base_dirs()


        # 2. get XNAT object
        self.get_xnat_data()


        # 3. Organise data
        df = pd.DataFrame(
            [
                {
                    'SUBJECT_ID': subject['subject_id'],
                    'XNAT_SUBJECT_URI': subject['subject_uri'],
                    'XNAT_NIFTI_DATA_URI': subject['data'][0]['action_data'],
                }
                for subject in self.raw_data
            ]
        )


        # 4. Parse regions.json file once (improvable as uses a single filename/modelname)
        if not (os.path.isfile(self.regions_json_path) and self.regions_json_path.endswith(".json")):
            raise TypeError("Regions JSON file not found or not specified.")
        
        with open(self.regions_json_path) as jsonfile:
            regions_cfg = json.load(jsonfile)
                    
        image_data_filename = regions_cfg["image_filenames"][0]  # TODO permit >1 image_data filename
        regions_to_train = regions_cfg["training_models"]
        modelname = regions_to_train[0]['modelname']             # TODO permit >1 modelname?
        contour_filenames = regions_to_train[0]['contour_filenames']
        image_channel_name = regions_cfg.get("image_channel_name", "CT")


        # 5. Validate Image Data
        image_data_validation = self.validate_data(files_to_validate=image_data_filename)
        df["IS_IMAGE_DATA_FILE"] = df["SUBJECT_ID"].map(lambda sub: image_data_validation[sub][image_data_filename])
        df["IMAGE_DATA_FILE_NAME"] = image_data_filename


        # 6. Validate contour files (single pass)
        contours_validation = self.validate_data(files_to_validate=contour_filenames)
        for contour_idx, fl in enumerate(contour_filenames):
            col_name = f"CONTOUR_{contour_idx+1}_FILE_NAME"
            col_status = f"IS_CONTOUR_{contour_idx+1}_FILE"
            
            df[col_status] = df["SUBJECT_ID"].map(lambda sub: contours_validation[sub][fl])
            df[col_name] = fl


        # 7. Check all contours present per subject and filter down
        is_contour_columns = [f"IS_CONTOUR_{i+1}_FILE" for i in range(len(contour_filenames))]        
        df["IS_ALL_CONTOUR_FILES"] = df[is_contour_columns].all(axis=1)

        logger.info(f"Number of subjects with all required contours: {df['IS_ALL_CONTOUR_FILES'].sum()}")
        logger.info(f"Number of subjects without all required contours: {(~df['IS_ALL_CONTOUR_FILES']).sum()}")        
        df = df[df["IS_ALL_CONTOUR_FILES"]].reset_index(drop=True)


        # 8. Create folder paths for nnU-Net
        self.make_nnunet_dataset_dirs(modelnames=modelname)


        # 9. Data split
        # note, for nnU-Net v2:
        # len(imagesTr) == len(labelsTr) == train_size
        # len(imagesTs) == test_size        
        df_train, df_test = train_test_split(
            df, train_size=self.train_fraction, test_size=self.test_fraction,
            random_state=self.random_seed
        )

        logger.info(f'{len(df_train)} cases in training set, out of {len(df)} total cases.')
        logger.info(f'{len(df_test)} cases in test set, out of {len(df)} total cases.')

        logger.info(f'Training Fraction = {self.train_fraction}')
        logger.info(f'Test Fraction = {self.test_fraction}')
        if self.train_fraction + self.test_fraction < 1:
            logger.warning('Train/Test split ratio < 1. Proceeding with subset of all data')        

        df["IS_TRAIN_SUBJECT"] = df.index.isin(df_train.index)
        df["IS_TEST_SUBJECT"] = df.index.isin(df_test.index)

        # 10. Map destination paths
        df["SUBJECT_INDEX"] = df.index + 1
        
        df["TRAIN_LABEL_DEST_URI"] = df.apply(
            lambda row: self._dataset_dest_uri(row["SUBJECT_INDEX"], "labelsTr", ".nii.gz") if row["IS_TRAIN_SUBJECT"] else None, axis=1
        )
        df["TRAIN_IMAGE_DEST_URI"] = df.apply(
            lambda row: self._dataset_dest_uri(row["SUBJECT_INDEX"], "imagesTr", "_0000.nii.gz") if row["IS_TRAIN_SUBJECT"] else None, axis=1
        )
        df["TEST_IMAGE_DEST_URI"] = df.apply(
            lambda row: self._dataset_dest_uri(row["SUBJECT_INDEX"], "imagesTs", "_0000.nii.gz") if row["IS_TEST_SUBJECT"] else None, axis=1
        )
        df["TEST_LABEL_DEST_URI"] = df.apply(
            lambda row: self._dataset_dest_uri(row["SUBJECT_INDEX"], "labelsTs", ".nii.gz") if row["IS_TEST_SUBJECT"] else None, axis=1
        )

        self.df = df


        # 11. Download each subject's XNAT session ONCE (reusing a single connection) and write
        #     its nnU-Net files from that single download (train: label + image; test: image).
        contour_column_names = [f"CONTOUR_{i+1}_FILE_NAME" for i in range(len(contour_filenames))]

        logger.info("Downloading subjects from XNAT and writing nnU-Net dataset files ...")
        with self.xnat_open_connection() as xnat_connection:
            for _, row in df[df["IS_TRAIN_SUBJECT"] | df["IS_TEST_SUBJECT"]].iterrows():
                self.process_subject(row, xnat_connection, contour_column_names)


        # 12. Construct dataset.json
        labels = ["background"] + [c.replace(".nii.gz", "").replace("Struct_", "").lower() for c in contour_filenames]
        label_values = list(range(len(labels)))

        self.generate_dataset_json(
            num_training_cases=int(df["IS_TRAIN_SUBJECT"].sum()),
            labels_dict=dict(zip(labels, label_values)),
            image_channel_name=image_channel_name,
        )

        logger.info("DataModule_nnUNetV2.setup() complete.")


    @staticmethod
    def fetch_resource_nifti(subject_data: SubjectData = None) -> List[ImageScanData]:
        """
        Walk the XNAT SubjectData object and return the ScanData "NIFTI" Resource object
        attached to the subject's RTSTRUCT scan.

        Note: this returns an XNAT object describing the files, not the files themselves;
        downloading happens later in ``copy_files_xnat_to_destination``.
        """

        output = []
        for exp in subject_data.experiments:
            for scan in exp.scans:
                # Identify RTSTRUCT XNAT ScanData object
                if scan.modality.lower() == 'rtstruct':
                    for resource in scan.resources:
                        # Identify NIFTI XNAT Resource object
                        if resource.label.lower() == 'nifti':
                            output.append(resource)
        if len(output) > 1:
            raise TypeError("More than one NIFTI resource found for subject.")
        return output

    def generate_dataset_json(self, num_training_cases: int, labels_dict: dict, image_channel_name: str = "CT"):
        """
        Invoke nnU-Net's ``generate_dataset_json`` to create the dataset.json file.

        Inputs:
            num_training_cases - number of training datasets
            labels_dict - dict of contour label names, e.g. {"background": 0, "lungs": 1}
            image_channel_name - input channel name written to dataset.json (e.g. "CT", "MR")
        """
        
        generate_dataset_json(
            os.path.join(self.nnunet_raw_dir, self.dataset_dir_name),
            channel_names={
                0: image_channel_name
            },
            labels=labels_dict,
            file_ending=".nii.gz",
            num_training_cases=num_training_cases
        )
        logger.info(f"Generated dataset.json (channel 0 = {image_channel_name}).")