import glob
import logging
import os
import tempfile
import time
from monai.config import KeysCollection
from monai.transforms import MapTransform, LoadImage
from monai.transforms import Transform
from platipy.dicom.io.rtstruct_to_nifti import convert_rtstruct
import xnat

logger = logging.getLogger(__name__)

class LoadImageXNAT3Dd(MapTransform):
    """
    MapTransform for importing 3D image and label data from XNAT
    Converts DICOM + RTSTRUCT to NIfTI.
    """

    def __init__(self, keys: KeysCollection, xnat_configuration: dict = None,
                 image_loader: Transform = LoadImage(), expected_filetype_ext: str = '.dcm',
                 validate_data: bool = False, verbose=False):
        super().__init__(keys)
        self.image_loader = image_loader
        self.xnat_configuration = xnat_configuration
        self.expected_filetype = expected_filetype_ext
        self.validate_data = validate_data
        self.verbose = verbose

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            if key not in data:
                continue

            with xnat.connect(server=self.xnat_configuration['server'],
                              user=self.xnat_configuration['user'],
                              password=self.xnat_configuration['password'],
                              verify=self.xnat_configuration.get('verify', True),
                              loglevel='ERROR') as session:

                for item in d[key]:
                    data_label = item['data_label']
                    if item['data_type'] == 'value':
                        d[data_label] = item['action_data']
                        continue

                    attempts = 0
                    while attempts < 3:
                        try:
                            with tempfile.TemporaryDirectory() as tmpdirname:
                                session_obj = session.create_object(item['action_data'])
                                session_obj.download_dir(tmpdirname, verbose=self.verbose)

                                # Collect DICOM files
                                images_path = glob.glob(os.path.join(tmpdirname, '**/*' + self.expected_filetype),
                                                        recursive=True)
                                if not images_path:
                                    raise FileNotFoundError(f"No DICOM files found for {item}")

                                # Single series check
                                image_dirs = list(set(os.path.dirname(p) for p in images_path))
                                if len(image_dirs) > 1:
                                    raise ValueError(f"Multiple image series found for {item}")

                                # Convert RTSTRUCT to NIfTI if label
                                if data_label == 'label' and 'action_label' in item:
                                    convert_rtstruct(
                                        input_dcm_dirname=image_dirs[0],
                                        input_rt_filename=item['action_data'],
                                        output_dir=item.get('output_dir', tmpdirname),
                                        output_img=item.get('output_image', os.path.join(tmpdirname, "image.nii.gz")),
                                        prefix=item.get('prefix', "")
                                    )
                                    d[data_label] = os.path.join(tmpdirname, "label.nii.gz")
                                else:
                                    # Load 3D image
                                    d[data_label] = self.image_loader(image_dirs[0])

                                break

                        except Exception as e:
                            attempts += 1
                            time.sleep(1.0)
                            if attempts == 3:
                                raise RuntimeError(f"XNAT loader failed for {item} after 3 attempts: {e}")

        return d
