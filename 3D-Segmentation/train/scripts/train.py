import sys
import os
import configparser
import json
import subprocess
import logging
import mlflow
from mlflow.tracking import MlflowClient

# Add src directory to Python path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(script_dir), 'src')
sys.path.insert(0, src_dir)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SegmentationTrainer:
    """3D Segmentation trainer using nnUNetV2 framework"""
    
    def __init__(self, config):
        self.config = config
        self.setup_environment()
        self.setup_xnat_config()
        self.setup_nnunet_config()
        self.setup_mlflow()
        
    def setup_environment(self):
        """Set up CUDA environment variables"""
        try:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config["system"]["CUDA_VISIBLE_DEVICES"]
        except KeyError:
            logger.warning("CUDA_VISIBLE_DEVICES not found in config. Defaulting to 0.")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        logger.info(f"CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")
        
    def setup_xnat_config(self):
        """Configure XNAT connection parameters"""
        self.xnat_configuration = {
            "server": self.config["xnat"]["SERVER"],
            "user": self.config["xnat"]["USER"],
            "password": self.config["xnat"]["PASSWORD"],
            "project": self.config["xnat"]["PROJECT"],
            "verify": self.config.getboolean("xnat", "VERIFY"),
        }
        
    def setup_nnunet_config(self):
        """Configure nnUNet parameters"""
        self.nnunet_config = {
            "configuration": self.config["nnunet"]["NNUNET_UNET_CONFIGURATION"],
            "fold": json.loads(self.config["nnunet"]["NNUNET_FOLD"]),
            "trainer": self.config["nnunet"]["NNUNET_TRAINER"],
            "device": self.config["nnunet"]["NNUNET_DEVICE"],
            "num_gpus": self.config["nnunet"]["NNUNET_NUM_GPUS"],
            "npz": self.config["nnunet"]["NNUNET_NPZ"],
        }
        
        # Ensure fold is a list
        if isinstance(self.nnunet_config["fold"], int):
            self.nnunet_config["fold"] = [self.nnunet_config["fold"]]
            
        # Validate GPU configuration
        if self.nnunet_config["device"] == 'cuda':
            gpu_count = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
            if gpu_count != int(self.nnunet_config["num_gpus"]):
                raise ValueError(
                    f"Config error: CUDA_VISIBLE_DEVICES ({gpu_count} GPUs) "
                    f"incompatible with NNUNET_NUM_GPUS ({self.nnunet_config['num_gpus']})"
                )
                
    def setup_mlflow(self):
        """Configure MLflow tracking"""
        if "MLFLOW_TRACKING_URI" in self.config["server"]:
            mlflow.set_tracking_uri(self.config["server"]["MLFLOW_TRACKING_URI"])
        
        experiment_name = self.config["project"]["NAME"]
        mlflow.set_experiment(experiment_name)
        
    def setup_data(self):
        """Set up data module and prepare nnUNet dataset structure"""
        # Import DataModule here to avoid circular imports
        from DataModule_nnUNetV2 import DataModule_nnUNetV2
        
        tmp_dirs_configuration = {
            "os_tmp_dir": self.config["tmp_dirs"]["OS_TMP_DIR"],
            "tmp_working_dir": self.config["tmp_dirs"]["TMP_WORKING_DIR"],
            "nnunet_raw_dir": self.config["nnunet"]["NNUNET_RAW_DIR"],
            "nnunet_results_dir": self.config["nnunet"]["NNUNET_RESULTS_DIR"],
            "nnunet_preprocessed_dir": self.config["nnunet"]["NNUNET_PREPROCESSED_DIR"],
        }
        
        self.dm = DataModule_nnUNetV2(
            xnat_configuration=self.xnat_configuration,
            train_fraction=float(self.config["data"]["TRAIN_FRACTION"]),
            test_fraction=float(self.config["data"]["TEST_FRACTION"]),
            num_workers=4,
            tmp_dirs_configuration=tmp_dirs_configuration,
            regions_json_path=self.config["data"]["REGIONS_JSON_PATH"]
        )
        
        # Setup nnUNet dataset structure
        self.dm.setup()
        
        # Set nnUNet environment variables
        os.environ["nnUNet_raw"] = self.dm.nnunet_raw_dir
        os.environ["nnUNet_preprocessed"] = self.dm.nnunet_preprocessed_dir
        os.environ["nnUNet_results"] = self.dm.nnunet_results_dir
        
        logger.info("nnUNet environment variables set:")
        logger.info(f"  nnUNet_raw = {os.environ['nnUNet_raw']}")
        logger.info(f"  nnUNet_preprocessed = {os.environ['nnUNet_preprocessed']}")
        logger.info(f"  nnUNet_results = {os.environ['nnUNet_results']}")
        
        # Extract dataset ID from directory name
        self.dataset_id = ''.join([i for i in self.dm.dataset_dir_name if i.isdigit()])
        
    def preprocess_data(self):
        """Run nnUNet preprocessing pipeline"""
        logger.info("Running nnUNetv2_plan_and_preprocess...")
        cmd = [
            "nnUNetv2_plan_and_preprocess",
            "-d", self.dataset_id,
            "--verify_dataset_integrity"
        ]
        logger.info(f"Command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logger.info("Preprocessing completed.")
        
    def train_model(self):
        """Execute nnUNet training"""
        logger.info("Starting nnUNetv2 training...")
        logger.info(f"Configuration: {self.nnunet_config['configuration']}")
        logger.info(f"Device: {self.nnunet_config['device']}")
        logger.info(f"Folds: {self.nnunet_config['fold']}")
        
        config = self.nnunet_config["configuration"]
        
        # Handle cascade configuration
        if config == "3d_cascade_fullres":
            # Train 3d_lowres first
            self._train_configuration("3d_lowres")
            # Then train cascade
            self._train_configuration("3d_cascade_fullres")
        else:
            self._train_configuration(config)
            
        logger.info("Training completed.")
        
    def _train_configuration(self, configuration):
        """Train a specific configuration"""
        for fold in self.nnunet_config["fold"]:
            logger.info(f"\nTraining {configuration} - Fold {fold}")
            
            cmd = [
                "nnUNetv2_train",
                self.dataset_id,
                configuration,
                str(fold),
                "-tr", self.nnunet_config["trainer"],
                "-device", self.nnunet_config["device"],
                "-num_gpus", self.nnunet_config["num_gpus"]
            ]
            
            if self.nnunet_config["npz"]:
                cmd.append(self.nnunet_config["npz"])
                
            logger.info(f"Command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
    def predict_test_set(self):
        logger.info("Running predictions on test set...")
        
        test_images_dir = os.path.join(
            self.dm.nnunet_raw_dir,
            self.dm.dataset_dir_name,
            "imagesTs"
        )
        test_labels_dir = os.path.join(
            self.dm.nnunet_results_dir,
            self.dm.dataset_dir_name,
            "labelsTs_predicted"
        )
        
        config = self.nnunet_config["configuration"]
        
        # Handle cascade configuration
        if config == "3d_cascade_fullres":
            # Predict with 3d_lowres first
            test_lowres_dir = os.path.join(
                self.dm.nnunet_results_dir,
                self.dm.dataset_dir_name,
                "labelsTs_lowres_predicted"
            )
            self._run_prediction("3d_lowres", test_images_dir, test_lowres_dir)
            
            # Then predict cascade using lowres predictions
            self._run_prediction(
                "3d_cascade_fullres",
                test_images_dir,
                test_labels_dir,
                prev_stage_predictions=test_lowres_dir
            )
        else:
            self._run_prediction(config, test_images_dir, test_labels_dir)
            
        logger.info("Predictions completed.")
        
    def _run_prediction(self, configuration, input_dir, output_dir, prev_stage_predictions=None):
        
        cmd = [
            "nnUNetv2_predict",
            "-d", self.dataset_id,
            "-i", input_dir,
            "-o", output_dir,
            "-tr", self.nnunet_config["trainer"],
            "-c", configuration,
            "-p", "nnUNetPlans",
            "-f"
        ] + [str(f) for f in self.nnunet_config["fold"]]
        
        if prev_stage_predictions:
            cmd.extend(["-prev_stage_predictions", prev_stage_predictions])
            
        logger.info(f"Command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
    def postprocess(self):
        """Apply nnUNet postprocessing if enabled"""
        if not self.nnunet_config["npz"]:
            logger.info("Skipping postprocessing (NPZ not enabled)")
            return
            
        logger.info("Running nnUNetv2_apply_postprocessing...")
        
        test_labels_dir = os.path.join(
            self.dm.nnunet_results_dir,
            self.dm.dataset_dir_name,
            "labelsTs_predicted"
        )
        test_labels_pp_dir = os.path.join(
            self.dm.nnunet_results_dir,
            self.dm.dataset_dir_name,
            "labelsTs_predicted_pp"
        )
        
        # Find postprocessing file
        postprocessing_file = self._locate_file(
            os.path.join(self.dm.nnunet_results_dir, self.dm.dataset_dir_name),
            "postprocessing.pkl"
        )
        
        if postprocessing_file:
            cmd = [
                "nnUNetv2_apply_postprocessing",
                "-i", test_labels_dir,
                "-o", test_labels_pp_dir,
                "-pp_pkl_file", postprocessing_file
            ]
            logger.info(f"Command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            logger.info("Postprocessing completed.")
        else:
            logger.warning("Postprocessing file not found.")
            
    def _locate_file(self, path, filename):
        
        for root, dirs, files in os.walk(path):
            for file in files:
                if filename in file:
                    return os.path.join(root, file)
        return None
        
    def log_artifacts(self):
        logger.info("Logging artefacts to MLflow...")
        
        # Save configuration
        config_path = os.path.join(
            self.dm.nnunet_results_dir,
            self.dm.dataset_dir_name,
            'config.txt'
        )
        self._save_config(config_path)
        
        # Log to MLflow
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                "model_type": "3d_segmentation",
                "framework": "nnUNetV2",
                "configuration": self.nnunet_config["configuration"],
                "trainer": self.nnunet_config["trainer"],
                "folds": str(self.nnunet_config["fold"]),
                "train_fraction": self.config["data"]["TRAIN_FRACTION"],
                "test_fraction": self.config["data"]["TEST_FRACTION"],
            })
            
            # Log artefacts
            mlflow.log_artifact(config_path)
            
            # Log nnUNet results
            results_dir = os.path.join(self.dm.nnunet_results_dir, self.dm.dataset_dir_name)
            if os.path.exists(results_dir):
                self._log_nnunet_artifacts(results_dir)
                
        logger.info("Artefact logging completed.")
        
    def _save_config(self, filepath):
        """Save configuration to file"""
        with open(filepath, 'w') as f:
            for section in self.config.sections():
                f.write(f"[{section}]\n")
                for key, value in self.config.items(section):
                    f.write(f"{key} = {value}\n")
                f.write("\n")
                
    def _log_nnunet_artifacts(self, path):
        """Log nnUNet artefacts to MLflow"""
        important_files = [
            'dataset.json', 'plans.json', 'postprocessing.json',
            'summary.json', 'progress.png', 'training_log'
        ]
        
        for root, dirs, files in os.walk(path):
            for file in files:
                if any(pattern in file for pattern in important_files):
                    file_path = os.path.join(root, file)
                    # Determine artifact path structure
                    if "fold_" in root:
                        for fold in range(5):
                            fold_str = f"fold_{fold}"
                            if fold_str in root:
                                if "validation" in root:
                                    artifact_path = f"{fold_str}/validation"
                                else:
                                    artifact_path = fold_str
                                mlflow.log_artifact(file_path, artifact_path)
                                break
                    else:
                        mlflow.log_artifact(file_path)
                        
    def run(self):
        """Execute complete training pipeline"""
        try:
            logger.info("=" * 60)
            logger.info("Starting 3D Segmentation Training Pipeline")
            logger.info("=" * 60)
            
            
            logger.info("\n[1/6] Setting up data...")
            self.setup_data()
            
            
            logger.info("\n[2/6] Preprocessing data...")
            self.preprocess_data()
            
            
            logger.info("\n[3/6] Training model...")
            self.train_model()
            
            
            logger.info("\n[4/6] Running predictions...")
            self.predict_test_set()
            
        
            logger.info("\n[5/6] Postprocessing...")
            self.postprocess()
            
            
            logger.info("\n[6/6] Logging artefacts...")
            self.log_artifacts()
            
            logger.info("\n" + "=" * 60)
            logger.info("3D Segmentation Training Pipeline Completed Successfully!")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise


def main():
    # Parse command line arguments
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        # Default config path relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(os.path.dirname(script_dir), 'config', 'local_config.cfg')
        
    # Check if config file exists
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
        
    logger.info(f"Loading configuration from: {config_path}")
    
    # Load configuration
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Create and run trainer
    trainer = SegmentationTrainer(config)
    trainer.run()


if __name__ == '__main__':
    main()