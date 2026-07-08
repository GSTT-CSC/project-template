import argparse
import configparser
import logging
import multiprocessing
import os

import mlflow

from src.datamodule import DataModule_nnUNetV2
import src.nnunet.commands as commands
import src.nnunet.runtime as nnunet_runtime

logger = logging.getLogger(__name__)


def setup_environment(config):
    """Set the GPU environment; return (num_workers, device)."""

    device = config["nnunet"]["NNUNET_DEVICE"].strip().lower()

    if device == "cuda":
        cuda_visible_devices = config["system"]["CUDA_VISIBLE_DEVICES"].strip()
        if not cuda_visible_devices:
            raise ValueError(
                "Config error: CUDA_VISIBLE_DEVICES must be set explicitly when NNUNET_DEVICE=cuda."
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    logger.info(f"CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']!r}")

    # Bridge the optional [nnunet] NNUNET_N_PROC_DA config value to nnU-Net's env var
    # (nnU-Net has no CLI flag for it). Blank -> leave unset -> nnU-Net default (~12).
    n_proc_da = config["nnunet"].get("NNUNET_N_PROC_DA", "").strip()
    if n_proc_da:
        if not n_proc_da.isdigit() or int(n_proc_da) < 1:
            raise ValueError(
                f"Config error: nnunet.NNUNET_N_PROC_DA must be a positive integer or blank; "
                f"got {n_proc_da!r}."
            )
        os.environ["nnUNet_n_proc_DA"] = n_proc_da
        logger.info(f"nnUNet_n_proc_DA = {n_proc_da}")

    max_workers = int(config["system"]["MAX_WORKERS"])
    num_workers = min(max_workers, multiprocessing.cpu_count())

    return num_workers, device


def setup_data(config, num_workers):
    """Build the nnU-Net dataset from XNAT and return the prepared DataModule."""

    xnat_configuration = {
        "server": config["xnat"]["SERVER"],
        "user": config["xnat"]["USER"],
        "password": config["xnat"]["PASSWORD"],
        "project": config["xnat"]["PROJECT"],
        "verify": config.getboolean("xnat", "VERIFY"),
    }

    tmp_dirs_configuration = {
        "os_tmp_dir": config["tmp_dirs"]["OS_TMP_DIR"],
        "tmp_working_dir": config["tmp_dirs"]["TMP_WORKING_DIR"],
        "nnunet_raw_dir": config["nnunet"]["NNUNET_RAW_DIR"],
        "nnunet_results_dir": config["nnunet"]["NNUNET_RESULTS_DIR"],
        "nnunet_preprocessed_dir": config["nnunet"]["NNUNET_PREPROCESSED_DIR"],
    }

    dm = DataModule_nnUNetV2(
        xnat_configuration=xnat_configuration,
        train_fraction=float(config["data"]["TRAIN_FRACTION"]),
        test_fraction=float(config["data"]["TEST_FRACTION"]),
        num_workers=num_workers,
        random_seed=int(config["system"]["RANDOM_SEED"]),
        tmp_dirs_configuration=tmp_dirs_configuration,
        regions_json_path=config["data"]["REGIONS_JSON_PATH"],
    )

    dm.setup()  # pull from XNAT + organise dataset locally as required by nnU-Net

    return dm


def train(config):
    """Full XNAT data loading -> nnU-Net v2 training -> MLflow logging workflow."""

    num_workers, device = setup_environment(config)
    dm = setup_data(config, num_workers)

    # define/bundle parameters for nnunetv2 to run via CLI
    spec = commands.build_nnunet_model_spec(config, dm)

    # base logging folder for MLFlow / nnU-Net
    artifact_dir = os.path.join(dm.nnunet_results_dir, spec.dataset_name)

    # set nnU-Net env variables
    os.environ["nnUNet_raw"] = dm.nnunet_raw_dir
    os.environ["nnUNet_preprocessed"] = dm.nnunet_preprocessed_dir
    os.environ["nnUNet_results"] = dm.nnunet_results_dir
    logger.info(f"nnUNet_raw = {dm.nnunet_raw_dir}")
    logger.info(f"nnUNet_preprocessed = {dm.nnunet_preprocessed_dir}")
    logger.info(f"nnUNet_results = {dm.nnunet_results_dir}")

    # nnU-Net v2 preprocessing
    cmd = commands.plan_and_preprocess_command(spec)
    logger.info(f"nnUNetv2_plan_and_preprocess: {cmd}")
    nnunet_runtime.run_command(cmd, artifact_dir=artifact_dir)

    # nnUNetv2 training (per fold; 3d_cascade_fullres config trains 3d_lowres first)
    for fold in spec.folds:
        for configuration in spec.training_configurations:
            cmd = commands.train_command(spec, fold, configuration, device)
            fold_dir = commands.fold_artifact_dir(dm.nnunet_results_dir, spec, fold, configuration)
            logger.info(f"nnUNetv2_train (fold {fold}, {configuration}): {cmd}")
            
            nnunet_runtime.run_command(
                cmd, artifact_dir=fold_dir, fold=fold,
                configuration=configuration, log_metrics=True,
            )

    # nnUNetv2 find best configuration
    if spec.use_npz:
        cmd = commands.find_best_configuration_command(spec)
        logger.info(f"nnUNetv2_find_best_configuration: {cmd}")
        nnunet_runtime.run_command(cmd, artifact_dir=artifact_dir)
    else:
        logger.info("Skipping nnUNetv2_find_best_configuration ...")

    # nnUNetv2 predict (test set). Predictions are written under nnUNet_results (not raw) so
    # log_nnunet_artifacts picks them up and logs them to MLflow.
    test_images_dir = os.path.join(dm.nnunet_raw_dir, spec.dataset_name, "imagesTs")
    test_labels_dir = os.path.join(dm.nnunet_results_dir, spec.dataset_name, "labelsTs_predicted")
    cmd = commands.predict_command(spec, test_images_dir, test_labels_dir, device)
    logger.info(f"nnUNetv2_predict: {cmd}")
    nnunet_runtime.run_command(cmd, artifact_dir=artifact_dir)

    # nnUNetv2 postprocessing
    if spec.use_npz:
        test_labels_pp_dir = os.path.join(
            dm.nnunet_results_dir, spec.dataset_name, "labelsTs_predicted_pp"
        )
        postprocessing_file = nnunet_runtime.locate_file(artifact_dir, "postprocessing.pkl")
        cmd = commands.apply_postprocessing_command(
            spec, test_labels_dir, test_labels_pp_dir, postprocessing_file
        )
        logger.info(f"nnUNetv2_apply_postprocessing: {cmd}")
        nnunet_runtime.run_command(cmd, artifact_dir=artifact_dir)
    else:
        logger.info("Skipping nnUNetv2_apply_postprocessing ...")

    # Log non-secret parts of config in MLFlow
    useful_keys = ['system', 'project', 'data', 'nnunet']
    with open(os.path.join(artifact_dir, 'config_log.txt'), 'w') as f:
        for section in useful_keys:
            for key, value in config.items(section):
                f.write(f'{key} = {value}\n')

    # Per-subject data manifest (the DataModule dataframe) for traceability
    dm.df.to_csv(os.path.join(artifact_dir, 'data_manifest.csv'), index=False)

    # MLflow metadata logging
    logger.info("Storing artifacts in MLflow ...")
    nnunet_runtime.log_nnunet_artifacts(artifact_dir)
    logger.info("Artifact storage complete.")

    # Log dummy model to enable MLflow registration
    logger.info("Creating and logging dummy model for registration")

    class DummyModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input):
            return "This is a dummy."

    mlflow.pyfunc.log_model(artifact_path="dummy_model", python_model=DummyModel())

    logger.info("nnU-Net v2 workflow complete.")


def main():
    # runs as train.py <config_file_path> via mlops run()
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    train(config)


if __name__ == '__main__':
    main()
