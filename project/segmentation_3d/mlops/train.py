import argparse
import configparser
import logging
import multiprocessing
import os
import tempfile

import mlflow

from src.datamodule import DataModule_nnUNetV2
import src.nnunet.commands as commands
import src.nnunet.runtime as nnunet_runtime
import src.nnunet.mlflow_logging as nnunet_mlflow

logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging for the run and return log_path, the path to the log file.

    log_path is the logging destination for the whole run:
    1) everything that appears in the console (via logger.*) is sent to the log file.
    2) so is nnU-Net subprocess output, which src.nnunet.runtime forwards to logger.*.

    The log file is ultimately sent to MLFlow.
    """

    log_path = os.path.join(tempfile.mkdtemp(), "run.log")
    logging.basicConfig( # match log format to csc-mlops
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
        force=True)

    return log_path


def log_run_to_mlflow(log_path):
    """Send run.log to MLflow, so the run is inspectable whether it succeeded or failed.

    Never raises: this is called from a finally block, where an error here would mask the
    exception that actually ended the run.
    """

    try:
        mlflow.log_artifact(log_path, artifact_path="logs")
    except Exception:
        logger.exception("Failed to log run log to MLflow.")


def setup_environment(config):
    """Sets up the following:
    1) GPU environment -> returns device
    2) number of workers for XNAT data download -> returns xnat_download_num_workers
    3) number of workers for nnU-Net data loading + augment -> sets nnUNet_n_proc_DA env var
    """

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
    
    logger.info(f"Env variable set: 'CUDA_VISIBLE_DEVICES' = '{os.environ['CUDA_VISIBLE_DEVICES']}'.")

    # nnunet data load and augment number of workers. If blank, defaults to nnU-Net default (12)
    n_proc_da = config["nnunet"].get("NNUNET_N_PROC_DA", "").strip()
    if n_proc_da:
        if not n_proc_da.isdigit() or int(n_proc_da) < 1:
            raise ValueError(
                f"Config error: nnunet.NNUNET_N_PROC_DA must be a positive integer or blank; "
                f"got {n_proc_da!r}."
            )
        os.environ["nnUNet_n_proc_DA"] = n_proc_da
        logger.info(f"Env variable set: 'nnUNet_n_proc_DA' = '{n_proc_da}'.")

    # xnat download number of workers
    max_workers = int(config["system"]["XNAT_DOWNLOAD_NUM_WORKERS"])
    xnat_download_num_workers = min(max_workers, multiprocessing.cpu_count())

    return xnat_download_num_workers, device


def setup_data(config, xnat_download_num_workers):
    """Build the nnU-Net dataset from XNAT and return the prepared DataModule."""

    xnat_configuration = {
        "server": config["xnat"]["SERVER"],
        "user": config["xnat"]["USER"],
        "password": config["xnat"]["PASSWORD"],
        "project": config["xnat"]["PROJECT"],
        "verify": config.getboolean("xnat", "VERIFY"),
    }

    tmp_dirs_configuration = {
        "tmp_working_dir": config["system"]["TMP_WORKING_DIR"],
        "nnunet_raw_dir": config["nnunet"]["NNUNET_RAW_DIR"],
        "nnunet_results_dir": config["nnunet"]["NNUNET_RESULTS_DIR"],
        "nnunet_preprocessed_dir": config["nnunet"]["NNUNET_PREPROCESSED_DIR"],
    }

    dm = DataModule_nnUNetV2(
        xnat_configuration=xnat_configuration,
        train_fraction=float(config["data"]["TRAIN_FRACTION"]),
        test_fraction=float(config["data"]["TEST_FRACTION"]),
        xnat_download_num_workers=xnat_download_num_workers,
        random_seed=int(config["system"]["RANDOM_SEED"]),
        tmp_dirs_configuration=tmp_dirs_configuration,
        regions_json_path=config["data"]["REGIONS_JSON_PATH"],
    )

    dm.setup()  # pull from XNAT + organise dataset locally as required by nnU-Net

    return dm


def train(config):
    """Full XNAT data loading -> nnU-Net v2 training -> MLflow logging workflow."""

    # add description of logged metrics in mlflow
    nnunet_mlflow.log_run_overview_description()

    xnat_download_num_workers, device = setup_environment(config)
    dm = setup_data(config, xnat_download_num_workers)

    # define/bundle parameters for nnunetv2 to run via CLI
    spec = commands.NNUNetModelSpec.from_config(config, dm)

    # base logging folder for MLFlow / nnU-Net
    artifact_dir = os.path.join(dm.nnunet_results_dir, spec.dataset_name)

    # set nnU-Net env variables
    os.environ["nnUNet_raw"] = dm.nnunet_raw_dir
    os.environ["nnUNet_preprocessed"] = dm.nnunet_preprocessed_dir
    os.environ["nnUNet_results"] = dm.nnunet_results_dir
    logger.info(f"Env variable set: 'nnUNet_raw' = '{dm.nnunet_raw_dir}'.")
    logger.info(f"Env variable set: 'nnUNet_preprocessed' = '{dm.nnunet_preprocessed_dir}'.")
    logger.info(f"Env variable set: 'nnUNet_results' = '{dm.nnunet_results_dir}'.")

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

        # Output final validation metrics as figures
        crossval_dir = commands.crossval_results_dir(dm.nnunet_results_dir, spec)
        nnunet_mlflow.log_metric_plots(
            summary_path=os.path.join(crossval_dir, "postprocessed", "summary.json"),
            dataset_json_path=os.path.join(crossval_dir, "dataset.json"),
            artifact_path="crossval_results",
        )
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

    # nnUNetv2 evaluate (test set): score the final test predictions against the test ground
    # truth (labelsTs) and output Test figures.
    try:
        test_gt_dir = os.path.join(dm.nnunet_raw_dir, spec.dataset_name, "labelsTs")
        test_pred_dir = test_labels_pp_dir if spec.use_npz else test_labels_dir
        
        if os.path.isdir(test_gt_dir) and os.listdir(test_gt_dir):
            dataset_json = os.path.join(dm.nnunet_raw_dir, spec.dataset_name, "dataset.json")
            plans_json = nnunet_runtime.locate_file(artifact_dir, "plans.json")
            cmd = commands.evaluate_folder_command(test_gt_dir, test_pred_dir, dataset_json, plans_json)
            logger.info(f"nnUNetv2_evaluate_folder: {cmd}")
            nnunet_runtime.run_command(cmd, artifact_dir=artifact_dir)
            nnunet_mlflow.log_metric_plots(
                summary_path=os.path.join(test_pred_dir, "summary.json"),
                dataset_json_path=dataset_json,
                artifact_path="test_set",
            )
        else:
            logger.warning("No test ground truth (labelsTs) found; skipping test-set evaluation.")
    except Exception:
        logger.exception("Test-set evaluation/plots failed; continuing.")

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
    nnunet_mlflow.log_nnunet_artifacts(artifact_dir)
    logger.info("Artifact storage complete.")

    # Log dummy model to enable MLflow registration
    logger.info("Creating and logging dummy model for registration")

    class DummyModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input):
            return "This is a dummy."

    mlflow.pyfunc.log_model(artifact_path="dummy_model", python_model=DummyModel())

    logger.info("nnU-Net v2 workflow complete.")


def main():

    log_path = setup_logging()

    # runs as train.py <config_file_path> via mlops run()
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    try:
        train(config)
    finally: # log the run to MLFlow whether success or fail
        log_run_to_mlflow(log_path)


if __name__ == '__main__':
    main()
