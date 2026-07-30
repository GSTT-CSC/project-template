"""
List of nnU-Net v2 system commands and config translation functions.
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional


# ======================================================================================
# Validated config objects
# ======================================================================================
@dataclass(frozen=True)
class NNUNetModelSpec:
    """Specs for the nnU-Net model/training run.
    A frozen dataclass is used so that nothing can modify the run's parameters.

    For now, always build via ``from_config``, which is where the values are parsed and validated.
    """
    dataset_id: str
    dataset_name: str
    folds: List[int]
    trainer_name: str
    plans_identifier: str
    configuration: str
    planner: str
    gpu_memory_target: Optional[int]
    use_npz: bool

    @classmethod
    def from_config(cls, config, dm):
        """Build the spec from the config file + DataModule (parses + validates)"""

        # dataset_id / dataset_name come from the DataModule, not the config file.
        if not str(dm.dataset_id).isdigit():
            raise ValueError(f"DataModule dataset_id must be a numeric string, got: {dm.dataset_id}")

        dataset_name = dm.dataset_dir_name.strip()
        if not dataset_name:
            raise ValueError("DataModule dataset_dir_name must not be blank.")

        # Plain string settings; blank is the only way these can be wrong here.
        plain_settings = {}
        for field_name, config_key in (
            ("trainer_name", "NNUNET_TRAINER"),
            ("plans_identifier", "NNUNET_PLANS_IDENTIFIER"),
            ("planner", "NNUNET_PLANNER"),
        ):
            plain_settings[field_name] = config["nnunet"][config_key].strip()
            if not plain_settings[field_name]:
                raise ValueError(f"Config error: nnunet.{config_key} must not be blank.")

        # NNUNET_UNET_CONFIGURATION: one configuration only. Several would mean ensembling,
        # which needs nnUNetv2_ensemble and dropping --disable_ensembling. This is not supported
        # in this template.
        configuration = config["nnunet"]["NNUNET_UNET_CONFIGURATION"].strip()
        if configuration == "3d_cascade_fullres":
            raise ValueError(
                "Config error: nnunet.NNUNET_UNET_CONFIGURATION '3d_cascade_fullres' is not "
                "supported yet: cascade test inference requires a two-stage predict that this "
                "template does not implement. Use 3d_fullres, 3d_lowres or 2d."
            )
        if configuration not in ("2d", "3d_fullres", "3d_lowres"):
            raise ValueError(
                f"Config error: nnunet.NNUNET_UNET_CONFIGURATION must be 2d, 3d_fullres or "
                f"3d_lowres; got: {configuration}."
            )

        # NNUNET_FOLD, e.g. "[0]" -> [0].
        raw_folds = config["nnunet"]["NNUNET_FOLD"].strip()
        try: # easiest way to parse a list of ints from a string is to use json.loads
            folds = json.loads(raw_folds)
        except json.JSONDecodeError:
            folds = None
        if not isinstance(folds, list) or not folds or not all(type(f) is int for f in folds):
            raise ValueError(
                f"Config error: nnunet.NNUNET_FOLD must be a non-empty list of integers, "
                f"e.g. [0] or [0, 1, 2, 3, 4]; got: {raw_folds}."
            )

        # NNUNET_GPU_MEMORY_TARGET in GB; blank -> None -> nnU-Net's own default.
        raw_gpu_memory_target = config["nnunet"]["NNUNET_GPU_MEMORY_TARGET"].strip()
        if not raw_gpu_memory_target:
            gpu_memory_target = None
        elif raw_gpu_memory_target.isdigit() and int(raw_gpu_memory_target) > 0:
            gpu_memory_target = int(raw_gpu_memory_target)
        else:
            raise ValueError(
                f"Config error: nnunet.NNUNET_GPU_MEMORY_TARGET must be a positive integer (GB) "
                f"or blank, got: {raw_gpu_memory_target}."
            )

        # NNUNET_NPZ: softmax .npz export, needed for find_best_configuration / postprocessing.
        raw_npz = config["nnunet"]["NNUNET_NPZ"].strip().lower()
        if raw_npz not in ("true", "false", "--npz"):
            raise ValueError(f"Config error: nnunet.NNUNET_NPZ must be True or False, got: {raw_npz}.")

        return cls(
            dataset_id=dm.dataset_id,
            dataset_name=dataset_name,
            folds=folds,
            configuration=configuration,
            gpu_memory_target=gpu_memory_target,
            use_npz=raw_npz in ("true", "--npz"),
            **plain_settings,
        )

    @property
    def preprocess_configurations(self):
        """
        Configurations to preprocess (cascade will use fullres and lowres).
        """

        if self.configuration == "3d_cascade_fullres":
            return ["3d_fullres", "3d_lowres"]
        return [self.configuration]

    @property
    def training_configurations(self):
        """
        Configurations to train, in order (cascade will use lowres then cascade).
        """

        if self.configuration == "3d_cascade_fullres":
            return ["3d_lowres", "3d_cascade_fullres"]
        return [self.configuration]


# ======================================================================================
# Pure nnU-Net CLI command builders
# ======================================================================================

def multi_gpu_args(device):
    """
    Return ['-num_gpus', N] when more than one CUDA device is visible, else [].
    """

    if device != "cuda":
        return []
    n_visible = len(
        [gpu for gpu in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if gpu.strip()]
    )
    return ["-num_gpus", str(n_visible)] if n_visible > 1 else []


def fold_artifact_dir(results_dir, spec, fold, configuration):
    """Directory nnU-Net writes a fold's checkpoints/logs to (used for live metric logging)."""

    return os.path.join(
        results_dir,
        spec.dataset_name,
        f"{spec.trainer_name}__{spec.plans_identifier}__{configuration}",
        f"fold_{fold}",
    )


def crossval_results_dir(results_dir, spec):
    """Grab dir that find_best_configuration writes the cross-validation results to.

    Current understanding looks like ``crossval_results_folds_0_1_2_3_4``.
    """

    folds_str = "_".join(str(fold) for fold in spec.folds)
    return os.path.join(
        results_dir,
        spec.dataset_name,
        f"{spec.trainer_name}__{spec.plans_identifier}__{spec.configuration}",
        f"crossval_results_folds_{folds_str}",
    )


def plan_and_preprocess_command(spec):
    """ Run CLI command as defined -> https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/experiment_planning/plan_and_preprocess_entrypoints.py"""
    
    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", spec.dataset_id,
        "-pl", spec.planner,
        "-c", *spec.preprocess_configurations,
        "--verify_dataset_integrity",
    ]
    if spec.gpu_memory_target is not None:
        cmd += ["-gpu_memory_target", str(spec.gpu_memory_target)]
    return cmd


def train_command(spec, fold, configuration, device):
    """
    Run CLI command -> https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/run/run_training.py.
    
    Training variants (spec.trainer_name) loosely defined in
    https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunetv2/training/nnUNetTrainer/variants.
    """
    
    cmd = [
        "nnUNetv2_train",
        spec.dataset_id,
        configuration,
        str(fold),
        "-tr", spec.trainer_name,
        "-p", spec.plans_identifier,
        "-device", device,
    ]
    if spec.use_npz:
        cmd += ["--npz"]
    return cmd + multi_gpu_args(device)


def find_best_configuration_command(spec):
    """
    Run CLI command -> https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/evaluation/find_best_configuration.py
    """

    # --disable_ensembling below is adequate on a single config. Extra guard here
    # such that if multiple configs are supported in the future, --disable_ensembling
    # can be removed. 
    if not isinstance(spec.configuration, str):
        raise ValueError("remove --disable_ensembling for multiple configurations.")
    return [
        "nnUNetv2_find_best_configuration",
        spec.dataset_id,
        "-p", spec.plans_identifier,
        "-c", spec.configuration,
        "-tr", spec.trainer_name,
        "--disable_ensembling",
        "-f", *[str(fold) for fold in spec.folds],
    ]


def predict_command(spec, input_dir, output_dir, device):
    """
    Runs CLI command -> https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/inference/predict_from_raw_data.py
    """

    # -c uses the single trained configuration. Extra guard here such that if multiple
    # configs are supported in the future and the optimal inference configuration is
    # determined by find_best_configuration, the final configuration should be read from
    # find_best_configuration's inference_instructions.txt instead of the training list.
    if not isinstance(spec.configuration, str):
        raise ValueError("read -c from inference_instructions.txt for multiple configurations.")

    return [
        "nnUNetv2_predict",
        "-d", spec.dataset_id,
        "-i", input_dir,
        "-o", output_dir,
        "-c", spec.configuration,
        "-tr", spec.trainer_name,
        "-p", spec.plans_identifier,
        "-device", device,
        "-f", *[str(fold) for fold in spec.folds],
    ]


def apply_postprocessing_command(spec, input_dir, output_dir, postprocessing_file):
    """
    Run CLI command -> https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/postprocessing/remove_connected_components.py
    """

    # if multiple configs are supported in the future, -plans_json and -dataset_json must be
    # passed explicitly (from an ensemble member).
    if not isinstance(spec.configuration, str):
        raise ValueError("pass -plans_json / -dataset_json for multiple configurations.")
    return [
        "nnUNetv2_apply_postprocessing",
        "-i", input_dir,
        "-o", output_dir,
        "-pp_pkl_file", postprocessing_file,
    ]


def evaluate_folder_command(gt_dir, pred_dir, dataset_json, plans_json):
    """
    Run CLI command -> https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/evaluation/evaluate_predictions.py

    Scores predicted segmentations against ground truth, writing summary.json into pred_dir.
    --chill: don't crash if a ground-truth case has no matching prediction.
    """

    return [
        "nnUNetv2_evaluate_folder",
        gt_dir,
        pred_dir,
        "-djfile", dataset_json,
        "-pfile", plans_json,
        "--chill",
    ]
