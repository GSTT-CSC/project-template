"""
List of nnU-Net v2 system commands and config translation functions.

Works in single configuration only.

nnU-Net's own workflow typically trains several variants (that they call configurations: 2d, 3d_fullres,
3d_lowres). It then compares them on the cross-validation, and can ensemble the best two.

This template deliberately does not allow multiple configurations. It trains the one configuration set
by NNUNET_UNET_CONFIGURATION and passes --disable_ensembling. To compare configurations, run the whole
pipeline once per configuration and compare the resulting MLflow runs by hand.

nnU-Net's way introduces a lot of complexity and compute time so we prefer to leave this bit manual.

nnU-Net remains self-configuring within that configuration: patch size, batch size and network
architecture are still derived from the data and the GPU memory budget. What this template takes over
manually is only the choice between configurations.
"""

import inspect
import json
import os
from dataclasses import dataclass
import nnunetv2
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from typing import List, Optional


# ======================================================================================
# Validated config objects
# ======================================================================================

def plans_identifier_for_planner(planner_name):

    """Ask nnU-Net which plans file the named planner writes.

    How nnU-Net handles this planner vs plan, as it's really not obvious:

    1. ``nnUNetv2_plan_and_preprocess -pl <planner>`` (NNUNET_PLANNER from config) runs that planner.
       Each planner writes a plan file with a hardcoded name, defined as the ``plans_name`` argument
       to its constructor. For the purpose of this template, we call this the plan identifier, 
       and its variable name is <plans_identifier>.

    2. The planner writes <nnUNet_preprocessed>/<dataset_name>/<plans_identifier>.json, which contains
       the patch size, batch size, spacing, etc. for that dataset. This is what nnU-Net uses to 
       configure training and inference.

    3. The variable <plans_identifier> must be handed back as ``-p`` to nnUNetv2_train, 
       _find_best_configuration and _predict. Each process basically recreates the path of the
       plan (as in step 2, <nnUNet_preprocessed>/<dataset_name>/<plans_identifier>.json) and
       loads it. 

    What this function does is read the planner's name, find the associated nnunet class, and then
    read its hardcoded ``plans_name`` attribute (which is directly <plans_identifier>) and returns
    this value.

    This sounds like a lot of work for nothing but nnU-Net works this way as it decouples the 
    planning and preprocessing step (can generate multiple plans from multiple datasets) to the
    training and inference parts.
    """

    planner_class = recursive_find_python_class(
        os.path.join(nnunetv2.__path__[0], "experiment_planning"),
        planner_name,
        current_module="nnunetv2.experiment_planning",
    )
    if planner_class is None:
        raise ValueError(f"Config error: nnunet.NNUNET_PLANNER '{planner_name}' is not a nnU-Net v2 planner class")

    # read the plan name from class constructor
    plans_name = inspect.signature(planner_class.__init__).parameters.get("plans_name")

    return plans_name.default


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

        # NNUNET_NPZ: softmax .npz export. Only needed to ensemble several configurations, which
        # this template does not do - but it currently also gates find_best_configuration and
        # apply_postprocessing in train.py, so leaving it True keeps those steps running.
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
    def plans_identifier(self):
        """Name of the plans file the chosen planner writes; passed back to nnU-Net as -p.
        See plans_identifier_for_planner for more info on planner vs plans file.
        """

        return plans_identifier_for_planner(self.planner)


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


def fold_artifact_dir(results_dir, spec, fold):
    """Directory nnU-Net writes a fold's checkpoints/logs to (used for live metric logging)."""

    return os.path.join(
        results_dir,
        spec.dataset_name,
        f"{spec.trainer_name}__{spec.plans_identifier}__{spec.configuration}",
        f"fold_{fold}",
    )


def crossval_results_dir(results_dir, spec):
    """Directory find_best_configuration writes the accumulated cross-validation results to.

    Named crossval_results_folds_<folds joined by _>, e.g. crossval_results_folds_0 when
    NNUNET_FOLD = [0].
    """

    folds_str = "_".join(str(fold) for fold in spec.folds)
    return os.path.join(
        results_dir,
        spec.dataset_name,
        f"{spec.trainer_name}__{spec.plans_identifier}__{spec.configuration}",
        f"crossval_results_folds_{folds_str}",
    )


def plan_and_preprocess_command(spec):
    """ Run CLI command as defined -> https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/experiment_planning/plan_and_preprocess_entrypoints.py
    
    This command
    1) measures the dataset (image sizes, spacings, intensity statistics),
    2) has NNUNET_PLANNER turn those measurements into a patch size, batch size and architecture,
    3) crops / normalises / resamples every case and write it out ready for training.
    """
    
    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", spec.dataset_id,
        "-pl", spec.planner,
        "-c", spec.configuration,
        "--verify_dataset_integrity",
    ]
    if spec.gpu_memory_target is not None:
        cmd += ["-gpu_memory_target", str(spec.gpu_memory_target)]
    return cmd


def train_command(spec, fold, device):
    """
    Run CLI command -> https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/run/run_training.py.

    1) on first call, create folds file (splits_final.json) which contains what data goes in what fold
    2) reads plans file for patch/batch size and architecture, and the trainer for epochs, DA, loss
    3) train one fold at a time as defined by the `fold` input.

    Checkpoints go to nnUNet_results/<dataset>/<trainer>__<plans>__<configuration>/fold_<fold>/
    (i.e. what fold_artifact_dir builds) and are: checkpoint_latest, checkpoint_best (best EMA
    dice), checkpoint_final (end of run)

    Once epochs are done, the final-epoch weights (checkpoint_final) predict the fold's validation
    cases - nnU-Net only uses checkpoint_best here if --val_best is passed, which it is not.
    These are written to <fold folder>/validation/ as .nii.gz segmentations, plus one .npz of
    softmax probabilities per case if --npz was set (only used for ensembling - see
    find_best_configuration_command).

    Training variants (spec.trainer_name) loosely defined in
    https://github.com/MIC-DKFZ/nnUNet/tree/master/nnunetv2/training/nnUNetTrainer/variants.
    """
    
    cmd = [
        "nnUNetv2_train",
        spec.dataset_id,
        spec.configuration,
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

    Copies the .nii.gz validation predictions out of each requested fold_N/validation/ into a single
    crossval_results_folds_<folds>/ directory and scores them against the ground truth
    (summary.json). On that basis it (in theory) makes two decisions:

    1) (NOT DONE IN THIS REPOSITORY) Which configuration is best by comparing the mean foreground
       Dice of each trained configuration, and of ensembles of two if allowed.
       Since this template trains one configuration and passes --disable_ensembling, there is always
       only one configuration that wins by default. The .npz softmax files are only read to
       build those ensembles, so here they are written during training and never used.

    2) Which "postprocessing method" is best - on the winner configuration, it tries keeping only the
       largest connected blob of voxels and deleting the rest, first across all foreground then
       per class. Models tend to emit stray voxels away from the real structure and dropping them
       can increase Dice, but this is not always true if the structure is multi-focal; each cleanup
       is kept only if it measurably improves Dice.
       The best "postprocessing method" is saved as postprocessing.pkl, which apply_postprocessing_command
       later replays on the test predictions.

    Overall the output of this command is the crossval_results_folds_<folds> directory with the following inside:
    - (a) fold's validation predictions (actual validation predictions before postprocessing)
    - (b) summary.json (summary of metrics before postprocessing)
    - (c) postprocessing.pkl (the chosen "postprocessing method") from point 2
    - (d) postprocessing.json (before/after Dice scores for the above decided postprocessing method)
    - (e) /postprocessed including the same as (a) and (b) (predictions and summary) but after postprocessing

    Plus, one level up in nnUNet_results/<dataset_name>/:
    - (f) inference_information.json, inference_instructions.txt (the winning model, its pre/post
          postprocessing Dice, paths to postprocessing.pkl and plans.json, and ready-made commands)
    """

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

    Segments every image in input_dir with the trained model, writing .nii.gz masks to output_dir.

    Does no post-processing (nnU-Net keeps that separate, see apply_postprocessing_command).

    -f lists which fold(s) to predict with: nnU-Net loads one model per fold and averages their
    predictions, so NNUNET_FOLD = [0] predicts with a single model while [0, 1, 2, 3, 4] ensembles
    five. Note this is ensembling across folds, unrelated to ensembling configurations.

    Predicts with checkpoint_final (the -chk default), which is also what the training run used for
    its own validation (--val_best is off by default), so the two sets of metrics are comparable.
    """

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

    Applies the postprocessing that find_best_configuration decided on (postprocessing.pkl) to a
    folder of predictions, writing the cleaned segmentations to output_dir.
    """

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
