import argparse
import configparser
import json
import logging
import multiprocessing
import os

import mlflow
import optuna
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from src.data_import_xnat import DataImportXNAT
from src.datamodule import DataModule
from src.network import Network
from src.utils.tools import format_tune

logger = logging.getLogger(__name__)


def setup_environment(config):
    """Set CUDA device and global random seed.
    Returns num_workers (int), capped at CPU count."""

    os.environ["CUDA_VISIBLE_DEVICES"] = config["system"]["cuda_visible_devices"]
    pl.seed_everything(int(config['system']['random_seed']), workers=True)

    max_workers = int(config['system']['max_workers'])
    return min(max_workers, multiprocessing.cpu_count())


def setup_data(config, num_workers):
    """ Raw data import function - connects to XNAT, 
    downloads images and returns raw dataset."""
    
    xnat_configuration = {
        'server': config['xnat']['server'],
        'user': config['xnat']['user'],
        'password': config['xnat']['password'],
        'project': config['xnat']['project'],
        'verify': config.getboolean('xnat', 'verify'),
    }

    importer = DataImportXNAT(xnat_configuration=xnat_configuration, num_workers=num_workers)
    raw_data = importer.import_xnat_data()
    return importer.xnat_image_download(raw_data)


def suggest_hyperparameters(trial, config):
    """ Updates selected hyperparameters in config by sampling one value 
    per config[tune] key using Optuna. Returns sampled values for logging."""

    tune = format_tune(config['tune'])
    suggested = {}

    for key in set(tune.keys()) - {'n_trials'}:
        if key not in config['params']:
            raise ValueError(
                f"[tune] key '{key}' not found in [params]. "
                f"Add a default value there before tuning it."
            )
        val = trial.suggest_categorical(key, tune[key])
        suggested[key] = val
        config['params'][key] = str(val)

    return suggested


def build_experiment(data, num_workers, config, checkpoint_dir):
    """Build DataModule, Network, and Trainer from config."""

    label_dict = json.loads(config['project']['label_dict'])

    dm = DataModule(
        data=data,
        label_dict=label_dict,
        batch_size=int(config['params']['batch_size']),
        validation_fraction=float(config['params']['validation_fraction']),
        test_fraction=float(config['params']['test_fraction']),
        num_workers=num_workers,
        random_seed=int(config['system']['random_seed']),
        image_size=int(config['params']['image_size']),
    )
    
    dm.setup()

    n_classes = len({v for v in label_dict.values() if v is not None})

    train_class_weights = dm.data_manifest["train"]["class_weights"]
    validation_class_weights = dm.data_manifest["validation"]["class_weights"]
    test_class_weights = dm.data_manifest["test"]["class_weights"]

    net = Network(
        n_classes=n_classes,
        label_dict=label_dict,
        model_name=config['params']['model'],
        pretrained=config.getboolean('params', 'pretrained'),
        learning_rate=float(config['params']['lr']),
        max_lr=float(config['params']['max_lr']),
        batch_size=int(config['params']['batch_size']),
        dropout=float(config['params']['dropout']),
        train_class_weights=train_class_weights,
        validation_class_weights=validation_class_weights,
        test_class_weights=test_class_weights,
        weighted_loss=config.getboolean('params', 'weighted_loss'),
        loss_fcn=config['params']['loss_fcn'],
        weight_decay=float(config['params']['weight_decay']),
        mixup_alpha=float(config['params']['mixup_alpha']),
        cutmix_alpha=float(config['params']['cutmix_alpha']),
        mixup_prob=float(config['params']['mixup_prob']),
        mixup_switch_prob=float(config['params']['mixup_switch_prob']),
        mixup_mode=config['params']['mixup_mode'],
        label_smoothing=float(config['params']['label_smoothing']),
    )

    checkpoint_metric = config['params']['checkpoint_metric']
    checkpoint_mode = config['params']['checkpoint_mode']

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor=checkpoint_metric,
        mode=checkpoint_mode,
        dirpath=checkpoint_dir,
    )
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        checkpoint_callback,
        EarlyStopping(
            monitor=checkpoint_metric,
            patience=int(config['params']['patience']),
            mode=checkpoint_mode,
            verbose=True,
        ),
    ]

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        precision_setting = "bf16-mixed"
    elif torch.cuda.is_available():
        precision_setting = "16-mixed"
    else:
        precision_setting = "32"    

    trainer = pl.Trainer(
        precision=precision_setting,
        callbacks=callbacks,
        devices=1 if torch.cuda.is_available() else "auto",
        accelerator="auto",
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        max_epochs=int(config['params']['max_epochs']),
        accumulate_grad_batches=int(config['params']['grad_batches']),
    )

    return dm, net, trainer, checkpoint_callback, n_classes


def objective(trial, data, num_workers, config):
    """Optuna trial objective; returns best model score for optimisation. 
    Opens one nested mlflow run per trial. """
    
    with mlflow.start_run(nested=True):

        # Update config with optuna suggested hps, and return updated params for logging
        suggested = suggest_hyperparameters(trial, config)
        mlflow.log_params(suggested)

        dm, net, trainer, checkpoint_callback, n_classes = build_experiment(
            data, num_workers, config, f'./checkpoint_trial_{trial.number}/'
        )
        mlflow.log_param('n_classes', n_classes)

        trainer.fit(net, dm)

        best_model = net.load_from_checkpoint(checkpoint_callback.best_model_path)
        net.evaluate_best_model(best_model)

        return checkpoint_callback.best_model_score.item()


def run_training(data, num_workers, config):
    """Training pipeline: fits, tests, exports the model as TorchScript, 
    and logs all artifacts."""

    with mlflow.start_run(nested=True):

        mlflow.pytorch.autolog(log_models=False)

        dm, net, trainer, checkpoint_callback, n_classes = build_experiment(
            data, num_workers, config, './checkpoint/'
        )
        mlflow.log_param('n_classes', n_classes)

        trainer.fit(net, dm)

        # run test set evaluation with best model only if non-zero test fraction
        if float(config['params']['test_fraction']) > 0:
            trainer.test(net, dm, ckpt_path=checkpoint_callback.best_model_path)

        # load best model checkpoint and log to mlflow
        checkpoint = torch.load(checkpoint_callback.best_model_path)
        net.load_state_dict(checkpoint['state_dict'])
        file_path = f"model-{config['project']['name']}-{mlflow.active_run().info.run_name}.pt"
        script = net.to_torchscript(file_path=file_path)

        checkpoint_info = {
            "monitored_metric": checkpoint_callback.monitor,
            "metric_value": checkpoint_callback.best_model_score.item(),
            "mode": checkpoint_callback.mode,
            "epoch": checkpoint["epoch"],
        }

        with open("checkpoint_info.json", "w") as f:
            json.dump(checkpoint_info, f, indent=2)

        mlflow.pytorch.log_model(script, file_path, extra_files=["checkpoint_info.json"])
        os.remove("checkpoint_info.json") # remove temp files after logging to mlflow

        useful_keys = ['project', 'system', 'params']
        with open('config_log.txt', 'w') as f:
            for section in useful_keys:
                for key, value in config.items(section):
                    f.write(f'{key} = {value}\n')
        mlflow.log_artifact('config_log.txt')

        logger.info('Finding best threshold')
        best_model = net.load_from_checkpoint(checkpoint_callback.best_model_path)
        net.evaluate_best_model(best_model, split='val')

        mlflow.log_artifact('val_transform_failures.csv')
        mlflow.log_artifact('train_transform_failures.csv')

        if float(config['params']['test_fraction']) > 0:
            net.evaluate_best_model(best_model, split='test')
            mlflow.log_artifact('test_transform_failures.csv')

        logger.info('Training complete')


def run_tuning(data, num_workers, config):
    """Tuning pipeline with Optuna: for each trial, run training cycle 
    with sampled hyperparmeters as defined in config[tune]. Logs a
    summary of best trial as an artifact. """

    mlflow.pytorch.autolog(log_models=False)

    direction = 'minimize' if config['params']['checkpoint_mode'] == 'min' else 'maximize'
    study = optuna.create_study(study_name="project-tune", direction=direction)
    study.optimize(
        lambda trial: objective(trial, data, num_workers, config),
        n_trials=int(config['tune']['n_trials']),
    )

    trial = study.best_trial

    with open('tune_log.txt', 'w') as f:
        
        f.write("Study statistics:\n")
        f.write(f"  Number of finished trials: {len(study.trials)}\n")
        f.write("Best trial:\n")
        f.write(f"  Trial number: {trial.number}\n")
        f.write(f"  Loss (trial value): {trial.value}\n")
        f.write("Params:\n")
        for key, value in trial.params.items():
            f.write(f"    {key} = {value}\n")

    mlflow.log_artifact('tune_log.txt')
    mlflow.log_artifact('val_transform_failures.csv')
    mlflow.log_artifact('train_transform_failures.csv')


def main():
    
    # runs as train_tune.py <config_file_path> via mlops run()
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    num_workers = setup_environment(config)
    data = setup_data(config, num_workers)

    mode = config['project']['mode']
    if mode == 'train':
        run_training(data, num_workers, config)
    elif mode == 'tune':
        run_tuning(data, num_workers, config)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Options are: train, tune")


if __name__ == '__main__':
    main()
