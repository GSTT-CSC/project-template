import sys
import configparser
import logging
import os
import multiprocessing
import json

import mlflow
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from ray.air.integrations.mlflow import setup_mlflow
from torch.cuda import is_available as cuda_available

from src.DataModule import DataModule
from src.Network import Network
from src.DataModule import label_dict
from src.XNATDataImport import XNATDataImport

logger = logging.getLogger(__name__)

def train(config):

    os.environ["CUDA_VISIBLE_DEVICES"] = config["system"]["cuda_visible_devices"]

    xnat_configuration = {'server': config['xnat']['SERVER'],
                          'user': config['xnat']['USER'],
                          'password': config['xnat']['PASSWORD'],
                          'project': config['xnat']['PROJECT'],
                          'verify': config.getboolean('xnat', 'VERIFY')}

    max_workers = 32
    num_workers = (
        max_workers
        if max_workers < multiprocessing.cpu_count()
        else multiprocessing.cpu_count()
    )

    importer = XNATDataImport(
        xnat_configuration = xnat_configuration,
        num_workers = num_workers
        )

    # Import raw data
    raw_data = importer.import_xnat_data()

    # Download images from XNAT
    data = importer.xnat_image_download(raw_data)
 
    # Set up mflow experiment
    setup_mlflow(
        tracking_uri=mlflow.get_tracking_uri(),
        experiment_id=mlflow.get_experiment_by_name(
            config["project"]["name"]
        ).experiment_id
        if mlflow.get_experiment_by_name(config["project"]["name"])
        else mlflow.create_experiment(config["project"]["name"]),
    )
    with mlflow.start_run(nested=True):
        save_best_model = True

        mlflow.pytorch.autolog(log_models=False)

        # initialise network and datamodule
        dm = DataModule(
            data = data,
            dm_batch_size = int(config['params']['dm_batch_size']),
            test_fraction = float(config['params']['test_fraction']),
            num_workers = num_workers,
            random_seed = int(config['params']['random_seed']),
            image_size = int(config['params']['image_size'])
        )

        dm.setup()
        
        n_classes = len(set([x for x in label_dict.values() if x is not None]))
        mlflow.log_param('n_classes', n_classes)

        train_class_weights = dm.data_manifest["train"]["class_weights"]
        validation_class_weights = dm.data_manifest["validation"]["class_weights"]

        net = Network(
            model_name = config['params']['model'],
            pretrained = config['params']['pretrained'],
            n_classes = n_classes,
            dropout = float(config['params']['dropout']),
            weighted_loss = config['params']['weighted_loss'],
            train_class_weights = train_class_weights,
            validation_class_weights = validation_class_weights,
            learning_rate = float(config['params']['lr']),
            max_lr = float(config['params']['max_lr']),
            nw_batch_size = int(config['params']['nw_batch_size']),
            label_smoothing = float(config['params']['label_smoothing']),
        )

        # Callbacks
        callbacks = []
        callbacks.append(LearningRateMonitor(logging_interval="step"))
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            dirpath="./checkpoint/",
        )
        callbacks.append(checkpoint_callback)

        # configure trainer
        trainer = pl.Trainer(
            precision="32" if cuda_available() else "16",
            callbacks=callbacks,
            devices= 1 if cuda_available() else "auto",
            accelerator="gpu" if cuda_available() else "auto",
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            max_epochs=int(config['params']['max_epochs']),
            accumulate_grad_batches= int(config['params']['grad_batches'])
        )

        trainer.fit(net, dm)

        if save_best_model:
            checkpoint = torch.load(checkpoint_callback.best_model_path)
            net.load_state_dict(checkpoint['state_dict'])
            file_path = f"model-{config['project']['name']}-{mlflow.active_run().info.run_name}.pt"
            script = net.to_torchscript(file_path=file_path)
            
            checkpoint_info = {}
            checkpoint_info["monitored_metric"] = checkpoint_callback.monitor
            checkpoint_info["metric_value"] = checkpoint_callback.best_model_score.item()
            checkpoint_info["mode"] = checkpoint_callback.mode
            checkpoint_info["epoch"] = checkpoint["epoch"]

            with open("checkpoint_info.json", "w") as f:
                json.dump(checkpoint_info, f, indent=2)

            mlflow.pytorch.log_model(
                script, file_path, extra_files=["checkpoint_info.json"]
            )

            _ = [
                os.remove(fp) for fp in ["checkpoint_info.json"]
            ]  # remove the temporary files after logging to mlflow

        # Prepare config for mlflow logging
        useful_keys = ['system',
                    'project',
                    'params',
                ]

        with open(('config_log.txt'), 'w') as f:
            for section in useful_keys:
                for key, value in config.items(section):
                    f.write(f'{key} = {value}\n')    
        
        mlflow.log_artifact('config_log.txt')
        
        logger.info('Finding best threshold')
        best_model_path = checkpoint_callback.best_model_path
        best_model = net.load_from_checkpoint(best_model_path)
        net.evaluate_best_model(best_model, threshold_tune=True)

        mlflow.log_artifact('val_transform_failures.csv')
        mlflow.log_artifact('train_transform_failures.csv')

        logger.info('Training complete')
        

if __name__ == '__main__':

    config_path = 'config/config.cfg'

    config = configparser.ConfigParser()
    config.read(config_path)
    train(config)