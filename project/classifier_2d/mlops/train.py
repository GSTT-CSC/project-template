import configparser
import json
import logging
import multiprocessing
import os

import mlflow
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from torch.cuda import is_available as cuda_available

from src.data_import_xnat import DataImportXNAT
from src.datamodule import DataModule
from src.network import Network

logger = logging.getLogger(__name__)

def train(config):

    os.environ["CUDA_VISIBLE_DEVICES"] = config["system"]["cuda_visible_devices"]
    pl.seed_everything(int(config['system']['random_seed']), workers=True)

    xnat_configuration = {'server': config['xnat']['server'],
                          'user': config['xnat']['user'],
                          'password': config['xnat']['password'],
                          'project': config['xnat']['project'],
                          'verify': config.getboolean('xnat', 'verify')}

    max_workers = config['system']['max_workers']
    num_workers = (
        max_workers
        if max_workers < multiprocessing.cpu_count()
        else multiprocessing.cpu_count()
    )

    importer = DataImportXNAT(
        xnat_configuration = xnat_configuration,
        num_workers = num_workers
        )

    # Import raw data
    raw_data = importer.import_xnat_data()

    # Download images from XNAT
    data = importer.xnat_image_download(raw_data)
 
    label_dict = json.loads(config['project']['label_dict'])

    # Set up mflow experiment
    with mlflow.start_run(nested=True):

        mlflow.pytorch.autolog(log_models=False)

        # initialise network and datamodule
        dm = DataModule(
            data = data,
            label_dict = label_dict,
            batch_size = int(config['params']['batch_size']),
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
            n_classes = n_classes,
            label_dict = label_dict,
            model_name = config['params']['model'],
            pretrained = config['params']['pretrained'],
            learning_rate = float(config['params']['lr']),
            max_lr = float(config['params']['max_lr']),
            batch_size = int(config['params']['batch_size']),
            dropout = float(config['params']['dropout']),
            train_class_weights = train_class_weights,
            validation_class_weights = validation_class_weights,
            weighted_loss = config['params']['weighted_loss'],
            loss_fcn = config['params']['loss_fcn'],
            weight_decay = float(config['params']['weight_decay']),
            mixup_alpha = float(config['params']['mixup_alpha']),
            cutmix_alpha = float(config['params']['cutmix_alpha']),
            mixup_prob = float(config['params']['mixup_prob']),
            mixup_switch_prob = float(config['params']['mixup_switch_prob']),
            mixup_mode = config['params']['mixup_mode'],
            label_smoothing = float(config['params']['label_smoothing']),
        )

        # Callbacks
        checkpoint_metric = config['params']['checkpoint_metric']
        checkpoint_mode = "min" if checkpoint_metric == "val_loss" else "max"
        
        # Callbacks
        callbacks = []
        callbacks.append(LearningRateMonitor(logging_interval="step"))
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor=checkpoint_metric,
            mode=checkpoint_mode,
            dirpath="./checkpoint/",
        )
        callbacks.append(checkpoint_callback)

        early_stopping_callback = EarlyStopping(
            monitor=checkpoint_metric,
            patience=10,
            mode=checkpoint_mode,
            verbose=True,
        )
        callbacks.append(early_stopping_callback)

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
        useful_keys = ['project',
                    'system',
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