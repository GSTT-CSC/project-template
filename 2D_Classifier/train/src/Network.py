import logging
import pytorch_lightning
import mlflow
import torch
from abc import ABC
from monai.data import decollate_batch
from monai.transforms import (
    AsDiscrete,
    Compose,
    Activations,
)
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, recall_score
from timm import create_model
from timm.data import Mixup
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy, F1Score
from src.DataModule import label_dict
from torchmetrics.classification import MulticlassAUROC
import numpy as np

logger = logging.getLogger(__name__)


class Network(pytorch_lightning.LightningModule, ABC):
    """
    Network object defines the model architecture, inherits from LightningModule.

    https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
    """

    def __init__(self,
                model_name,
                pretrained,
                n_classes,
                dropout,
                weighted_loss,
                train_class_weights,
                validation_class_weights,
                learning_rate,
                max_lr,
                nw_batch_size,
                label_smoothing,
                **kwargs):
        super().__init__()

        self.save_hyperparameters()
        self.n_classes = n_classes
        self.dropout = dropout
        self.report_interval = 5
        self.label_smoothing = label_smoothing

        self.y_pred_trans = Compose([Activations(softmax=True)])
        self.y_trans = Compose([AsDiscrete(to_onehot=self.n_classes)])

        self.learning_rate = learning_rate
        self.max_lr = max_lr
        self.batch_size = nw_batch_size

        self.val_acc = Accuracy(task="multiclass", num_classes=self.n_classes, top_k=1)
        self.val_f1 = F1Score(task="multiclass", num_classes=self.n_classes, top_k=1)
        self.val_auroc = MulticlassAUROC(num_classes=self.n_classes, average='macro', thresholds=None)

        self.targets, self.labels = list(
            map(list, zip(*[(target, label) for target, label in label_dict.items() if label is not None])))

        self.model_name = model_name
        self.pretrained = pretrained
        self.model = create_model(self.model_name, pretrained=self.pretrained, in_chans=1,
                                  num_classes=self.n_classes, drop_rate=self.dropout)
        
        self.weighted_loss = weighted_loss
        self.train_class_weights = train_class_weights
        self.validation_class_weights = validation_class_weights

        self.train_loss_function = CrossEntropyLoss(
            weight=torch.tensor(self.train_class_weights)
            if self.weighted_loss
            else None
        )

        self.validation_loss_function = CrossEntropyLoss(
            weight=torch.tensor(self.validation_class_weights)
            if self.weighted_loss
            else None
        )

        self.mixup_fn = Mixup(
            mixup_alpha = 0.2,
            cutmix_alpha = 0.2,
            prob=0.5,
            switch_prob=0.5,
            mode='batch',
            label_smoothing=0.1,
            num_classes=self.n_classes,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Training step
        :param batch:
        :param batch_idx:
        :return: loss
        """
        valid_mask = batch['valid'].bool()
        if valid_mask.sum() == 0:
            return None

        x = batch['image'][valid_mask]
        y = batch['label'][valid_mask]
        if x.size(0) % 2 == 0:
            x, y = self.mixup_fn(x, y)

        y_hat = self(x)
        loss = self.train_loss_function(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step
        :param batch:
        :param batch_idx:
        :return: loss
        """
        valid_mask = batch['valid'].bool()
        if valid_mask.sum() == 0:
            return None

        x = batch['image'][valid_mask]
        y = batch['label'][valid_mask]
        y_hat = self(x)

        loss = self.validation_loss_function(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        # additional metrics
        self.val_acc(y_hat, y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        y_onehot = [self.y_trans(i) for i in decollate_batch(y, detach=True)]
        y_pred_act = [self.y_pred_trans(i) for i in decollate_batch(y_hat, detach=True)]

        self.val_f1(y_hat, y)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        self.val_auroc(y_hat, y)
        self.log('val_auroc', self.val_auroc, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        return {"loss": loss, 'y_onehot': y_onehot, 'y_pred_act': y_pred_act}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.max_lr, total_steps=self.trainer.estimated_stepping_batches
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def evaluate_best_model(self, model, threshold_tune = False, epoch=0):
        model.eval()
        model.freeze()

        val_loader = self.trainer.datamodule.val_dataloader()

        preds, labels = [], []
        if threshold_tune == False:
            for batch in val_loader:
                images = batch['image'].to(self.device)
                label = batch['label']
                y_hat = model(images)
                pred = torch.argmax(y_hat, dim=1).cpu()

                preds.append(pred)
                labels.append(label)
            
            self._create_classification_report(preds, labels)
            self._create_confusion_matrix(preds, labels)
        else:
            for batch in val_loader:
                images = batch['image'].to(self.device)
                label = batch['label']
                y_hat = model(images)
                pred = torch.softmax(y_hat, dim=1)[:, 1].cpu().numpy()

                preds.extend(pred)
                labels.extend(label)
            
            best_thresh = 0.5
            best_sensitivity = 0
        
            for t in np.linspace(0,1,100):
                t_preds = (np.array(preds) >= t).astype(int)

                sens = recall_score(labels, t_preds)
                if sens > best_sensitivity:
                    best_sensitivity = sens
                    best_thresh = t
            
            d_preds = (np.array(preds) >= 0.5).astype(int)
            d_sens = recall_score(labels,d_preds)

            threshold_text = f"Default Sensitivity: {d_sens}\nBest Threshold: {best_thresh}\nBest Sensitivity: {best_sensitivity}"
            with open("threshold.txt", "w") as f:
                f.write(threshold_text)
            mlflow.log_artifact('threshold.txt')
            preds = (np.array(preds) >= best_thresh).astype(int)
            self._create_classification_report(preds, labels)
            self._create_confusion_matrix(preds, labels)

    def validation_epoch_end(self, validation_step_outputs) -> None:
        if self.current_epoch % self.report_interval == 0:
            self._attribute(n_samples_plot=4, step_type='validation')

    def _create_classification_report(self, preds, labels):
        val_report = classification_report(labels, preds, labels=self.labels, target_names=self.targets)
        mlflow.log_text(val_report, f'classification_report.txt')

    def _create_confusion_matrix(self, preds, labels):
        confmat = confusion_matrix(labels, preds, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=confmat)
        disp.plot()
        mlflow.log_figure(disp.figure_, 'confusion_matrix.jpg')