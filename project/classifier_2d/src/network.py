import logging
import random
from abc import ABC

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pytorch_lightning
import torch
from captum.attr import Occlusion
from monai.data import decollate_batch
from monai.transforms import Activations, AsDiscrete, Compose
from sklearn.metrics import (
                             ConfusionMatrixDisplay, confusion_matrix,
                             precision_recall_curve, roc_curve, auc, average_precision_score,
                             precision_score, recall_score, f1_score)
from timm import create_model
from timm.data import Mixup
from torch.utils.data import DataLoader, Subset
from torchmetrics import Accuracy, F1Score, FBetaScore, Precision, Recall
from torchmetrics.classification import MulticlassAUROC, BinaryPrecisionRecallCurve

from src.utils import get_loss_function

logger = logging.getLogger(__name__)

class Network(pytorch_lightning.LightningModule, ABC):
    """
    Network object defines the model architecture, inherits from LightningModule.

    https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html
    """

    def __init__(self,
                n_classes,
                label_dict,
                model_name,
                pretrained,
                learning_rate,
                max_lr,
                batch_size,
                dropout,
                train_class_weights,
                validation_class_weights,
                test_class_weights,
                weighted_loss,
                loss_fcn,
                weight_decay,
                mixup_alpha,
                cutmix_alpha,
                mixup_prob,
                mixup_switch_prob,
                mixup_mode,
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
        self.batch_size = batch_size

        task = "binary" if self.n_classes == 2 else "multiclass"
        self.train_acc = Accuracy(task=task, num_classes=self.n_classes, top_k=1)
        self.train_f1 = F1Score(task=task, num_classes=self.n_classes, top_k=1)
        self.train_f05 = FBetaScore(task=task, num_classes=self.n_classes, beta=0.5)
        self.train_precision = Precision(task=task, num_classes=self.n_classes)
        self.train_recall = Recall(task=task, num_classes=self.n_classes)
        self.train_auroc = MulticlassAUROC(num_classes=self.n_classes, average='macro', thresholds=None)
        self.train_pr_curve = BinaryPrecisionRecallCurve()

        self.val_acc = Accuracy(task=task, num_classes=self.n_classes, top_k=1)
        self.val_f1 = F1Score(task=task, num_classes=self.n_classes, top_k=1)
        self.val_f05 = FBetaScore(task=task, num_classes=self.n_classes, beta=0.5)
        self.val_precision = Precision(task=task, num_classes=self.n_classes)
        self.val_recall = Recall(task=task, num_classes=self.n_classes)
        self.val_auroc = MulticlassAUROC(num_classes=self.n_classes, average='macro', thresholds=None)
        self.val_pr_curve = BinaryPrecisionRecallCurve()

        self.test_acc = Accuracy(task=task, num_classes=self.n_classes, top_k=1)
        self.test_f1 = F1Score(task=task, num_classes=self.n_classes, top_k=1)
        self.test_f05 = FBetaScore(task=task, num_classes=self.n_classes, beta=0.5)
        self.test_precision = Precision(task=task, num_classes=self.n_classes)
        self.test_recall = Recall(task=task, num_classes=self.n_classes)
        self.test_auroc = MulticlassAUROC(num_classes=self.n_classes, average='macro', thresholds=None)
        self.test_pr_curve = BinaryPrecisionRecallCurve()

        self.targets, self.labels = list(
            map(list, zip(*[(target, label) for target, label in label_dict.items() if label is not None])))

        self.model_name = model_name
        self.pretrained = pretrained
        self.model = create_model(self.model_name, pretrained=self.pretrained, in_chans=1,
                                  num_classes=self.n_classes, drop_rate=self.dropout)
        
        self.weighted_loss = weighted_loss
        self.loss_fcn = loss_fcn
        self.weight_decay = weight_decay

        if self.weighted_loss:
            self.register_buffer('train_class_weights_tensor',
                                torch.tensor(train_class_weights, dtype=torch.float32))
            self.register_buffer('validation_class_weights_tensor',
                                torch.tensor(validation_class_weights, dtype=torch.float32))
            self.register_buffer('test_class_weights_tensor',
                                torch.tensor(test_class_weights, dtype=torch.float32))
        else:
            self.train_class_weights_tensor = None
            self.validation_class_weights_tensor = None
            self.test_class_weights_tensor = None

        self.train_loss_function = get_loss_function(loss_fcn, weight=self.train_class_weights_tensor, label_smoothing=self.label_smoothing)
        self.validation_loss_function = get_loss_function(loss_fcn, weight=self.validation_class_weights_tensor)
        self.test_loss_function = get_loss_function(loss_fcn, weight=self.test_class_weights_tensor)

        self.mixup_fn = Mixup(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            prob=mixup_prob,
            switch_prob=mixup_switch_prob,
            mode=mixup_mode,
            label_smoothing=self.label_smoothing,
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
            return torch.tensor(0.0, requires_grad=True, device=self.device)

        x = batch['image'][valid_mask]
        y = batch['label'][valid_mask]

        valid_labels_mask = y >= 0
        if valid_labels_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        
        x = x[valid_labels_mask]
        y = y[valid_labels_mask]
        y_original = y.clone()

        if x.size(0) % 2 == 0:
            x, y = self.mixup_fn(x, y)

        y_hat = self(x)
        loss = self.train_loss_function(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        # log metrics but expect acc and F1 score to be reduced if mixup is used
        train_probs = torch.softmax(y_hat, dim=1)[:, 1] if self.n_classes == 2 else y_hat
        
        self.train_acc(train_probs, y_original)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        self.train_f1(train_probs, y_original)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        self.train_f05(train_probs, y_original)
        self.log('train_f05', self.train_f05, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        self.train_precision(train_probs, y_original)
        self.log('train_precision', self.train_precision, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        self.train_recall(train_probs, y_original)
        self.log('train_recall', self.train_recall, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        self.train_auroc(y_hat, y_original)
        self.log('train_auroc', self.train_auroc, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        self.train_pr_curve.update(train_probs, y_original)

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
            return torch.tensor(0.0, device=self.device)

        x = batch['image'][valid_mask]
        y = batch['label'][valid_mask]

        valid_labels_mask = y >= 0
        if valid_labels_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        x = x[valid_labels_mask]
        y = y[valid_labels_mask]
        y_hat = self(x)

        loss = self.validation_loss_function(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        # additional metrics
        val_probs = torch.softmax(y_hat, dim=1)[:, 1] if self.n_classes == 2 else y_hat
        
        self.val_acc(val_probs, y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        self.val_f1(val_probs, y)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        self.val_auroc(y_hat, y)
        self.log('val_auroc', self.val_auroc, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        self.val_f05(val_probs, y)
        self.log('val_f05', self.val_f05, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        self.val_precision(val_probs, y)
        self.log('val_precision', self.val_precision, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        self.val_recall(val_probs, y)
        self.log('val_recall', self.val_recall, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        self.val_pr_curve.update(val_probs, y)

        y_onehot = [self.y_trans(i) for i in decollate_batch(y, detach=True)]
        y_pred_act = [self.y_pred_trans(i) for i in decollate_batch(y_hat, detach=True)]

        return {"loss": loss, 'y_onehot': y_onehot, 'y_pred_act': y_pred_act}

    def test_step(self, batch, batch_idx):
        valid_mask = batch['valid'].bool()
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        x = batch['image'][valid_mask]
        y = batch['label'][valid_mask]

        valid_labels_mask = y >= 0
        if valid_labels_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        x = x[valid_labels_mask]
        y = y[valid_labels_mask]
        y_hat = self(x)

        # if test_fraction=0, loss calculation will raise an error - this is intentional, and aims to flag
        # to the user that this should not happen - test step should not run if test_fraction=0.
        loss = self.test_loss_function(y_hat, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        test_probs = torch.softmax(y_hat, dim=1)[:, 1] if self.n_classes == 2 else y_hat

        self.test_acc(test_probs, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        self.test_f1(test_probs, y)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        self.test_f05(test_probs, y)
        self.log('test_f05', self.test_f05, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        self.test_precision(test_probs, y)
        self.log('test_precision', self.test_precision, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        self.test_recall(test_probs, y)
        self.log('test_recall', self.test_recall, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        self.test_auroc(y_hat, y)
        self.log('test_auroc', self.test_auroc, on_step=False, on_epoch=True, batch_size=self.batch_size, sync_dist=True)

        self.test_pr_curve.update(test_probs, y)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                    lr=self.learning_rate,
                                    weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.max_lr, total_steps=self.trainer.estimated_stepping_batches
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def evaluate_best_model(self, model, split='val'):
        model.eval()
        model.freeze()

        loader = self.trainer.datamodule.val_dataloader() if split == 'val' else self.trainer.datamodule.test_dataloader()

        probs, labels = [], []
        for batch in loader:
            valid_mask = batch['valid'].bool()
            if valid_mask.sum() == 0:
                continue
            images = batch['image'][valid_mask].to(self.device)
            label = batch['label'][valid_mask]
            valid_labels_mask = label >= 0
            if valid_labels_mask.sum() == 0:
                continue
            images = images[valid_labels_mask]
            label = label[valid_labels_mask]
            y_hat = model(images)
            prob = torch.softmax(y_hat, dim=1)[:, 1].cpu().numpy()
            probs.extend(prob)
            labels.extend(label.cpu().numpy())

        probs = np.array(probs)
        labels = np.array(labels)

        self._create_roc_curve(probs, labels, split)
        self._create_pr_curve(probs, labels, split)
        self._create_threshold_analysis(probs, labels, split)
        self._create_confusion_matrices(probs, labels, split)
        self._attribute(model=model, n_samples_plot=4, step_type='best', epoch='best', split=split)

    def _create_roc_curve(self, probs, labels, split='val'):
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--', lw=1)
        ax.set_xlabel('False Positive Rate (1 - Specificity)')
        ax.set_ylabel('True Positive Rate (Sensitivity)')
        ax.set_title('ROC Curve')
        ax.legend(loc='lower right')
        plt.tight_layout()
        mlflow.log_figure(fig, f'evaluation/{split}/roc_curve.png')
        mlflow.log_metric(f'{split}_best_model_auroc', roc_auc)
        plt.close(fig)

    def _create_pr_curve(self, probs, labels, split='val'):
        precision, recall, _ = precision_recall_curve(labels, probs)
        ap = average_precision_score(labels, probs)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AP = {ap:.3f})')
        baseline = labels.mean()
        ax.axhline(y=baseline, color='grey', linestyle='--', lw=1, label=f'Baseline (prevalence = {baseline:.2f})')
        ax.set_xlabel('Recall (Sensitivity)')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='upper right')
        plt.tight_layout()
        mlflow.log_figure(fig, f'evaluation/{split}/pr_curve.png')
        mlflow.log_metric(f'{split}_best_model_auprc', ap)
        plt.close(fig)

    def _create_threshold_analysis(self, probs, labels, split='val'):
        thresholds = np.linspace(0.01, 0.99, 200)
        rows = []
        for t in thresholds:
            preds = (probs >= t).astype(int)
            if preds.sum() == 0:
                prec, rec, f1, spec = 1.0, 0.0, 0.0, 1.0
            else:
                prec = precision_score(labels, preds, zero_division=1)
                rec = recall_score(labels, preds, zero_division=0)
                f1 = f1_score(labels, preds, zero_division=0)
                tn = ((1 - labels) * (1 - preds)).sum()
                fp = ((1 - labels) * preds).sum()
                spec = tn / (tn + fp) if (tn + fp) > 0 else 1.0
            rows.append((t, prec, rec, spec, f1))

        rows = np.array(rows)
        thresh_arr, prec_arr, rec_arr, spec_arr, f1_arr = rows.T

        # Plot precision & recall vs threshold
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(thresh_arr, prec_arr, label='Precision', color='darkorange', lw=2)
        ax.plot(thresh_arr, rec_arr, label='Recall (Sensitivity)', color='steelblue', lw=2)
        ax.plot(thresh_arr, spec_arr, label='Specificity', color='green', lw=2)
        ax.plot(thresh_arr, f1_arr, label='F1', color='purple', lw=1.5, linestyle='--')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Score')
        ax.set_title('Precision, Recall, Specificity & F1 vs Threshold')
        ax.legend()
        plt.tight_layout()
        mlflow.log_figure(fig, f'evaluation/{split}/metrics_vs_threshold.png')
        plt.close(fig)

        # Precision-anchored summary table
        precision_targets = [0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55]
        lines = [f"{'Precision Target':>18} | {'Threshold':>10} | {'Achieved Precision':>14} | {'Recall':>8} | {'Specificity':>12} | {'F1':>6}"]
        lines.append('-' * 82)
        for target in precision_targets:
            # find rows where precision >= target, pick lowest threshold (highest recall)
            mask = prec_arr >= target
            if mask.any():
                idx = np.where(mask)[0][0]  # lowest threshold meeting precision target
                lines.append(
                    f"{target:>18.0%} | {thresh_arr[idx]:>10.3f} | {prec_arr[idx]:>14.3f} | "
                    f"{rec_arr[idx]:>8.3f} | {spec_arr[idx]:>12.3f} | {f1_arr[idx]:>6.3f}"
                )
            else:
                lines.append(f"{target:>18.0%} | {'N/A':>10} | {'model cannot reach this precision target':>50}")

        # Also report Youden's J optimal threshold for reference
        j_scores = rec_arr + spec_arr - 1
        best_j_idx = np.argmax(j_scores)
        lines.append('')
        lines.append(f"Youden's J optimum  | threshold={thresh_arr[best_j_idx]:.3f} | "
                     f"precision={prec_arr[best_j_idx]:.3f} | recall={rec_arr[best_j_idx]:.3f} | "
                     f"specificity={spec_arr[best_j_idx]:.3f}")

        mlflow.log_text('\n'.join(lines), f'evaluation/{split}/threshold_analysis.txt')

    def _create_confusion_matrices(self, probs, labels, split='val'):
        # Confusion matrices at a few key thresholds
        key_thresholds = [0.3, 0.5, 0.7, 0.9]
        fig, axs = plt.subplots(1, len(key_thresholds), figsize=(5 * len(key_thresholds), 5))
        for ax, t in zip(axs, key_thresholds):
            preds = (probs >= t).astype(int)
            cm = confusion_matrix(labels, preds, normalize='true')
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.targets)
            disp.plot(ax=ax, colorbar=False)
            prec = precision_score(labels, preds, zero_division=1)
            rec = recall_score(labels, preds, zero_division=0)
            ax.set_title(f'Threshold={t:.2f}\nPrec={prec:.2f} Rec={rec:.2f}')
        plt.tight_layout()
        mlflow.log_figure(fig, f'evaluation/{split}/confusion_matrices.png')
        plt.close(fig)

    def _compute_pr_metrics(self, pr_curve_metric, prefix):
        precision, recall, _ = pr_curve_metric.compute()
        pr_curve_metric.reset()
        auprc = torch.trapezoid(precision.flip(0), recall.flip(0)).item()
        mask = precision[:-1] >= 0.90
        recall_at_p90 = recall[:-1][mask].max().item() if mask.any() else 0.0
        self.log(f'{prefix}_auprc', auprc, sync_dist=True)
        self.log(f'{prefix}_recall_at_p90', recall_at_p90, sync_dist=True)

    def on_training_epoch_end(self) -> None:
        self._compute_pr_metrics(self.train_pr_curve, 'train')

    def on_validation_epoch_end(self) -> None:
        self._compute_pr_metrics(self.val_pr_curve, 'val')
        if self.current_epoch % self.report_interval == 0:
            self._attribute(n_samples_plot=4, step_type='validation')
    
    def on_test_epoch_end(self) -> None:
        self._compute_pr_metrics(self.test_pr_curve, 'test')

    def _attribute(self, model = None, n_samples_plot: int = 4, step_type: str = '', epoch = None, split='val'):
        
        if model is not None:
            model.eval()
        ds = self.trainer.datamodule.val_dataset if split == 'val' else self.trainer.datamodule.test_dataset
        sample_idx = random.sample(range(len(ds)), min(len(ds), n_samples_plot))
        fig, axs = plt.subplots(len(sample_idx), 3, figsize=(16, 16), dpi=80)
        subset = Subset(ds, sample_idx)
        test_dl = DataLoader(subset, batch_size=1)  

        for i, batch in enumerate(test_dl):
            image = batch['image'].to(self.device).detach().requires_grad_()
            if model is not None:
                output = model(image)
            else:
                output = self(image)
            logit, pred_label_idx = torch.topk(F.softmax(output, dim=1), 1)
            prediction_score = logit.item()
            id = batch['subject_id']
            label = batch['label']

            occlusion = Occlusion(model if model is not None else self)

            attributions_occ = occlusion.attribute(image,
                                                   strides=(1, 32, 32),
                                                   target=pred_label_idx,
                                                   sliding_window_shapes=(1, 64, 64),
                                                   baselines=0,
                                                   show_progress=False)

            if len(sample_idx) == 1:  # add singleton dim to axes
                axs = axs[None, :]

            axs[i, 0].set_title(f"Image {id} label: {self.targets[label.item()]}")
            axs[i, 0].imshow(image[0, 0, :, :].cpu().detach().squeeze(), cmap="gray", )

            axs[i, 1].set_title(f"attr_occ - pred: {self.targets[pred_label_idx]} ({prediction_score :.3f})")
            axs[i, 1].imshow(attributions_occ[0, 0, :, :].cpu().detach().squeeze())

            axs[i, 2].set_title(
                f"attr_occ_overlay - pred: {self.targets[pred_label_idx]} ({prediction_score :.3f})")
            axs[i, 2].imshow(image[0, 0, :, :].cpu().detach().squeeze(), cmap="gray")
            axs[i, 2].imshow(attributions_occ[0, 0, :, :].cpu().detach().squeeze(), cmap="BrBG", alpha=0.4)

        plt.tight_layout()
        plt.show()
        epoch_str = str(epoch if epoch is not None else str(self.current_epoch).zfill(4))
        filename = f"attribution_maps/{step_type}/epoch_{epoch_str}.png"
        mlflow.log_figure(fig, filename)
        plt.close()