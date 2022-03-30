import torch.nn as nn
from torchmetrics import AUROC

from .AbsModule import ABSModule


class ClassificationModule(ABSModule):
    """
    pl.LightningModule for training classification on receipts
    """
    def __init__(self,
                 seq_encoder: nn.Module,
                 head: nn.Module,
                 lr: float = 1e-4,
                 weight_decay: float = 0.1,
                 lr_scheduler_step_size: int = 2,
                 lr_scheduler_step_gamma: float = 0.5,
                 print_info=False
                 ):
        """
        Параметры во многом те же, что и в EmbeddingModule
        """
        train_params = {
            'train.lr': lr,
            'train.weight_decay': weight_decay,
            'lr_scheduler': {
                'step_size': lr_scheduler_step_size,
                'step_gamma': lr_scheduler_step_gamma
            }
        }
        super().__init__(train_params, seq_encoder)

        self._head = head
        self.print_info = print_info

    @property
    def metric_name(self):
        return 'auc-roc'

    @property
    def is_requires_reduced_sequence(self):
        return True

    def get_loss(self):
        loss = nn.BCELoss()
        return loss

    def get_validation_metric(self):
        return AUROC()

    def validation_step(self, batch, _):
        y_h, y = self.shared_step(*batch)
        if y_h.shape != y.shape:
            y_h = y_h.unsqueeze(0)
        val_metric = self._validation_metric(y_h, y.clone().int())

        if self.print_info:
            print(f'Val metric: {val_metric}')

    def shared_step(self, x, y):
        y_h = self(x)
        if self._head is not None:
            y_h = self._head(y_h)
        return y_h, y

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        y_h = self(batch[0])
        if self._head is not None:
            y_h = self._head(y_h)
        return y_h

