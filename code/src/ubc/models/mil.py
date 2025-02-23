import torch
from typing import Optional
import torch.nn.functional as F
from torchmetrics import F1Score, Precision, Recall
import pytorch_lightning as L
from torchmetrics.functional.classification import multiclass_f1_score, multiclass_recall, multiclass_precision
from torch.nn import functional as F
import torch.nn as nn
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from sklearn.metrics import balanced_accuracy_score
from torch import Tensor


class AttentionMIL(nn.Module):
    def __init__(self, d_features: int, hidden_dim: Optional[int] = 256, batchnorm: bool = True, num_classes=None):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d_features, hidden_dim), nn.ReLU())
        self.attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))
        self.pre_head = nn.Sequential(nn.BatchNorm1d(hidden_dim), nn.Dropout()) if batchnorm else nn.Dropout()
        self.head = nn.Linear(in_features=hidden_dim, out_features=num_classes)

    def forward(self, feats, mask, *args, **kwargs):
        # print("feats", feats.shape)  # torch.Size([1, 49, 768])
        # print("mask", mask.shape)  # torch.Size([1, 49]) # Only if padding was required
        embeddings = self.encoder(feats)  # B, N, D
        attention = self.attention(embeddings).squeeze(-1)  # B, N
        attention = torch.masked_fill(attention, ~mask, -torch.inf)  # B, N
        attention = F.softmax(attention, dim=-1)  # B, N
        embeddings = embeddings * attention.unsqueeze(-1)  # B, N, D
        slide_tokens = embeddings.sum(dim=-2)  # B, D
        slide_tokens = self.pre_head(slide_tokens)  # B, D
        logits = self.head(slide_tokens).squeeze(-1)
        return logits


# https://lightning.ai/docs/torchmetrics/stable/pages/implement.html#implement
class BalancedAccurary(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds: Tensor, target: Tensor) -> None:
        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        y_pred = dim_zero_cat(self.preds)
        y_true = dim_zero_cat(self.target)
        score = balanced_accuracy_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
        return torch.tensor(score)


class UBCAttentionMILModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.tta = False

        # Classifier
        self.head = AttentionMIL(d_features=config.mil["d_features"], hidden_dim=config.mil["hidden_dim"],
                                 batchnorm=config.mil["batchnorm"], num_classes=config.num_classes)

        self.valid_f1 = F1Score(task="multiclass", num_classes=config.num_classes, average="macro")
        self.valid_precision = Precision(task="multiclass", num_classes=config.num_classes, average="macro")
        self.valid_recall = Recall(task="multiclass", num_classes=config.num_classes, average="macro")
        self.valid_balanced_accuracy = BalancedAccurary()

    # Classifier
    def forward_classifier(self, batch):
        feats, mask, targets, *_ = batch
        logits = self.head(feats, mask)
        return logits

    def forward(self, batch):
        # Classifier
        logits = self.forward_classifier(batch)
        # print("logits", logits)
        return logits

    def _get_preds_loss_metrics(self, batch, is_valid=False):
        feats, mask, y, *_ = batch  # (BS, N_TILES, 768), (BS, N_TILES), (BS, N_CLASSES)
        logits = self(batch)  # (BS, NC)
        preds = torch.argmax(logits, dim=1)  # (BS, C)

        w = torch.ones(self.config.num_classes, device=self.device, dtype=torch.float)
        w = w/w.sum()
        loss = F.cross_entropy(logits, y, weight=w, label_smoothing=self.config.label_smoothing, reduction="sum")

        y_true = torch.argmax(y, dim=1)  # (BS)

        # Batch score
        f1 = multiclass_f1_score(preds, y_true, num_classes=self.config.num_classes, average="macro")
        recall = multiclass_recall(preds, y_true, num_classes=self.config.num_classes, average="macro")
        precision = multiclass_precision(preds, y_true, num_classes=self.config.num_classes, average="macro")
        # Accumulate predictions/ground truth
        if is_valid:
            self.valid_f1.update(preds, y_true)
            self.valid_precision.update(preds, y_true)
            self.valid_recall.update(preds, y_true)
            self.valid_balanced_accuracy.update(preds, y_true)
        return preds, loss, f1, recall, precision

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        _, loss, f1, recall, precision = self._get_preds_loss_metrics(batch)
        # Log loss and metric
        self.log('train_step_loss', loss, batch_size=self.config.batch_size)
        self.log('train_step_f1', f1, batch_size=self.config.batch_size)
        self.log('train_step_recall', recall, batch_size=self.config.batch_size)
        self.log('train_step_precision', precision, batch_size=self.config.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        preds, loss, f1, recall, precision = self._get_preds_loss_metrics(batch, is_valid=True)
        # Log loss and metric
        self.log('val_step_loss', loss, batch_size=self.config.val_batch_size)
        self.log('val_step_f1', f1, batch_size=self.config.val_batch_size)
        self.log('val_step_recall', recall, batch_size=self.config.val_batch_size)
        self.log('val_step_precision', precision, batch_size=self.config.val_batch_size)
        return preds

    def on_validation_epoch_end(self):
        self.log('val_f1', self.valid_f1.compute())
        self.log('val_precision', self.valid_precision.compute())
        self.log('val_recall', self.valid_recall.compute())
        self.log('val_balanced_accuracy', self.valid_balanced_accuracy.compute())
        self.valid_f1.reset()
        self.valid_precision.reset()
        self.valid_recall.reset()
        self.valid_balanced_accuracy.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(batch)
        preds = torch.argmax(logits, dim=1)
        topk = torch.topk(torch.softmax(logits, dim=1), 1)
        return topk.indices, logits, topk.values

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config.epochs, eta_min=self.config.lrf)
        return [optimizer], [scheduler]
