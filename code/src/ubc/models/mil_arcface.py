import torch
from typing import Optional
import math
import numpy as np
from torchmetrics import F1Score, Precision, Recall
import pytorch_lightning as L
from torchmetrics.functional.classification import multiclass_f1_score, multiclass_recall, multiclass_precision
from torch.nn import functional as F
import torch.nn as nn
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
from sklearn.metrics import balanced_accuracy_score
from torch import Tensor


class ArcMarginProduct_subcenter(nn.Module):
    def __init__(self, in_features, out_features, k=3):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features * k, in_features))
        self.reset_parameters()
        self.k = k
        self.out_features = out_features

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features, label=None):
        cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
        cosine_all = cosine_all.view(-1, self.out_features, self.k)
        cosine, _ = torch.max(cosine_all, dim=2)
        return cosine  # Logits


class ArcFaceLossAdaptiveMargin(nn.Module):
    def __init__(self, margins, n_classes, s=30.0):
        super().__init__()
        self.s = s
        self.margins = margins if isinstance(margins, np.ndarray) else np.array(margins)
        self.out_dim = n_classes
        # print('self.margins:', self.margins)

    def forward(self, logits, labels=None):
        ms = self.margins[labels.cpu().numpy()]
        cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
        sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
        th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
        mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
        labels = F.one_hot(labels, self.out_dim).float()
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * cos_m.view(-1, 1) - sine * sin_m.view(-1, 1)
        phi = torch.where(cosine > th.view(-1, 1), phi, cosine - mm.view(-1, 1))
        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        return output


class ArcFaceLossAdaptiveMarginSubcenter(nn.Module):
    def __init__(self, in_features, n_classes, margins, s=30.0, k=3):
        super().__init__()

        self.subcenter = ArcMarginProduct_subcenter(in_features, n_classes, k=k)
        self.loss_module = ArcFaceLossAdaptiveMargin(margins, n_classes, s=s)

    def forward(self, features, label=None):
        logits = self.subcenter.forward(features)
        if self.training:
            logits = self.loss_module.forward(logits, label)
        return logits


def get_loss_module(cfg):
    if cfg.loss_module == "subcenter_arcfaceadaptivemargin":
        return ArcFaceLossAdaptiveMarginSubcenter(cfg.fc_dim, cfg.num_classes, cfg.loss_module_margins,
                                                  s=cfg.loss_module_cosine_scale, k=cfg.loss_module_k)
    else:
        return nn.Linear(in_features=cfg.fc_dim, out_features=cfg.num_classes)


class AttentionMILArcFace(nn.Module):
    def __init__(self, d_features: int, hidden_dim: Optional[int] = 256, batchnorm: bool = True, num_classes=None, cfg=None):
        super().__init__()

        self.cfg = cfg
        self.encoder = nn.Sequential(nn.Linear(d_features, hidden_dim), nn.ReLU())
        self.attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(), nn.Linear(hidden_dim // 2, 1))
        self.pre_head = nn.Sequential(nn.BatchNorm1d(hidden_dim), nn.Dropout()) if batchnorm else nn.Dropout()
        # Loss module that could cluster embeddings for the given classes
        self.head = get_loss_module(self.cfg)

    def forward(self, feats, mask, *args, **kwargs):
        # print("feats", feats.shape)  # torch.Size([1, 49, 768])
        # print("mask", mask.shape)  # torch.Size([1, 49]) # Only if padding was required
        labels = kwargs.get("labels")  # torch.Size([1, 5])
        if labels is not None:
            # OHE is not expected here.
            labels = torch.argmax(labels.long(), dim=1)

        embeddings = self.encoder(feats)  # BS, N_TILES, D
        attention = self.attention(embeddings).squeeze(-1)  # BS, N_TILES
        attention = torch.masked_fill(attention, ~mask, -torch.inf)  # BS, N_TILES
        attention = F.softmax(attention, dim=-1)  # BS, N_TILES
        embeddings = embeddings * attention.unsqueeze(-1)  # BS, N_TILES, D
        slide_tokens = embeddings.sum(dim=-2)  # BS, D
        slide_tokens = self.pre_head(slide_tokens)  # BS, D

        # print("slide_tokens", slide_tokens.shape)  # torch.Size([BS, 512])
        # print("labels", labels.shape)  # torch.Size([BS, 5])
        # Labels (training only) must not be OHE here, slide_tokens are our embeddings for clustering
        if self.cfg.loss_module == "subcenter_arcfaceadaptivemargin":
            logits = self.head(slide_tokens, label=labels).squeeze(-1)
        else:
            logits = self.head(slide_tokens).squeeze(-1)

        return logits, slide_tokens


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


class UBCAttentionMILModelArcFace(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.tta = False

        # Classifier
        self.head = AttentionMILArcFace(d_features=config.mil["d_features"], hidden_dim=config.mil["hidden_dim"],
                                        batchnorm=config.mil["batchnorm"], num_classes=config.num_classes, cfg=config)

        self.valid_f1 = F1Score(task="multiclass", num_classes=config.num_classes, average="macro")
        self.valid_precision = Precision(task="multiclass", num_classes=config.num_classes, average="macro")
        self.valid_recall = Recall(task="multiclass", num_classes=config.num_classes, average="macro")
        self.valid_balanced_accuracy = BalancedAccurary()

    # Classifier
    def forward_classifier(self, batch):
        feats, masks, targets, *_ = batch
        logits = self.head(feats, masks, labels=targets)
        # logits = self.head(batch)
        return logits

    def forward(self, batch):
        # Classifier
        logits = self.forward_classifier(batch)
        # print("logits", logits)
        return logits

    def _get_preds_loss_metrics(self, batch, is_valid=False):
        feats, mask, y, *_ = batch  # (BS, N_TILES, 768), (BS, N_TILES), (BS, N_CLASSES)
        logits, embeddings = self(batch)  # (BS, NC)
        preds = torch.argmax(logits, dim=1)  # (BS, C)

        w = torch.ones(self.config.num_classes, device=self.device, dtype=torch.float)
        w = w/w.sum()
        loss = F.cross_entropy(logits, y, weight=w, label_smoothing=self.config.label_smoothing, reduction="sum")
        # loss = F.cross_entropy(logits, y, label_smoothing=self.config.label_smoothing)

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
        logits, embeddings = self(batch)
        preds = torch.argmax(logits, dim=1)
        topk = torch.topk(torch.softmax(logits, dim=1), 1)
        return topk.indices, logits, topk.values, embeddings

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr0)
        if self.config.scheduler == "lw_cos_lr":
            from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
            scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.config.warmup_epochs,
                                                      max_epochs=self.config.epochs, eta_min=self.config.lrf)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.config.epochs, eta_min=self.config.lrf)
        return [optimizer], [scheduler]
