"""
Módulo de la capa de aplicación.
"""
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import pytorch_lightning as pl
from typing import Tuple, Dict, Any
from config.settings import BERTConfig
from domain.architecture import BERT


class BERTPreTrainingLightning(pl.LightningModule):
    """Módulo Lightning maestro que coordina los loops y los gradientes."""
    def __init__(self, config: BERTConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.bert = BERT(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            max_len=config.max_len,
            dropout=config.dropout,
            pad_idx=config.pad_idx
        )
        self.mlm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size, bias=False
            )
        self.mlm_head.weight = self.bert.embedding.token_embedding.weight
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
            self,
            input_ids: torch.Tensor,
            segment_ids: torch.Tensor
            ) -> torch.Tensor:
        """Forwad."""
        hidden_states = self.bert(input_ids, segment_ids)
        return self.mlm_head(hidden_states) + self.bias

    def training_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            batch_idx: int
            ) -> torch.Tensor:
        """Training Step."""
        input_ids, segment_ids, labels = batch
        logits = self(input_ids, segment_ids)

        logits_flat = logits.view(-1, self.config.vocab_size)
        labels_flat = labels.view(-1)

        loss = self.criterion(logits_flat, labels_flat)
        mask = labels_flat != -100
        if mask.sum() > 0:
            preds = logits_flat.argmax(dim=-1)
            correct = (preds[mask] == labels_flat[mask]).sum().float()
            accuracy = correct / mask.sum().float()
        else:
            accuracy = torch.tensor(0.0, device=self.device)

        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", current_lr, prog_bar=True, logger=True)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_acc", accuracy, prog_bar=True, logger=True)
        return loss

    def validation_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            batch_idx: int
            ) -> torch.Tensor:

            input_ids, segment_ids, labels = batch
            logits = self(input_ids, segment_ids)

            logits_flat = logits.view(-1, self.config.vocab_size)
            labels_flat = labels.view(-1)

            loss = self.criterion(logits_flat, labels_flat)
            mask = labels_flat != -100

            if mask.sum() > 0:
                preds = logits_flat.argmax(dim=-1)
                correct = (preds[mask] == labels_flat[mask]).sum().float()
                accuracy = correct / mask.sum().float()
            else:
                accuracy = torch.tensor(0.0, device=self.device)

            self.log(
                "val_loss",
                loss,
                prog_bar=True,
                logger=True,
                sync_dist=True
                )

            self.log(
                "val_acc",
                accuracy,
                prog_bar=True,
                logger=True,
                sync_dist=True
                )

            return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configuring Optimizers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate
            )

        try:
            batches_per_epoch = len(self.trainer.datamodule.train_dataloader())
            total_steps = batches_per_epoch * self.trainer.max_epochs
        except Exception:
            total_steps = 100

        # Evitamos que warmup_steps sea 0 en datasets pequeños
        warmup_steps = max(1, min(self.config.warmup_steps, total_steps // 10))

        # Configuramos los schedulers
        scheduler1 = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps)

        # T_max no puede ser 0, lo protegemos
        t_max = max(1, total_steps - warmup_steps)
        scheduler2 = CosineAnnealingLR(optimizer, T_max=t_max)

        scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[warmup_steps]
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
                }
        }
