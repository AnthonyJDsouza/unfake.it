from transformers import AutoModelForImageClassification, AutoImageProcessor
from transformers.optimization import get_cosine_schedule_with_warmup
import lightning as L
from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import Accuracy
from .loss import FocalLoss

class ImageClassifier(L.LightningModule):
    def __init__(
            self,
            model_name,
            num_labels,
            learning_rate = 1e-5,
            warmup_steps = 500,
            lora_rank = 8,
            lora_alpha = 8,
            target_modules = ['query', 'value'],
            lora_dropout = 0.1,
            use_rslora = True,
            loss_fn = nn.BCEWithLogitsLoss(),
    ):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules
        self.lora_dropout = lora_dropout
        self.use_rslora = use_rslora
        # self.modules_to_save = modules_to_save
        self.loss_fn = loss_fn
        self.train_accuracy = Accuracy(task='binary')
        self.val_accuracy = Accuracy(task='binary')

        self.model = AutoModelForImageClassification(
            model_name,
            num_labels = 1,
            ignore_mismatched_sizes = True
        )

        lora_config = LoraConfig(
            r = self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            use_rslora=self.use_rslora
        )

        self.lora_model = get_peft_model(self.model, lora_config)

    def forward(self, x):
        return self.lora_model(pixel_values = x).logits
    

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = self.loss_fn(y_hat, y.float())
        acc = self.train_accuracy(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = self.loss_fn(y_hat, y.float())
        acc = self.val_accuracy(y_hat, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_accuracy", acc, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.lora_model.parameters(),
            lr = self.learning_rate,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=int(self.trainer.estimated_stepping_batches),
            num_warmup_steps=self.warmup_steps
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            },
        }




