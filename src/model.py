from transformers import AutoModelForImageClassification, AutoImageProcessor
from transformers.optimization import get_cosine_schedule_with_warmup
import lightning as L
from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import Accuracy
# from .loss import FocalLoss

class ImageClassifier(L.LightningModule):
    def __init__(
            self,
            learning_rate = 1e-5,
            warmup_steps = 500,
            lora_rank = 8,
            lora_alpha = 8,
            target_modules = ['query', 'value'],
            lora_dropout = 0.1,
            use_rslora = True,
            class_weight = None,
            loss_fn = None
    ):
        super().__init__()


        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.target_modules = target_modules
        self.lora_dropout = lora_dropout
        self.use_rslora = use_rslora
        # self.modules_to_save = modules_to_save
        self.class_weight = class_weight
        self.loss_fn = loss_fn
        self.train_accuracy = Accuracy(task='binary')
        self.val_accuracy = Accuracy(task='binary')

        self.model = AutoModelForImageClassification.from_pretrained(
            'google/vit-base-patch32-224-in21k',
            num_labels = 1,
            ignore_mismatched_sizes = True,
            token = False
        )

        lora_config = LoraConfig(
            r = self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            use_rslora=self.use_rslora,
            modules_to_save=['classifier']
        )

        self.lora_model = get_peft_model(self.model, lora_config)

    def forward(self, x):
        return self.lora_model(pixel_values = x).logits
    

    def training_step(self, batch, batch_idx):
        X, y = batch
        #print(f"Before :: label: {y} :: labels shape: {y.shape} :: labels dtype: {y.dtype}")
        y = y.float()
        #print(f"After :: label: {y} :: labels shape: {y.shape} :: labels dtype: {y.dtype}")
        logits = self(X)
        logits = logits.view(-1)
        # print(f"Logits shape :: {logits.shape}")
        preds = (torch.sigmoid(logits)>0.5).long()
        loss = self.loss_fn(logits, y)
        acc = self.train_accuracy(preds, y.long())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_accuracy", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y = y.float()
        logits = self(X)
        logits = logits.view(-1)
        
        preds = (torch.sigmoid(logits)>0.5).long()
        loss = self.loss_fn(logits, y)
        acc = self.val_accuracy(preds, y.long())
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




