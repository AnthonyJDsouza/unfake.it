from src.loss import FocalLoss
from src.model import ImageClassifier
from src.data_gpu import prepare_dataloaders
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
import wandb
from datasets import load_from_disk, concatenate_datasets
import argparse

MODEL_NAME = "google/vit-base-patch32-224-in21k"

parser = argparse.ArgumentParser()
parser.add_argument('--ds0', type=str, required=True)
parser.add_argument('--ds1', type=str, required=True)
parser.add_argument('--ds2', type=str, required=True)
args = parser.parse_args()


ds0 = load_from_disk(args.ds0)
ds1 = load_from_disk(args.ds1)
ds2 = load_from_disk(args.ds2)

ds = concatenate_datasets([ds1, ds2])

train_dl, val_dl, num_labels = prepare_dataloaders(
    train_dataset=ds,
    val_dataset=ds0,
    model_name=MODEL_NAME
)

loss_fn = FocalLoss(gamma=0.8, alpha=0.75)

model = ImageClassifier(
    model_name=MODEL_NAME,
    num_labels=num_labels,
    loss_fn=loss_fn
)

wandb_logger = WandbLogger(project = "ViT finetuning")
trainer = Trainer(
    max_epochs=1,
    accelerator='auto',
    precision='bf16-mixed',
    logger=wandb_logger
)

trainer.fit(
    model = model,
    train_dataloaders=train_dl,
    val_dataloaders=val_dl,
)

torch.save(model.state_dict(), '/scratch/ajdsouza/models/vit/test_gpu.pth')

