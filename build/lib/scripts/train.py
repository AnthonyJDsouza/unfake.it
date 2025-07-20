from src.loss import FocalLoss
from src.model import ImageClassifier
from src.data import DFDCDataset, PadToSquare
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision.transforms import v2
from transformers import AutoImageProcessor
from torch.utils.data import DataLoader, ConcatDataset, Subset, random_split
import torch
import torch.nn as nn
import wandb
import argparse
import pandas as pd

MODEL_NAME = "google/vit-base-patch32-224-in21k"

# parser = argparse.ArgumentParser()
# parser.add_argument('--csv_path', type=str, required=True)
# parser.add_argument('--ds1', type=str, required=True)
# parser.add_argument('--ds2', type=str, required=True)
# args = parser.parse_args()

#c = pd.read_csv(args.csv_path)
#l = c['label'].to_list()
#lf = len(l) / 2 * len([x for x in l if x == 'FAKE'])
#lr = len(l) / 2 * len([x for x in l if x == 'REAL'])
# weight = torch.tensor([lf, lr])
# loss_fn = FocalLoss(gamma=0.8, alpha=0.75)
loss_fn = nn.BCEWithLogitsLoss()
#loss_fn = nn.CrossEntropyLoss(weight=weight)

model = ImageClassifier(
    model_name=MODEL_NAME,
    num_labels=1,
    loss_fn=loss_fn
)
loss_callback = ModelCheckpoint(
    monitor = 'train_loss',
    mode = 'min',
    filename = f"vit-best-loss",
    save_top_k=3,
    save_weights_only=False
)
processor = AutoImageProcessor.from_pretrained(MODEL_NAME, token=False)
transforms = v2.Compose([
        PadToSquare(fill = 0),
        v2.Resize(224, antialias=True),
        # PadToSquare(fill = 0),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale = True),
        v2.Normalize(mean = processor.image_mean, std = processor.image_std),
    ])
train_dfdc = DFDCDataset(csv_path='/scratch/ajdsouza/dfdc/imgagedataset/labels.csv', transform=transforms)
train_scraped = DFDCDataset(csv_path = '/scratch/ajdsouza/dfdc/dataset/scraped/label.csv', transform = transforms)
train_pexels = DFDCDataset(csv_path = '/scratch/ajdsouza/dfdc/dataset/pexels/labels.csv', transform = transforms)

mixed_ds = ConcatDataset([train_dfdc, train_scraped, train_pexels])

indices = torch.randperm(len(mixed_ds))
shuffled_mix = Subset(mixed_ds, indices)

train_size = int(0.8 * len(shuffled_mix))
val_size = len(shuffled_mix) - train_size

train_set, val_set = random_split(
    shuffled_mix,
    [train_size, val_size],
    generator = torch.Generator().manual_seed(1337)
        )


train_dl = DataLoader(dataset = train_set, batch_size = 32, num_workers = 4, shuffle = True)
valid_dl = DataLoader(dataset = val_set, batch_size = 32, num_workers = 4, shuffle = False)
wandb_logger = WandbLogger(project = "ViT-AI-Media-Detection")
trainer = Trainer(
    max_epochs=10,
    accelerator='auto',
    precision='16-mixed',
    logger=wandb_logger,
    callbacks=loss_callback
)

trainer.fit(
    model = model,
    train_dataloaders=train_dl,
    val_dataloaders = valid_dl
)

# torch.save(model.state_dict(), 'model/cpu_test.pth')

