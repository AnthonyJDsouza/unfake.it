from torchvision.transforms import v2
import cv2
import numpy as np
import pandas as pd
from transformers import AutoImageProcessor
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset, DataLoader
import torch
import argparse
# import albumentations as A


LABEL_MAP = {'FAKE': 0, 'REAL': 1}

class PadToSquare:
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img):
        _, h, w = img.shape
        size = max(h, w)
        pad_v = int((size - h) / 2)
        pad_h = int((size - w) / 2)
        padding = (pad_h, pad_v, pad_h, pad_v)
        return v2.functional.pad(img, padding, fill=self.fill, padding_mode='constant')
        
        

# def prepare_dataloaders(train_dataset, val_dataset, model_name, batch_size = 64):
#     processor = AutoImageProcessor.from_pretrained(model_name, use_fast = True, token=False)

#     transforms = v2.Compose([
#         # PadToSquare(fill = 0),
#         v2.Resize(224, antialias=True),
#         PadToSquare(fill = 0),
#         v2.ToImage(),
#         v2.ToDtype(torch.uint8, scale = True),
#         v2.Normalize(mean = processor.image_mean, std = processor.image_std),
#     ])

#     def transform_example(example):
#         image = cv2.imread(example['image'])
#         if image is not None:
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image = image.transpose((2,0,1)) # HWC -> CHW, as torch expects
#         image = torch.from_numpy(image)
#         image = transforms(image)
#         # label_id = LABEL_MAP[example['label']]
#         return {'pixel_values': image, 'label': example['label']}
    
#     train_ds = train_dataset.map(transform_example)
#     val_ds = val_dataset.map(transform_example)

#     train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle = True)
#     val_dataloader = DataLoader(val_ds, batch_size=batch_size)
#     num_classes = len(train_dataset.features['label'].names)
#     return train_dataloader, val_dataloader, num_classes




class DFDCDataset(Dataset):
    def __init__(self, csv_path, transform = None):
        self.labels_csv = pd.read_csv(csv_path)
        # self.image_dir = image_dir
        self.transform = transform
        self.label2idx = {'FAKE':0, 'REAL':1}
        

    def __len__(self):
        return len(self.labels_csv)

    def __getitem__(self, idx):
        img_path = self.labels_csv.iloc[idx, 1]
        label = LABEL_MAP[self.labels_csv.iloc[idx, 2]]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2,0,1))
        image = torch.from_numpy(image)
        image = self.transform(image)
        return image, label
    
parser = argparse.ArgumentParser()
parser.add_argument('--csv_path', type=str, required=True)
parser.add_argument('--model-name', type=str, required=True)
args = parser.parse_args()
processor = AutoImageProcessor.from_pretrained(args.model_name, use_fast = True, token=False)
transforms = v2.Compose([
        PadToSquare(fill = 0),
        v2.Resize(224, antialias=True),
        # PadToSquare(fill = 0),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale = True),
        v2.Normalize(mean = processor.image_mean, std = processor.image_std),
    ])

training_data = DFDCDataset(csv_path = args.csv_path, transform = transforms)

training_dataloader = DataLoader(training_data, batch_size = 32, num_workers=4, shuffle = True)
image, label = next(iter(training_dataloader))
print(f"image shape: {image.shape}, label: {label.shape}")
# demo_transforms = v2.Compose([
#     PadToSquare(fill=0)
# ])

# p2s = PadToSquare(fill=0)
# r = v2.Resize(224)
# image = cv2.imread('data/video/cat.jpeg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image = image.transpose((2,0,1))
# image = torch.from_numpy(image)
# image_np = p2s(image).numpy()
# #image_np = (image_np * 255.).clip(0,255).astype(np.uint8)
# image_np = image_np.astype(np.uint8)
# # print(image_np)
# image_np = image_np.transpose((1,2,0))
# image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
# cv2.imwrite('data/video/cat_r.jpeg', image_np, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

