from torchvision.transforms import v2

from transformers import AutoImageProcessor
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
import torch

LABEL_MAP = {'FAKE': 0, 'REAL': 1}

class PadToSquare:
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            _, h, w = img.shape
        else:
            w, h = img.size  # PIL image

        size = max(h, w)
        pad_h = (size - h) // 2
        pad_w = (size - w) // 2
        padding = (pad_w, pad_h, size - w - pad_w, size - h - pad_h)

        if isinstance(img, torch.Tensor):
            return v2.functional.pad(img, padding, fill=self.fill)
        else:
            return v2.functional.pad(v2.ToImage()(img), padding, fill=self.fill)
        

def prepare_dataloaders(train_dataset, val_dataset, model_name, batch_size = 64):
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast = True, token=False)

    gpu_transforms = v2.Compose([
        # PadToSquare(fill = 0),
        v2.Resize(224, antialias=True),
        PadToSquare(fill = 0),
        v2.ToDtype(torch.uint8, scale = True),
        v2.Normalize(mean = processor.image_mean, std = processor.image_std),
    ])

    def transform_example(example):
        image = v2.ToImage()(example['image'])
        # label_id = LABEL_MAP[example['label']]
        return {'pixel_values': image, 'label': example['label']}
    
    train_ds = train_dataset.map(transform_example)
    val_ds = val_dataset.map(transform_example)

    def collate_fn(batch):
        images = torch.stack([item['pixel_values'] for item in batch]).to('cuda')
        labels = torch.stack([item['label'] for item in batch]).to('cuda')
        images = gpu_transforms(images)
        return {'pixel_values': images, 'labels': labels}
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, collate_fn = collate_fn, shuffle = True)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, collate_fn = collate_fn)
    num_classes = len(train_dataset.features['label'].names)
    return train_dataloader, val_dataloader, num_classes
