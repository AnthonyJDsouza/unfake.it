from src.model import ImageClassifier
import torch
import torch.nn as nn
import lightning as L
from peft import PeftModel

model_path = ''

model = ImageClassifier.load_from_checkpoint(model_path)
peft_model = model.lora_model

peft_model.save_pretrained('/scratch/ajdsouza/models/dfdc/vit_lora_adapter')