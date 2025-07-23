from src.model import ImageClassifier
import torch
import torch.nn as nn
import lightning as L
from peft import PeftModel
from dotenv import load_dotenv
import gc, os
import psutil
from transformers import AutoFeatureExtractor
from huggingface_hub import login
load_dotenv()

token = os.getenv("HUGGINGFACE_HUB")
base_path = '/local/ajdsouza/dfdc/models/vit-dfdc'
model_name = 'vit-best-loss-v1.ckpt'
model = ImageClassifier.load_from_checkpoint(os.path.join(base_path, model_name))
tmodel = torch.load(os.path.join(base_path, model_name), map_location = 'cpu')
sd = tmodel['state_dict']
#model = ImageClassifier.load_from_checkpoint(model_path)
model = model.load_state_dict(sd, strict=False)
#print(type(peft_model))

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

gc.collect()
print(f"RAM used: {psutil.Process().memory_info().rss / 1e9:.2f} GB")

ae = AutoFeatureExtractor.from_pretrained("google/vit-base-patch32-224-in21k")
dummy_input = ae(images=[torch.rand(3, 224, 224)], return_tensors="pt")

with torch.no_grad():
    outputs = model(**dummy_input)
print(outputs.last_hidden_state.shape)

#peft_model.save_pretrained('/scratch/ajdsouza/models/dfdc/vit_lora_adapter')
