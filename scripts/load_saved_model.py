from src.model import ImageClassifier
import torch
import torch.nn as nn
import lightning as L
from peft import PeftModel
from dotenv import load_dotenv
<<<<<<< HEAD
import gc, os
import psutil
=======
import gc
import psutil, os
>>>>>>> 0b1dde7b6c8c2a1a6b19d7a0ee39fa5ac27d8def
from transformers import AutoFeatureExtractor
from huggingface_hub import login
load_dotenv()

token = os.getenv("HUGGINGFACE_HUB")
base_path = 'ViT-AI-Media-Detection/7s7az2wv/checkpoints/'
model_name = 'vit-best-loss-v1.ckpt'
model = ImageClassifier.load_from_checkpoint(os.path.join(base_path, model_name),
    model_name = "google/vit-base-patch32-224-in21k",
    num_labels = 1
        )

<<<<<<< HEAD
peft_model = model.lora_model
=======
model = ImageClassifier.load_from_checkpoint(model_path)
model = model.load_state_dict(model_path, strict=False)
#print(type(peft_model))
>>>>>>> 0b1dde7b6c8c2a1a6b19d7a0ee39fa5ac27d8def

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")

gc.collect()
print(f"RAM used: {psutil.Process().memory_info().rss / 1e9:.2f} GB")

#ae = AutoFeatureExtractor.from_pretrained("google/vit-base-patch32-224-in21k")
#dummy_input = ae(images=[torch.rand(3, 224, 224)], return_tensors="pt")

<<<<<<< HEAD
#with torch.no_grad():
#    outputs = peft_model(**dummy_input)
#print(outputs.last_hidden_state.shape)

peft_model.save_pretrained('/scratch/ajdsouza/models/dfdc/vit_lora_adapter')

login(token = token)
peft_model.push_to_hub('ajdsouza/vit-AI-vs-real-lora-ft')
=======
with torch.no_grad():
    outputs = model(**dummy_input)
print(outputs.last_hidden_state.shape)

#peft_model.save_pretrained('/scratch/ajdsouza/models/dfdc/vit_lora_adapter')
>>>>>>> 0b1dde7b6c8c2a1a6b19d7a0ee39fa5ac27d8def
