import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification
from peft import PeftModel
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import wandb
import io

# -------- Helper Functions -------- #

def get_attention_rollout(model, img_tensor, apply_residual=True):
    """
    Computes attention rollout for a ViT model (Hugging Face + PEFT).
    """
    attentions = []

    def hook_fn(module, input, output):
        # output: (B, heads, tokens, tokens)
        attentions.append(output.detach())

    hooks = []
    for blk in model.base_model.model.encoder.layer:
        hooks.append(blk.attention.attention_dropout.register_forward_hook(hook_fn))

    with torch.no_grad():
        _ = model(pixel_values=img_tensor)

    for h in hooks:
        h.remove()

    rollout = None
    for attn in attentions:
        attn_heads = attn[0].mean(0)  # mean over heads
        if apply_residual:
            attn_heads += torch.eye(attn_heads.size(0)).to(attn_heads.device)
        attn_heads /= attn_heads.sum(dim=-1, keepdim=True)
        if rollout is None:
            rollout = attn_heads
        else:
            rollout = attn_heads @ rollout

    cls_attn = rollout[0, 1:]
    return cls_attn

def show_attention_on_image(img, attn_map, patch_size, wandb_log=False, wandb_tag="attention_rollout"):
    h, w = img.size[1], img.size[0]
    num_patches = attn_map.shape[0]
    grid_size = int(num_patches ** 0.5)
    attn_map = attn_map.reshape(grid_size, grid_size).unsqueeze(0).unsqueeze(0)
    attn_map = F.interpolate(attn_map, size=(h, w), mode='bilinear', align_corners=False)
    attn_map = attn_map.squeeze().cpu().numpy()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.imshow(attn_map, cmap='jet', alpha=0.5)
    ax.axis('off')
    ax.set_title("Attention Rollout")

    if wandb_log:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        wandb.log({wandb_tag: wandb.Image(buf, caption=wandb_tag)})
        buf.close()

    plt.show()

# -------- Main Function -------- #

def main(image_path, peft_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Init Weights & Biases
    wandb.init(project="vit-lora-attention-rollout")

    model_name = 'google/vit-base-patch16-224'
    base_model = AutoModelForImageClassification.from_pretrained(model_name)
    model = PeftModel.from_pretrained(base_model, peft_path)
    model.eval().to(device)
    processor = AutoImageProcessor.from_pretrained(model_name)

    img = Image.open(image_path).convert('RGB')
    inputs = processor(images=img, return_tensors="pt")
    img_tensor = inputs['pixel_values'].to(device)

    cls_attn = get_attention_rollout(model, img_tensor)

    show_attention_on_image(img, cls_attn, patch_size=16, wandb_log=True)

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--peft_path', type=str, required=True, help='Path to PEFT (LoRA) checkpoint')
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image {args.image} not found")
    if not os.path.exists(args.peft_path):
        raise FileNotFoundError(f"PEFT path {args.peft_path} not found")

    main(args.image, args.peft_path)
