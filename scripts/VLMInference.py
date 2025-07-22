import torch
import pandas as pd
from transformers import AutoProcessor, AutoModelForImageTextToText

work_dir = str(input('Enter working directory: '))
BASE_PATH = '/scratch/ajdsouza/dfdc/dataset/'
MODEL = 'HuggingFaceTB/SmolVLM2-256M-Video-Instruct'

labels = pd.read_csv('')
processor = AutoProcessor.from_pretrained(MODEL)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL,
    torch_dtype = torch.bfloat16,
    _attn_implementation = 'flash_attention_2'
).to('cuda')

#system = 'You are a helpful AI helping humans detect AI generated images using common sense reasoning. You look for logical inconsistencies in the video / image and hilight them'

system = 'You are a helpful AI, helping humans detect AI generated images and videos. Users provide you with the video and a high level description of why the image is fake or real and you have to be super specific about what makes the video AI generated and give an enhanced description by incorporating your observations..'


long_descriptions = []
for idx, row in labels.iterrows():
    path = row['path']
    desc = row['description']

    messages = [
        {
            'role': 'system',
            'content': [
                {'type': 'text', 'text' : system}
            ]
        },
        {
            'role': 'user',
            'content': [
                {'type': 'video', 'path': f"{path}"},
                {'type': 'text', 'text':f"{desc}"}
            ]
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt = True,
        tokenize = True,
        return_dict = True,
        return_tensors = 'pt'
    ).to(model.device, dtype = torch.bfloat16)

    generated_ids = model.generate(
        **inputs,
        do_sample = False,
        max_new_tokens = 512
    )

    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens = True
    )
    long_descriptions.append(generated_texts[0])


labels['FineDescriptions'] = long_descriptions

labels.to_csv('f{work_dir}/fine_labels.csv', index = False)