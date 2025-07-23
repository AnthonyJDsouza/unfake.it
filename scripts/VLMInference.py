import os
import torch
from tqdm import tqdm
import pandas as pd
from transformers import AutoProcessor, AutoModelForImageTextToText

# work_dir = str(input('Enter working directory: '))
BASE_PATH = '/local/ajdsouza/dfdc/unfakeit-scraped-videos/videos'
BASE_DIR = '/local/ajdsouza/dfdc/unfakeit-scraped-videos'
MODEL = 'HuggingFaceTB/SmolVLM2-256M-Video-Instruct'

labels = pd.read_csv(os.path.join(BASE_PATH, 'metadata.csv'))
processor = AutoProcessor.from_pretrained(MODEL, token = False)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL,
    torch_dtype = torch.bfloat16,
    # _attn_implementation = 'flash_attention_2',
    token = False
).to('cuda')

#system = 'You are a helpful AI helping humans detect AI generated images using common sense reasoning. You look for logical inconsistencies in the video / image and hilight them'

system = 'You are a helpful AI, helping humans detect AI generated images and videos. Users provide you with a video and a high level description of why the image or video is fake or real and you have to be super specific about what makes it AI generated or real and give an enhanced description by incorporating your observations with the provided descriptions.'


long_descriptions = []
for idx, row in tqdm(labels.iterrows(), total = len(labels)):

    path = row['filename']
    desc = row['description']
    tqdm.write(f"Processing::: {path}")
    #print(BASE_DIR)
    #print(path)
    #print(os.path.join(BASE_DIR, path))
    messages = [
        #{
        #    'role': 'system',
        #    'content': [
        #        {'type': 'text', 'text' : system}
        #    ]
        #},
        {
            'role': 'user',
            'content': [
                {'type': 'video', 'path': os.path.join(BASE_DIR, path)},
                {'type': 'text', 'text':f"{system}. {desc}"}
            ]
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt = False,
        tokenize = True,
        return_dict = True,
        return_tensors = 'pt'
    ).to(model.device, dtype = torch.bfloat16)

    generated_ids = model.generate(
        **inputs,
        do_sample = False,
        max_new_tokens = 256
    )

    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens = True
    )
    long_descriptions.append(generated_texts[0])


labels['FineDescriptionsSmolVLM'] = long_descriptions

labels.to_csv(os.path.join(BASE_PATH, 'fine_labels_smolvlm.csv'), index = False)

# print(messages)
