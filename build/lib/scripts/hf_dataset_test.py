from datasets import load_from_disk
from huggingface_hub import login

login(str(input("Enter huggingface token: ")))

ds = load_from_disk('data/hf')

# iter_ds = ds.to_iterable_dataset()

# for ex in iter_ds:
#     print(ex)


ds.push_to_hub('tororoin/demo_img')