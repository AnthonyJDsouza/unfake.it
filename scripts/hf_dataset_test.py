from datasets import load_from_disk, Features, ClassLabel
import argparse

def string_to_classlabel(dataset_path):
    d = load_from_disk(dataset_path=dataset_path)
    unique_labels = list(set(d['train']['label']))
    new_features = d.features.copy()
    new_features['label'] = ClassLabel(names=unique_labels)
    
    d = d.cast(new_features)
    print("cast to ClassLabel")


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', typr=str, required = True)
args = parser.parse_args()
string_to_classlabel(args.dataset)

dset = load_from_disk(args.dataset)
print(dset.features)