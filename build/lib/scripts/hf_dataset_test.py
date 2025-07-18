from datasets import load_from_disk, Features, ClassLabel, Image
import argparse

def string_to_classlabel(dataset_path, save_dir):
    d = load_from_disk(dataset_path=dataset_path)
    unique_labels = ['FAKE', 'REAL']
    new_features = d.features.copy()
    new_features['label'] = ClassLabel(names=unique_labels)
    new_features['image'] = Image(decode=False)
    d = d.cast(new_features)
    print("casted to ClassLabel and no PIL decode in Image")
    d.save_to_disk(save_dir)
    


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required = True)
parser.add_argument('--save_dir', type=str, required = True)
args = parser.parse_args()
string_to_classlabel(args.dataset, args.save_dir)

dset = load_from_disk(args.save_dir)
print(dset.features)
