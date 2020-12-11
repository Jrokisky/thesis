# Compute various metrics using the trained binary classifier.

import argparse
import torch
from torch import nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset

from PIL import Image
from pathlib import Path
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve


import json

root = Path('/floyd/input/arid_crops')
stats_root = Path('/floyd/input/thesis')

class AridCropDataset(Dataset):

    def __init__(self, crop_root, object_name, transform):
        self.crop_root = Path(crop_root)
        self.transform = transform
        self.items = []
        pos = []
        neg = []
        for obj in self.crop_root.iterdir():
            if not obj.is_dir():
                continue
            obj_instance_id = obj.stem
            obj_cat = '_'.join(object_name.split('_')[:-1])

            if obj_instance_id.startswith(obj_cat) and obj_instance_id != object_name:
                print(obj_instance_id)
                continue
                
            label = 1 if obj_instance_id.startswith(object_name) else 0

            for img_path in obj.iterdir():
                if label == 1:
                    pos.append((img_path, label))
                else:
                    neg.append((img_path, label))

        self.items.extend(pos)
        self.items.extend(neg)


    def __len__(self):
        return len(self.items)


    def __getitem__(self, idx):
        path, lbl = self.items[idx]
        img = Image.open(path)
        img = img.convert('RGB')
        sample = self.transform(img)
        return sample, lbl


def get_tranform(mean, std):
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def main():
    parser = argparse.ArgumentParser(description='download some images')
    parser.add_argument('obj')
    args = parser.parse_args()
    obj = args.obj
 
    with open(Path(stats_root / f'stats-{obj}.json')) as outfile:
        stats = json.load(outfile)

    model = models.resnet152(pretrained=False)
    model.fc = nn.Linear(in_features=2048, out_features=1, bias=True)
    model.load_state_dict(torch.load(Path(stats_root / f'{obj}.pth')))
    model.cuda()
    dset = AridCropDataset(root, obj, get_tranform(stats['mean'], stats['std']))
    loader = DataLoader(dset, batch_size=600, shuffle=False, num_workers=8)
    
    with torch.no_grad():
        model.eval()
        labels = []
        preds = []
        
        for inputs, lbls in tqdm(loader):
            inputs = inputs.cuda()
            lbls = lbls.cuda()

            outputs = model(inputs)
            labels.extend([l.item() for l in lbls])
            preds.extend([o.item() for o in outputs.sigmoid()])

        arid_f1 = f1_score(labels, np.array([1 if p > 0.5 else 0 for p in preds]))
        roc_auc = roc_auc_score(labels, np.array(preds))
        precision, recall, thresholds = precision_recall_curve(labels, np.array(preds))
        acc = accuracy_score(labels, np.array([1 if p > 0.5 else 0 for p in preds]))
        results = {
            'f1': arid_f1,
            'roc_auc': roc_auc,
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'accuracy': acc,
        }
        with open(Path(f'/floyd/home/results-{obj}.json'), 'w') as outfile:
            json.dump(results, outfile)


if __name__ == '__main__':
    main()

