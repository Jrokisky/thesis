# Various Torch datasets used in thesis code.

from torch.utils.data import Dataset
import random

from PIL import Image

from pathlib import Path
import time


class SelectiveSearchProposalsDataset(Dataset):
    """Create a dataset of the given selective search proposals.
    """
    def __init__(self, img_path, proposals, transform):
        self.proposals = proposals
        self.transform = transform
        self.img_path = img_path
        self.img = Image.open(img_path)
        
        # Hacky fix to force image to load in the init so we don't get error in get item.
        test = self.img.crop((0, 0, 10, 10))



    def __len__(self):
        return len(self.proposals)

    
    def __getitem__(self, idx):
        x1, y1, x2, y2 = self.proposals[idx]
        proposal = self.img.crop((x1, y1, x2, y2))
        return self.transform(proposal)

    
class WeblyDataset(Dataset):

    def __init__(self, search_root, transform, dloader=True):
        self.search_root = search_root
        self.transform = transform
        self.items = []
        self.dloader = dloader
        
        for img in self.search_root.iterdir():  
            if img.stem != '.floyddata':
                self.items.append(img)

    
    def __len__(self):
        return len(self.items)

    
    def __getitem__(self, idx):
        path = self.items[idx]
        img = Image.open(path)
        img = img.convert('RGB')
        sample = self.transform(img)
        if self.dloader:
            return sample
        else:
            return sample, path

        
class CocoCropDataset(Dataset):
    """Dataset for working with coco crops.
    """
    def __init__(self, crop_root, transform, labels=False):
        self.crop_root = Path(crop_root)
        self.items = []
        self.transform = transform
        self.labels = labels

        for cls in self.crop_root.iterdir():
            cls_imgs = {}
            for img in cls.iterdir():
                cls_imgs[img] = img.stat().st_size
            
            # Get the top 25 largest coco crops.
            cls_imgs = {k: v for k,v in sorted(cls_imgs.items(), key=lambda x: x[1], reverse=True)}
            cls_imgs = list(cls_imgs.keys())[:25]
            cls_imgs = zip(cls_imgs, [0]*len(cls_imgs))
            self.items.extend(cls_imgs)
    
                
    def __len__(self):
        return len(self.items)

    
    def __getitem__(self, idx):
        path, lbl = self.items[idx]
        img = Image.open(path)
        img = img.convert('RGB')
        sample = self.transform(img)
        if self.labels:
            return sample, lbl  
        else:
            return sample

    
class SortedListDataset(Dataset):

    def __init__(self, items, transform, threshold=None, labels=False):
        self.transform = transform
        self.threshold = threshold
        if self.threshold is not None:
            idx = int(len(items)*threshold)
            items = items[:idx]
        self.items = items
        self.labels = labels
        
    
    def __len__(self):
        return len(self.items)

    
    def __getitem__(self, idx):
        path = self.items[idx]
        img = Image.open(path)
        img = img.convert('RGB')
        sample = self.transform(img)
        if self.labels:
            return sample, 1
        else:
            return sample
