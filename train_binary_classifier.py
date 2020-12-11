# Code for training the binary classifier used in the thesis.

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models
from tqdm import tqdm
from PIL import Image
import torch.optim as optim
from torchvision import transforms
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='PIL')

import datasets as d
import transforms as t

def main():
    parser = argparse.ArgumentParser(description='download some images')
    parser.add_argument('obj_instance', help='the query string')
    parser.add_argument('search_term', help='the number of images to download')
    args = parser.parse_args()
    OBJECT_INSTANCE = args.obj_instance
    SEARCH_TERM = args.search_term
    SS_TYPE = 'fast'

    print(OBJECT_INSTANCE)
    print(SEARCH_TERM)

    # For feature extractor to compute webly mean + dataset mean.
    model = models.resnet152(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    model.cuda()
    model.eval()

    ## Sort webly elements by how close they are in the embedded feature space to the average item.

    # Compute mean of entire webly dataset.
    webly_root = Path('/floyd/input/cleaned_webly_dataset')
    target_webly_dir = Path(webly_root / SEARCH_TERM.replace(' ', '_'))
    webly_dset = d.WeblyDataset(target_webly_dir, t.webly_transform)
    webly_loader = DataLoader(webly_dset, batch_size=196, num_workers=8)

    running_total = torch.zeros([2048])
    with torch.no_grad():
        for inputs in tqdm(webly_loader):
            inputs = inputs.cuda()
            outputs = model(inputs)
            running_total += outputs.sum(dim=0, keepdim=True).squeeze().cpu()

        mean = torch.div(running_total, len(webly_dset))

    # Compute distances to mean.
    webly_dset = d.WeblyDataset(target_webly_dir, t.webly_transform, dloader=False)
    mean = mean.cuda()
    distances = {}
    with torch.no_grad():
        for i, p in tqdm(webly_dset):
            i = i.cuda()
            o = model(i.unsqueeze(0))
            b = torch.dist(mean, o.squeeze())
            distances[p] = b.item()

    sorted_distances = {k:v for k,v in sorted(distances.items(), key=lambda x: x[1])}
    top_paths = list(sorted_distances.keys())[:10]

    top_ten = []
    for tp in top_paths:
        tp = Path(*tp.parts[len(webly_root.parts):])
        top_ten.append(str(tp))

    with open(f'/floyd/home/{OBJECT_INSTANCE}-ranking.json', 'w') as outfile:
        json.dump(top_ten, outfile)

    # Initialize datasets for finding dataset mean.
    webly_sorted_dset = d.SortedListDataset(list(sorted_distances.keys()), t.resize, threshold=0.7)
    coco_val_crop_dset = d.CocoCropDataset('/floyd/input/coco_2017_val_crops', t.resize)


    # Compute dataset mean
    datas = ConcatDataset([coco_val_crop_dset, webly_sorted_dset])
    loader = DataLoader(datas, batch_size=1000, num_workers=1)
    num_pixels = len(datas) * 128 * 128

    total_sum = [0, 0, 0]
    for batch in tqdm(loader):
        for bi in batch:
            r = bi[0]
            g = bi[1]
            b = bi[2]
            total_sum[0] += r.sum().item()
            total_sum[1] += g.sum().item()
            total_sum[2] += b.sum().item()

    dataset_mean = list(map(lambda t: t / num_pixels, total_sum))

    # compute dataset std.
    sum_of_squared_error = [0, 0, 0]
    for batch in tqdm(loader):
        for bi in batch:
            sum_of_squared_error[0] += ((bi[0] - dataset_mean[0]).pow(2)).sum().cpu()
            sum_of_squared_error[1] += ((bi[1] - dataset_mean[1]).pow(2)).sum().cpu()
            sum_of_squared_error[2] += ((bi[2] - dataset_mean[2]).pow(2)).sum().cpu()

    r_std = torch.sqrt(sum_of_squared_error[0] / num_pixels).item()
    g_std = torch.sqrt(sum_of_squared_error[1] / num_pixels).item()
    b_std = torch.sqrt(sum_of_squared_error[2] / num_pixels).item()

    dataset_std = [r_std, g_std, b_std]

    with open(f'/floyd/home/stats-{OBJECT_INSTANCE}.json', 'w') as outfile:
        json.dump({'mean': dataset_mean, 'std': dataset_std}, outfile)


    # Model that will be trained.
    model = models.resnet152(pretrained=True)
    model.fc = nn.Linear(in_features=2048, out_features=1, bias=True)
    model = model.cuda()
    model.train()

    # Initialize datasets for training.
    webly_sorted_dset = d.SortedListDataset(list(sorted_distances.keys()), t.get_webly_transform(dataset_mean, dataset_std), threshold=0.7, labels=True)
    coco_val_crop_dset = d.CocoCropDataset('/floyd/input/coco_2017_val_crops', t.get_coco_crop_transform(dataset_mean, dataset_std), labels=True)
    train_set = ConcatDataset([webly_sorted_dset, coco_val_crop_dset])

    PIN_MEM = torch.cuda.is_available()
    num_workers = 8
    batch_size = 128

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=PIN_MEM)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.8, nesterov=True)

    # criterion
    criterion = nn.BCEWithLogitsLoss()

    # Training
    NUM_EPOCHS = 10

    from torch.utils.tensorboard import SummaryWriter
    # Set up tensorboard writer
    writer = SummaryWriter(f'runs/{OBJECT_INSTANCE}')
    for e in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        for data in tqdm(train_loader):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs).flatten()
            loss = criterion(outputs, labels.float())

            train_labels.extend([l.item() for l in labels])
            train_preds.extend([o.item() for o in outputs.sigmoid()])

            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        accuracy = accuracy_score(train_labels, np.array([1 if p > 0.5 else 0 for p in train_preds]))
        writer.add_pr_curve('train_pr', np.array(train_labels), np.array(train_preds), e)
        writer.add_scalar('train_loss', train_loss / len(train_loader.dataset), e)
        writer.add_scalar('train_accuracy', accuracy, e)

    weights = f'/floyd/home/{OBJECT_INSTANCE}.pth'
    torch.save(model.state_dict(), Path(weights))


    # Compute result via trained model.
    arid_root = Path('/floyd/input/arid_floyd')
    ss_root = Path('/floyd/input/selective_search_arid_fast')

    with open('/floyd/input/img_mapping/img_mapping.json') as json_file:
        img_mapping = json.load(json_file)[OBJECT_INSTANCE]

    with torch.no_grad():
        model.eval()
        # Loop through all images that we know contain the target object.
        for wp_title, img_ids in img_mapping.items():
            print(f'{wp_title}-{"|".join(img_ids)}')

            # Load selective search region proposals.
            with open(Path(ss_root / f'ss-{SS_TYPE}-{wp_title}.json')) as ss_file:
                ss_data = json.load(ss_file)

            savefile = Path(f'/floyd/home/{OBJECT_INSTANCE}-{wp_title}.json')

            if not savefile.exists():
                results = {}
                for img_id in tqdm(img_ids):
                    results[img_id] = []
                    img_path = Path(arid_root / wp_title / f'{img_id}.png')
                    ss_boxes = ss_data[img_id]['boxes']
                    ss_dset = d.SelectiveSearchProposalsDataset(img_path, ss_boxes, t.get_ss_transform(dataset_mean, dataset_std))
                    ss_loader = DataLoader(ss_dset, batch_size=900, num_workers=8)
                    for i in ss_loader:
                        i = i.cuda()
                        o = model(i)
                        results[img_id].extend([sc.item() for sc in o.sigmoid()])

                with open(savefile, 'w') as outfile:
                    json.dump(results, outfile)
            else:
                print('already done')


if __name__ == '__main__':
    main()
