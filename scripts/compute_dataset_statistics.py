import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import numpy as np

from config import PreprocessedSatelliteDataset
from runner import Runner

from tqdm.auto import tqdm

def compute_mean_std(dataset, split):
    rootPath = Runner.get_dataset_root(dataset_name=dataset)
    if split == 'train':
        dataframe = os.path.join(rootPath, 'train.csv')
    elif split == 'val':
        dataframe = os.path.join(rootPath, 'val.csv')
    else:
        raise ValueError("Invalid split value. Expected 'train' or 'val'.")

    # Just basic ToTensor conversion
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = PreprocessedSatelliteDataset(
        data_path=rootPath,
        dataframe=dataframe,
        image_transforms=train_transforms,
        use_weighted_sampler=None,
        use_memmap=True,
        remove_corrupt=False
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    num_workers = 0  # Windows-safe

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    
    mean = 0.
    std = 0.
    nb_samples = 0.

    with torch.no_grad():
        for data in tqdm(dataloader):
            data, _ = data
            data = data.to(device=device, non_blocking=True)
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    return mean, std

def get_normalization_transform(mean, std):
    """Return a torchvision transform that normalizes with computed mean/std"""
    return transforms.Normalize(mean=mean.tolist(), std=std.tolist())

if __name__ == '__main__':
    # Parameters
    dataset = 'ai4forest_camera'
    split = 'train'

    # Compute and print the mean and std
    mean, std = compute_mean_std(dataset=dataset, split=split)
    print(f'Mean: {mean}')
    print(f'Std: {std}')

    # Save the statistics
    dump_path = os.path.join(os.getcwd(), f'{dataset}_{split}_mean_std.txt')
    with open(dump_path, 'w') as f:
        f.write(f'Mean: {mean.tolist()}\n')
        f.write(f'Std: {std.tolist()}\n')

    # Show example transform you can now use
    normalization_transform = get_normalization_transform(mean, std)
    print("\n✅ You can now add the following normalization step to your training transforms:\n")
    print(f"transforms.Normalize(mean={mean.tolist()}, std={std.tolist()})")
