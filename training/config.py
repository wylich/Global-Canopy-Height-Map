from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import glob
import os
import torch
import numpy as np
import pandas as pd
import pdb
from torch.utils.data.dataloader import default_collate
import sys

means = {
    'ai4forest_camera': (7357.0898, 8271.5566, 5203.8892, 4658.9609, 742.1922, 1059.8594,
            1323.3116, 1651.3381, 2184.9229, 2404.3860, 2462.6221, 2561.5564,
            2586.2090, 2028.3472),    # Not the true values, change for your dataset
}

stds = {
    'ai4forest_camera': (844.3572, 896.9669, 927.5775, 869.1962, 177.6913, 212.9341, 
            281.3696, 280.8196, 335.8968, 373.0046, 396.8661, 388.6885, 
            370.9695, 340.5086),  # Not the true values, change for your dataset
}

percentiles = {
    'ai4forest_camera': {
        1: (-8840.0, -8507.0, -13840.0, -13691.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        2: (-7893.0, -7532.0, -12816.0, -12815.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        5: (-6485.0, -6197.0, -11442.0, -11590.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        95: (25781.0, 25389.0, 21531.0, 22175.0, 16200.0, 16072.0, 15898.0, 15795.0, 15783.0, 15759.0, 15743.0, 15682.0, 11848.0, 12923.0),
        98: (27269.0, 26922.0, 23100.0, 23874.0, 16432.0, 16159.0, 15935.0, 15819.0, 15804.0, 15780.0, 15763.0, 15713.0, 13045.0, 13989.0),
        99: (28415.0, 28071.0, 24452.0, 25078.0, 16544.0, 16231.0, 16072.0, 15891.0, 15822.0, 15827.0, 15960.0, 15747.0, 13677.0, 14965.0),
    }  # Not the true values, change for your dataset
}

class FixValDataset(Dataset):
    """
    Dataset class to load the fixval dataset.
    """
    def __init__(self, data_path, dataframe, image_transforms=None):
        self.data_path = data_path
        self.df = pd.read_csv(dataframe, index_col=False)
        self.files = list(self.df["path"].apply(lambda x: os.path.join(data_path, "data", x)))
        self.image_transforms = image_transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index].replace(r"'", "")
        fileName = file[file.rfind('data_')+5: file.rfind('.npz')]
        data = np.load(file)

        image = data["data"].astype(np.float32)
        # Move the channel axis to the last position (required for torchvision transforms)
        image = np.moveaxis(image, 0, -1)
        if self.image_transforms:
            image = self.image_transforms(image)

        return image, fileName

class PreprocessedSatelliteDataset(Dataset):
    """
    Dataset class for preprocessed satellite imagery.
    """

    def __init__(self, data_path, dataframe=None, image_transforms=None, label_transforms=None, joint_transforms=None, use_weighted_sampler=False,
                  use_weighting_quantile=None, use_memmap=False, remove_corrupt=True, load_labels=True, patch_size=512):
        self.use_memmap = use_memmap
        self.patch_size = patch_size
        self.load_labels = load_labels  # If False, we only load the images and not the labels
        df = pd.read_csv(dataframe)
        
        if remove_corrupt:
            old_len = len(df)
            #df = df[df["missing_s2_flag"] == False] # Use only the rows that are not corrupt, i.e. those where df["missing_s2_flag"] == False

            # Use only the rows that are not corrupt, i.e. those where df["has_corrupt_s2_channel_flag"] == False
            df = df[df["has_corrupt_s2_channel_flag"] == False]
            sys.stdout.write(f"Removed {old_len - len(df)} corrupt rows.\n")

        self.files = list(df["path"].apply(lambda x: os.path.join(data_path, "data", x))) # added "data" to the path <- adds individual files (images) to list

        if use_weighted_sampler not in [False, None]:
            assert use_weighted_sampler in ['g5', 'g10', 'g15', 'g20', 'g25', 'g30']
            weighting_quantile = use_weighting_quantile
            assert weighting_quantile in [None, 'None'] or int(weighting_quantile) == weighting_quantile, "weighting_quantile must be an integer."
            if weighting_quantile in [None, 'None']:
                self.weights = (df[use_weighted_sampler] / df["totals"]).values.clip(0., 1.)
            else:
                # We do not clip between 0 and 1, but rather between the weighting_quantile and 1.
                weighting_quantile = float(weighting_quantile)
                self.weights = (df[use_weighted_sampler] / df["totals"]).values

                # Compute the quantiles, ignoring nan values and zero values
                tmp_weights = self.weights.copy()
                tmp_weights[np.isnan(tmp_weights)] = 0.
                tmp_weights = tmp_weights[tmp_weights > 0.]

                quantile_min = np.nanquantile(tmp_weights, weighting_quantile / 100)
                sys.stdout.write(f"Computed weighting {weighting_quantile}-quantile-lower bound: {quantile_min}.\n")

                # Clip the weights
                self.weights = self.weights.clip(quantile_min, 1.0)

            # Set the nan values to 0.
            self.weights[np.isnan(self.weights)] = 0.

        else:
            self.weights = None
        self.image_transforms, self.label_transforms, self.joint_transforms = image_transforms, label_transforms, joint_transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if self.use_memmap:
            item = self.getitem_memmap(index)
        else:
            item = self.getitem_classic(index)

        return item

    def getitem_memmap(self, index):
        file = self.files[index]
        with np.load(file, mmap_mode='r') as npz_file:
            image = npz_file['data'].astype(np.float32)
            # Move the channel axis to the last position (required for torchvision transforms)
            image = np.moveaxis(image, 0, -1)
            if self.image_transforms:
                image = self.image_transforms(image)
            if self.load_labels:
                label = npz_file['labels'].astype(np.float32)

                # Process label
                label = label[:3]  # Everything after index/granule 3 is irrelevant
                label = label / 100  # Convert from cm to m
                label = np.moveaxis(label, 0, -1)

                if self.label_transforms:
                    label = self.label_transforms(label)
                if self.joint_transforms:
                    image, label = self.joint_transforms(image, label)
                return image, label

        return image

    def getitem_classic(self, index):
        file = self.files[index]
        data = np.load(file)

        image = data["data"].astype(np.float32)
        # Move the channel axis to the last position (required for torchvision transforms)
        image = np.moveaxis(image, 0, -1)[:self.patch_size,:self.patch_size]
        if self.image_transforms:
            image = self.image_transforms(image)
        if self.load_labels:
            label = data["labels"].astype(np.float32)

            # Process label
            label = label[:3]  # Everything after index 3 is irrelevant
            label = label[:,:self.patch_size, :self.patch_size]
            label = label / 100  # Convert from cm to m
            label = np.moveaxis(label, 0, -1)

            if self.label_transforms:
                label = self.label_transforms(label)
            if self.joint_transforms:
                image, label = self.joint_transforms(image, label)
            return image, label

        return image
