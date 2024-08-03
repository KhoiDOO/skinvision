from sklearn.model_selection import StratifiedShuffleSplit
from pytorch_balanced_sampler.sampler import SamplerFactory
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from .utils import resize_with_pad, data_transforms
from glob import glob

import pytorch_lightning as pl
import imageio.v3 as iio
import argparse
import pandas as pd
import numpy as np
import math
import cv2
import os


class CropDataset(Dataset):
    def __init__(self, root, names, targets, mid, transform) -> None:
        super().__init__()

        self.root = root
        self.names = names
        self.targets = targets
        self.mid = mid
        self.transform = transform
    
    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        print(index)
        name = self.names[index]
        img_path = os.path.join(self.root, f'{name}.png')

        img = iio.imread(img_path)
        size = img.shape[0]

        if size < self.mid:
            resized_img = resize_with_pad(img, (self.mid, self.mid), padding_color=(0, 0, 0))
        elif size > self.mid:
            resized_img = cv2.resize(img, (self.mid, self.mid), interpolation=cv2.INTER_LINEAR)
        else:
            resized_img = img
        
        if self.transform:
            transformed_img = self.transform(image=resized_img)["image"]
        else:
            transformed_img = resized_img
        
        return {
            'image' : transformed_img,
            'target' : self.targets[index]
        }


class DataModule(pl.LightningDataModule):

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.root = cfg.data.root
        if self.cfg.data.crop:
            self.ds_interface = CropDataset
        else:
            raise ValueError(f'Only support self.cfg.data.crop = True')
        self.trdf = pd.read_csv(self.cfg.data.trdf_path)
        self.tsdf = pd.read_csv(self.cfg.data.tsdf_path)

        self.names = self.trdf['isic_id'].values
        self.label = self.trdf['target'].values

        # self.ts_names = self.tsdf['isic_id'].values.tolist()
        # self.ts_label = self.tsdf['target'].values.tolist()

        self.sss = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=self.cfg.seed)
        splits = self.sss.split(self.names, self.label)
        for self.tr_idxs, self.vl_idxs in splits:
            pass
        self.tr_names, self.tr_label = self.names[self.tr_idxs].tolist(), self.label[self.tr_idxs].tolist()
        self.vl_names, self.vl_label = self.names[self.vl_idxs].tolist(), self.label[self.vl_idxs].tolist()

        self.tr_idx_0 = np.argwhere(np.array(self.tr_label) == 0).flatten().tolist()
        self.tr_idx_1 = np.argwhere(np.array(self.tr_label) == 1).flatten().tolist()

        self.vl_idx_0 = np.argwhere(np.array(self.vl_label) == 0).flatten().tolist()
        self.vl_idx_1 = np.argwhere(np.array(self.vl_label) == 1).flatten().tolist()

        self.batch_sampler = WeightedRandomSampler(weights=(0.8, 0.2), num_samples=len(self.tr_names))

        assert len(self.tr_names) == len(self.tr_label)
        assert len(self.vl_names) == len(self.vl_label)

        print(f'[INFO]: Found {len(self.tr_names)} training samples, {len(self.vl_names)} validation samples')
        print(f'[INFO]: {len(self.tr_idx_0)} 0s, {len(self.tr_idx_1)} 1s in train')
        print(f'[INFO]: {len(self.vl_idx_0)} 0s, {len(self.vl_idx_1)} 1s in valid')

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = self.ds_interface(
                root=self.root, 
                names=self.tr_names, 
                targets=self.tr_label, 
                mid=self.cfg.data.mid,
                transform=data_transforms['train'] if self.cfg.data.aug else data_transforms['valid'])
            
        if stage in [None, "fit", "validate"]:
            self.val_dataset = self.ds_interface(
                root=self.root, 
                names=self.vl_names, 
                targets=self.vl_label, 
                mid=self.cfg.data.mid,
                transform=data_transforms['valid'])
            
        # if stage in [None, "test", "predict"]:
        #     self.test_dataset = self.ds_interface(
        #         root=self.root, 
        #         names=self.ts_names, 
        #         targets=self.ts_label, 
        #         mid=self.cfg.data.mid,
        #         transform=data_transforms['valid'])

    def general_loader(self, dataset, batch_size) -> DataLoader:
        return DataLoader(dataset, num_workers=self.cfg.data.num_workers, batch_size=batch_size)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, 
            num_workers=self.cfg.data.num_workers, 
            batch_size=self.cfg.data.batch_size, 
            shuffle=self.cfg.data.shuffle,
            sampler=self.batch_sampler)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, 
            num_workers=self.cfg.data.num_workers, 
            batch_size=self.cfg.data.batch_size)

    # def test_dataloader(self) -> DataLoader:
    #     return DataLoader(
    #         self.test_dataset, 
    #         num_workers=self.cfg.data.num_workers, 
    #         batch_size=self.cfg.data.batch_size)