import torch.distributed as dist
import pandas as pd
import cv2
import torch
import os
import numpy as np
import pickle
from pathlib import Path
from natsort import natsorted
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision.datasets.utils import download_url

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def multi_class_split_dataset(method):
    def wrapper(self, data_folder, train=True):
        metadata, n_cls = method(self, data_folder, train)
        if train:
            self.log.info(f'Making a train-val split ({self.val_amount})')
            train_df, val_df = train_test_split(
                metadata,
                test_size=self.val_amount,
                shuffle=True,
                random_state=self.seed,
                stratify=metadata.target,
            )
            return train_df, val_df, None, n_cls
        return None, None, metadata, n_cls
    return wrapper


def wrap_channels(imgs):
    if len(imgs.shape) == 3:
        return np.stack((imgs[:, :, :, None], imgs[:, :, :, None], imgs[:, :, :, None]), axis=3).squeeze()
    return imgs


def make_image_target_df(imgs, labels):
    list_rows = [
        {"data": imgs[i, :, :, :], "target": labels[i]}
        for i in range(len(labels))
    ]
    return pd.DataFrame(list_rows)


class DataProvider(object):
    allowed_datasets = ["brats", "lgg"]
    in_memory_datasets = ["brats", "lgg"]

    def __init__(self, cfg, logger, rank=0, distributed=False):
        if cfg.data.dataset not in self.allowed_datasets:
            raise ValueError(f"Unsupported dataset {cfg.data.dataset}")

        self.cfg = cfg
        self.val_amount = cfg.data.val_amount
        self.dataset = cfg.data.dataset
        self.seed = cfg.seed
        self.data_folder = cfg.data.data_dir
        # os.makedirs(self.data_folder, exist_ok=True)
        self.metadata = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.transforms = None
        self.rank = rank
        self.log = logger
        self.distributed = distributed

    def init_brats(self, *args, **kwargs):
        fullname = os.path.join(self.data_folder, self.cfg.data.pkl_filename)
        if os.path.isfile(fullname):
            with open(fullname, 'rb') as f:
                data = pickle.load(f)
            train_df = data[self.cfg.data.fold_index][0]
            test_df = data[self.cfg.data.fold_index][1]
        else:
            raise ValueError(f'{fullname} not found.')

        num_train_samples = num_valid_samples = -1
        if self.cfg.data.valid_samples is not None:
            num_valid_samples = self.cfg.data.valid_samples
        if self.cfg.data.training_samples is not None:
            if self.cfg.data.valid_samples is None:
                num_valid_samples = self.cfg.data.training_samples
            num_train_samples = self.cfg.data.training_samples

        train_df = train_df if num_train_samples == -1 else train_df.iloc[:num_train_samples]
        test_df = test_df if num_valid_samples == -1 else test_df.iloc[:num_valid_samples]
        return train_df, test_df, None, None

    def init_lgg(self, *args, **kwargs):
        fullname = os.path.join(self.data_folder, self.cfg.data.pkl_filename)
        if os.path.isfile(fullname):
            with open(fullname, 'rb') as f:
                data = pickle.load(f)
            train_df = data[self.cfg.data.fold_index][0]
            test_df = data[self.cfg.data.fold_index][1]
        else:
            raise ValueError(f'{fullname} not found.')

        num_train_samples = num_valid_samples = -1
        if self.cfg.data.valid_samples is not None:
            num_valid_samples = self.cfg.data.valid_samples
        if self.cfg.data.training_samples is not None:
            if self.cfg.data.valid_samples is None:
                num_valid_samples = self.cfg.data.training_samples
            num_train_samples = self.cfg.data.training_samples

        train_df = train_df if num_train_samples == -1 else train_df.iloc[:num_train_samples]
        test_df = test_df if num_valid_samples == -1 else test_df.iloc[:num_valid_samples]
        return train_df, test_df, None, None

    def init_splits(self):
        if self.dataset == 'brats':
            if self.rank == 0:
                self.log.info(f'Getting {self.dataset} from {self.data_folder}')
            train_df, val_df, _, _ = getattr(self, f"init_{self.dataset}")()
        elif self.dataset == 'lgg':
            if self.rank == 0:
                self.log.info(f'Getting {self.dataset} from {self.data_folder}')
            train_df, val_df, _, _ = getattr(self, f"init_{self.dataset}")()
        else:
            raise ValueError(f'Not support dataset {self.dataset}')

        if self.rank == 0:
            self.log.info(
                f"The split has been loaded from disk by all processes")
        if self.distributed:
            dist.barrier()

        return train_df, val_df
