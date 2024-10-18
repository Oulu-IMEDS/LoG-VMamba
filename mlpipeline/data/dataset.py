from typing import Any, Iterator

import cv2
import torch
import torch.nn.functional as functional
import solt
import pandas as pd
import numpy as np
import os
import math
import PIL
from pathlib import Path
from natsort import natsorted
from torch.utils.data import Dataset, Sampler
from typing import Optional
import nibabel as nib

import mlpipeline.utils.common as common
import monai.transforms as MonaiTrans


class DataFrameDataset(Dataset):
    def __init__(
        self, metadata: pd.DataFrame,
        data_key: str ='data',
        target_key: Optional[str] ='target',
        **kwargs,
    ):
        self.metadata = metadata
        self.data_key = data_key
        self.target_key = target_key

    def read_data(self, entry):
        return getattr(entry, self.data_key)

    def __getitem__(self, idx):
        entry = self.metadata.iloc[idx]
        # Getting the data from the dataframe
        res = self.read_data(entry)
        res['idx'] = idx
        return res

    def __len__(self):
        return self.metadata.shape[0]


class DataFrameImageDataset(DataFrameDataset):
    def __init__(
        self, metadata: pd.DataFrame, transforms: solt.Stream,
        data_key: str = 'data',
        target_key: Optional[str] = 'target', mean=None, std=None,
        **kwargs,
    ):
        super(DataFrameImageDataset, self).__init__(metadata, data_key, target_key)
        self.transforms = transforms
        self.mean = mean
        self.std = std

    def read_data(self, entry):
        img = getattr(entry, self.data_key)
        return self.transforms(img, mean=self.mean, std=self.std)['image']

def check_missing_data(x):
    return (isinstance(x, str) and "missing" in x) or not x or x != x


class BRATSDataset(DataFrameDataset):
    def __init__(
        self, root: str, metadata: pd.DataFrame, transforms: solt.Stream, stage,
        data_key: str = 'data',
        target_key: Optional[str] = 'target', mean=None, std=None,
        **kwargs,
    ):
        super().__init__(metadata, data_key, target_key)
        # self.transform_lib = 'monai'
        self.stage = stage
        self.transforms = transforms
        self.stats = {'mean': mean, 'std': std}
        self.root = root

        self.cfg = kwargs.get("config", None)
        self.train_crop_size = 128 if self.cfg is None else self.cfg.image_size
        if (self.cfg is None) and (self.cfg.image_depth is None):
            self.train_crop_depth = min(self.train_crop_size, 128)
        else:
            self.train_crop_depth = self.cfg.image_depth
        self.label_name = None if self.cfg is None else self.cfg.label_name
        self.use_geo = ("geo" in self.root) and os.path.isdir(self.root.geo) and False

        # H W D
        roi_size = (self.train_crop_size, self.train_crop_size, self.train_crop_depth)

        if self.use_geo and (self.label_name is not None):
            keys = ["image", "label", "dist"]
        else:
            keys = ["image", "label"]

        if (self.train_crop_size is not None) and (self.train_crop_size > 0):
            self.train_transforms = MonaiTrans.compose.Compose([
                # MonaiTrans.EnsureChannelFirstd(keys="image"),
                MonaiTrans.EnsureTyped(keys=keys),
                MonaiTrans.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                MonaiTrans.CropForegroundd(keys=keys, source_key="image", k_divisible=roi_size),
                MonaiTrans.RandSpatialCropd(keys=keys, roi_size=roi_size, random_size=False),
                MonaiTrans.RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                MonaiTrans.RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                MonaiTrans.RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
                MonaiTrans.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                MonaiTrans.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                MonaiTrans.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                MonaiTrans.ToTensord(keys=keys),
            ])
        else:
            self.train_transforms = MonaiTrans.compose.Compose([
                # MonaiTrans.EnsureChannelFirstd(keys="image"),
                MonaiTrans.EnsureTyped(keys=keys),
                MonaiTrans.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                MonaiTrans.RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                MonaiTrans.RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                MonaiTrans.RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
                MonaiTrans.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                MonaiTrans.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                MonaiTrans.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                MonaiTrans.ToTensord(keys=keys),
            ])
        self.val_transforms = MonaiTrans.compose.Compose([
            MonaiTrans.EnsureTyped(keys=keys),
            MonaiTrans.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            MonaiTrans.CropForegroundd(keys=keys, source_key="image", k_divisible=roi_size),
            MonaiTrans.RandSpatialCropd(keys=keys, roi_size=roi_size, random_size=False),
            MonaiTrans.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            MonaiTrans.ToTensord(keys=keys),
        ])

    def apply_transform(self, img_input, img_gt=None):
        return None, None

    def read_data(self, entry):
        sample = {}
        input_names = ['t1', 't2', 'flair', 't1ce']
        target_name = 'seg'

        fullname_gt = os.path.join(self.root.brats, entry[target_name])
        gts = nib.load(fullname_gt).get_fdata()
        images = []
        input_filenames = []

        for inp in input_names:
            fullname_input = os.path.join(self.root.brats, entry[inp])
            input_filenames.append(entry[inp])
            img = nib.load(fullname_input)

            if img is None:
                raise ValueError(f'Not found image file {fullname_input}.')

            images.append(img.get_fdata())

            if (self.label_name is None) or len(self.label_name) == 0:
                continue

        images = np.stack(images, axis=0)
        gts = gts.astype(float)
        transformed_dict = {'image': images, 'label': gts}

        if self.stage == 'train':
            obj_trans = self.train_transforms(transformed_dict)
        else:
            obj_trans = self.val_transforms(transformed_dict)

        if gts is None:
            raise ValueError(f'Not found image file {fullname_gt}.')

        sample = {}
        sample['input'] = obj_trans['image']
        sample['gt'] = obj_trans['label']
        sample['paths'] = [os.path.dirname(entry[target_name])]
        if self.use_geo and "dist" in obj_trans:
            sample['dist'] = obj_trans['dist'] - 1.0

        return sample


class LGGDataset(DataFrameDataset):
    def __init__(
        self, root: str, metadata: pd.DataFrame, transforms: solt.Stream, stage,
        data_key: str = 'data',
        target_key: Optional[str] = 'target', mean=None, std=None,
        **kwargs,
    ):
        super().__init__(metadata, data_key, target_key)
        # self.transform_lib = 'monai'
        self.stage = stage
        self.transforms = transforms
        self.stats = {'mean': mean, 'std': std}
        self.root = root

        self.cfg = kwargs.get("config", None)
        self.train_crop_size = 128 if self.cfg is None else self.cfg.image_size
        if (self.cfg is None) and (self.cfg.image_depth is None):
            self.train_crop_depth = min(self.train_crop_size, 128)
        else:
            self.train_crop_depth = self.cfg.image_depth
        self.label_name = None if self.cfg is None else self.cfg.label_name
        self.use_geo = ("geo" in self.root) and os.path.isdir(self.root.geo)

        # H W D
        roi_size = (self.train_crop_size, self.train_crop_size, self.train_crop_depth)

        if self.use_geo and (self.label_name is not None):
            keys = ["image", "label", "dist"]
        else:
            keys = ["image", "label"]

        if (self.train_crop_size is not None) and (self.train_crop_size > 0):
            self.train_transforms = MonaiTrans.compose.Compose([
                # MonaiTrans.EnsureChannelFirstd(keys="image"),
                MonaiTrans.EnsureTyped(keys=keys),
                MonaiTrans.CropForegroundd(keys=keys, source_key="image", k_divisible=roi_size),
                MonaiTrans.RandSpatialCropd(keys=keys, roi_size=roi_size, random_size=False),
                MonaiTrans.RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                MonaiTrans.RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                MonaiTrans.RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
                MonaiTrans.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                MonaiTrans.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                MonaiTrans.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                MonaiTrans.ToTensord(keys=keys),
            ])
        else:
            self.train_transforms = MonaiTrans.compose.Compose([
                # MonaiTrans.EnsureChannelFirstd(keys="image"),
                MonaiTrans.EnsureTyped(keys=keys),
                MonaiTrans.CropForegroundd(keys=keys, source_key="image", k_divisible=roi_size),
                MonaiTrans.RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
                MonaiTrans.RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                MonaiTrans.RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
                MonaiTrans.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
                MonaiTrans.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                MonaiTrans.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                MonaiTrans.ToTensord(keys=keys),
            ])
        self.val_transforms = MonaiTrans.compose.Compose([
            MonaiTrans.EnsureTyped(keys=keys),
            MonaiTrans.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            MonaiTrans.ToTensord(keys=keys)
        ])

    def apply_transform(self, img_input, img_gt=None):
        return None, None

    def read_data(self, entry):
        sample = {}
        input_names = ['flair_0', 'flair_1', 'flair_2']
        target_name = 'seg'

        fullname_gt = os.path.join(self.root.lgg, entry[target_name])
        gts = nib.load(fullname_gt).get_fdata()
        images = []
        input_filenames = []
        geo_gts = []

        for inp in input_names:
            fullname_input = os.path.join(self.root.lgg, entry[inp])
            input_filenames.append(entry[inp])
            img = nib.load(fullname_input)

            if img is None:
                raise ValueError(f'Not found image file {fullname_input}.')

            images.append(img.get_fdata())

            if (self.label_name is None) or len(self.label_name) == 0:
                continue
            if self.use_geo:
                fullname_geo = os.path.join(self.root.geo, self.label_name, str(entry[inp]))
                fullname_geo = fullname_geo.replace(inp, f"{self.label_name}_{inp}")
                geo_gt = nib.load(fullname_geo).get_fdata()

                geo_gts.append(geo_gt)

        images = np.stack(images, axis=0)

        if self.use_geo and len(geo_gts) > 0:
            geo_gts = np.stack(geo_gts, axis=-1)
            geo_gts = np.mean(geo_gts, axis=-1)
            geo_gts = geo_gts + 1.0
            geo_gts = np.transpose(geo_gts, [3, 0, 1, 2])
            transformed_dict = {'image': images, 'label': np.expand_dims(gts, 0), 'dist': np.expand_dims(geo_gts)}
        else:
            transformed_dict = {'image': images, 'label': np.expand_dims(gts, 0)}

        if self.stage == 'train':
            obj_trans = self.train_transforms(transformed_dict)
        else:
            obj_trans = self.val_transforms(transformed_dict)

        if gts is None:
            raise ValueError(f'Not found image file {fullname_gt}.')

        sample = {}
        sample['input'] = obj_trans['image']
        sample['gt'] = obj_trans['label']
        sample['paths'] = [os.path.dirname(entry[target_name])]
        if self.use_geo and "dist" in obj_trans:
            sample['dist'] = obj_trans['dist'] - 1.0

        return sample


class DataFrameMultilabelImageDataset(DataFrameImageDataset):
    def __getitem__(self, idx):
        res = DataFrameImageDataset.__getitem__(self, idx)
        res["target"] = torch.from_numpy(res["target"]).float()
        return res


class ImageFolderDataset(DataFrameImageDataset):
    def __init__(self, metadata: pd.DataFrame, transforms: solt.Stream,
            data_key: str = 'data',
            target_key: Optional[str] = 'target', mean=None, std=None):
        super(ImageFolderDataset, self).__init__(metadata, transforms, data_key, target_key, mean, std)

    def read_data(self, entry) -> dict:
        img = cv2.imread(str(getattr(entry, self.data_key)))
        return self.transforms(img, mean=self.mean, std=self.std)['image']
