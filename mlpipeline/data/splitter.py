import pickle

import numpy as np
import pandas as pd
import torch
from sklearn import model_selection
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

try:  # Handling API difference between pytorch 1.1 and 1.2
    from torch.utils.data.dataloader import default_collate
except ImportError:
    from torch.utils.data._utils.collate import default_collate


class Splitter(object):
    def __init__(self):
        self.__ds_chunks = None
        self.__folds_iter = None
        pass

    def __next__(self):
        if self.__folds_iter is None:
            raise NotImplementedError
        else:
            next(self.__folds_iter)

    def __iter__(self):
        if self.__ds_chunks is None:
            raise NotImplementedError
        else:
            return self

    def dump(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.__ds_chunks, f, pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.__ds_chunks = pickle.load(f)
            self.__folds_iter = iter(self.__ds_chunks)


class FoldSplit(Splitter):
    def __init__(self, ds: pd.DataFrame, n_folds: int = 5, target_col: str or None = None,
                 group_col: str or None = None, random_state: int or None = None):
        super().__init__()
        if target_col is None and group_col is None:
            splitter = model_selection.KFold(n_splits=n_folds, random_state=random_state, shuffle=True)
            split_iter = splitter.split(ds)            
        elif group_col is None:
            splitter = model_selection.StratifiedKFold(n_splits=n_folds, random_state=random_state, shuffle=True)
            split_iter = splitter.split(ds, ds[target_col])
        else:
            splitter = model_selection.GroupKFold(n_splits=n_folds)
            split_iter = splitter.split(ds, ds[target_col], groups=ds[group_col])

        self.__cv_folds_idx = [(train_idx, val_idx) for (train_idx, val_idx) in split_iter]
        self.__ds_chunks = [(ds.iloc[split[0]], ds.iloc[split[1]]) for split in self.__cv_folds_idx]
        self.__folds_iter = iter(self.__ds_chunks)

    def __next__(self):
        return next(self.__folds_iter)

    def __iter__(self):
        return self

    def dump(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.__ds_chunks, f, pickle.HIGHEST_PROTOCOL)

    def fold(self, i):
        return self.__ds_chunks[i]

    def n_folds(self):
        return len(self.__cv_folds_idx)

    def fold_idx(self, i):
        return self.__cv_folds_idx[i]
