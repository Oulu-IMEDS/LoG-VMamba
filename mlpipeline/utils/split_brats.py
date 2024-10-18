import nibabel as nib
import numpy as np
from pathlib import Path
import pandas as pd
import os
import click
import pickle
import random
from natsort import natsorted

from mlpipeline.data.splitter import FoldSplit


def generate_metadata(root_dir, prefix, out_dir=None, id_len=3):
    dirs = [d for d in os.listdir(root_dir) if os.path.isdir(root_dir / Path(d))]
    suffices = ['flair', 't1ce', 't1', 't2', 'seg']
    data = []
    for dir_name in dirs:
        id = dir_name[-id_len:]
        row = {'ID': id}
        for suffix in suffices:
            fullname = root_dir / Path(dir_name) / Path(f"{prefix}_{id}_{suffix}.nii.gz")
            if os.path.isfile(fullname):
                row[suffix] = Path(dir_name) / Path(f"{prefix}_{id}_{suffix}.nii.gz")
        data.append(row)

    data = pd.DataFrame(data)
    return data


def split_train_test(df, output_dir, target_col=None, group_col=None, seed=28, n_folds=5):
    # Split train vs test
    splitter = FoldSplit(ds=df, n_folds=n_folds, target_col=target_col, group_col=group_col, random_state=seed)
    df_train_val, df_test = splitter.fold(0)

    # Split k-fold
    data = []
    output_fullname = os.path.join(output_dir, f"cv_split_{n_folds}folds_brats_{seed:05d}.pkl")
    splitter_kf = FoldSplit(ds=df_train_val, n_folds=n_folds, target_col=target_col, group_col=group_col, random_state=seed)
    for fold_ind in range(n_folds):
        df_train, df_val = splitter_kf.fold(fold_ind)
        data.append((df_train, df_val))
        print(f"Fold {fold_ind}, Dataset: BraTs, Size: {len(df_train_val)} to {len(df_train)}/{len(df_val)}")

    with open(output_fullname, "wb") as f:
        print(f"Saving metadata to {output_fullname}")
        pickle.dump(data, f, protocol=4)

    # Save test data
    output_fullname = os.path.join(output_dir, f"cv_split_{n_folds}folds_brats_test_{seed:05d}.pkl")
    print(len(df_test))
    with open(output_fullname, "wb") as f:
        print(f"Saving metadata to {output_fullname}")
        pickle.dump(df_test, f, protocol=4)


@click.command()
@click.option("--root")
@click.option("--output_dir")
@click.option("--seed", default=99999)
def main(root, output_dir, seed):
    dataset = 'brats'
    print(f'Seed: {seed}')
    train_dir = root / Path("MICCAI_BraTS2020_TrainingData")
    metadata_fullname = Path(output_dir) / f'{dataset}_metadata_brats_{seed}.csv'
    df = generate_metadata(train_dir, prefix="BraTS20_Training")
    print(f'Saving metadata to {metadata_fullname}')
    df.to_csv(metadata_fullname, index=None)

    # Generate seed
    if seed == -1:
        seed = random.randint(1, 99999)

    # Data
    n_folds = 5
    split_train_test(df, output_dir=output_dir, n_folds=n_folds, seed=seed)


if __name__ == "__main__":
    main()
