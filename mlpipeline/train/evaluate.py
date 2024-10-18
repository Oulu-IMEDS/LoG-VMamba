import os
import re
import json
import csv
import pickle
import math
from pathlib import Path

import click
import numpy as np
import cv2
import nibabel as nib
import torch
import torch.nn.functional as functional
import monai.transforms as MonaiTrans
from omegaconf import OmegaConf
from natsort import natsorted
from tqdm import tqdm

import mlpipeline.utils.common as common
from mlpipeline.metrics.metric_collectors import SemanticSegmentationMetricsCollector
from mlpipeline.data.augs import resize_mask


class Evaluator:
    def __init__(
        self,
        output_dir, log_dir, visual_dir,
        metadata,
        cfg,
        dataset_name,
        seeds, folds,
        key_metric="EDice",
    ):
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.visual_dir = str(Path(visual_dir) / dataset_name)
        self.metadata = metadata
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.seeds = seeds
        self.folds = folds
        self.batch_size = 1 if self.dataset_name.lower() in ["lgg"] else 2

        self.metrics_collector = SemanticSegmentationMetricsCollector(
            local_rank=0,
            cfg=self.cfg,
        )
        self.key_metric = key_metric
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        assert key_metric in SemanticSegmentationMetricsCollector.METRIC_NAMES

        if self.dataset_name.lower() == "brats":
            self.label_transforms = MonaiTrans.compose.Compose([
                MonaiTrans.EnsureTyped(keys=["label"]),
                MonaiTrans.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
                MonaiTrans.ToTensord(keys=["label"]),
            ])
        elif self.dataset_name.lower() == "lgg":
            self.label_transforms = MonaiTrans.compose.Compose([
                MonaiTrans.EnsureChannelFirstd(keys=["label"]),
                MonaiTrans.EnsureTyped(keys=["label"]),
                MonaiTrans.ToTensord(keys=["label"]),
            ])

        # Get all directories of inference results
        self.experiment_dirs = Path(output_dir).glob(f"*")
        self.experiment_dirs = [d for d in self.experiment_dirs if d.is_dir()]
        print(output_dir, len(self.experiment_dirs))

        # Get gt files
        if dataset_name.lower() in ["brats"]:
            self.gt_paths = natsorted([Path(cfg.data.image_dir.brats) / filename for filename in metadata["seg"]])
        elif dataset_name.lower() in ["lgg"]:
            self.dataset_name = "LGG"
            self.gt_paths = natsorted([Path(cfg.data.image_dir.lgg) / filename for filename in metadata["seg"]])

        self.gt_paths = [gt_path for gt_path in self.gt_paths]
        self.gt_masks = []

        pbar = tqdm(self.gt_paths, total=len(self.gt_paths), desc="Predicting::")
        for gt_path in pbar:
            # Read gt data
            if str(gt_path).endswith(".nii.gz"):
                raw_gt_mask = common.read_nii_data(gt_path)
                raw_gt_mask = self.label_transforms({"label": raw_gt_mask})["label"]
                if self.cfg.metrics.mode == "multiclass":
                    gt_mask = np.sum(raw_gt_mask, axis=0)
                    # gt_mask = np.zeros([3, raw_gt_mask.shape[0], raw_gt_mask.shape[1], raw_gt_mask.shape[2]])
                    # gt_mask[0, raw_gt_mask == 1] = 1
                    # gt_mask[1, raw_gt_mask == 2] = 1
                    # gt_mask[2, raw_gt_mask == 4] = 1
                else:
                    gt_mask = raw_gt_mask
                self.target_size = (raw_gt_mask.shape[1], raw_gt_mask.shape[2], raw_gt_mask.shape[3])
                gt_mask = torch.tensor(gt_mask).long().unsqueeze(dim=0).to(self.device)
                self.gt_masks.append(gt_mask)
            else:
                gt_mask = common.read_image(str(gt_path), gray=True)
                self.target_size = (gt_mask.shape[0], gt_mask.shape[1])
                gt_mask = torch.tensor(gt_mask / 255.0).long().unsqueeze(dim=0).to(self.device)
                self.gt_masks.append(gt_mask)

        print("GT:", len(self.gt_masks))

    def _map_gt_to_input_name(self, gt_name):
        if self.dataset_name.lower() == "brats":
            self.dataset_name = "BraTS"
            return gt_name.replace("_seg.nii", "")
        elif self.dataset_name.lower() == "lgg":
            self.dataset_name = "LGG"
            return gt_name.replace("_seg.nii", "")
        return gt_name

    def visualize_results(self, gt_path, gt_image, output_logits, setting, seed):
        image_name = self._map_gt_to_input_name(gt_path.stem)
        preds = self.metrics_collector.to_class_output(
            torch.tensor(output_logits),
            threshoding=False,
            dim=1,
        ).squeeze(dim=1)
        preds = preds.cpu().numpy()

        if preds.ndim == 3:
            preds = preds.transpose(1, 2, 0)
            preds = (preds > self.cfg.metrics.threshold)
            preds = (preds * 255).astype(np.uint8).squeeze(axis=-1)
            gt_image = common.read_image(str(gt_path), gray=True)
            gt_image = cv2.resize(gt_image, (512, 512), interpolation=cv2.INTER_LANCZOS4)
            if gt_image.shape[0] != preds.shape[0] or gt_image.shape[1] != preds.shape[1]:
                preds = cv2.resize(preds, (gt_image.shape[1], gt_image.shape[0]))
            output_image = np.concatenate([preds, gt_image], axis=1)
            # output_image = preds

            cv2.imwrite(
                str(Path(self.visual_dir) / setting / f"seed-{seed}_name-{image_name}.png"),
                output_image,
            )

        elif preds.ndim == 5:
            preds = np.squeeze(preds, axis=0)
            image = nib.Nifti1Image(preds.astype(float), affine=np.eye(4))
            nib.save(image, Path(self.visual_dir) / setting / f"seed-{seed}_name-{image_name}.nii.gz")
            gt_image = gt_image.cpu().numpy()
            final_display_image = []
            for index in range(3):
                gt_layer = gt_image[0, index, :, :, 80] * 255
                gt_layer = np.stack([gt_layer] * 3, axis=-1)
                output_layer = preds[index, :, :, 80]
                thresholded_layer = np.stack([(output_layer > 0.0) * 255] * 3, axis=-1)
                output_layer = np.clip(output_layer * 128 + 128, a_min=0, a_max=255).astype(np.uint8)
                output_layer = common.convert_grayscale_to_heatmap(output_layer, "inferno")
                output_layer = np.concatenate([gt_layer, output_layer, thresholded_layer], axis=0)
                final_display_image.append(output_layer)
            final_display_image = np.concatenate(final_display_image, axis=1)
            cv2.imwrite(str(Path(self.visual_dir) / setting / f"seed-{seed}_name-{image_name}.png"), final_display_image)
        return

    def compute_metrics_on_setting(self, setting):
        metrics_dict = dict()

        # Filter by setting
        experiment_dirs = [
            d for d in self.experiment_dirs
            if d.name.startswith(setting)
        ]
        os.makedirs(Path(self.visual_dir) / setting, exist_ok=True)

        # Loop over seeds
        n_seeds = 0
        pbar = tqdm(self.seeds, total=len(self.seeds), desc="Computing::")
        for seed in pbar:
            output_masks = []
            self.metrics_collector.reset()

            for path_index, gt_path in enumerate(self.gt_paths):
                image_name = self._map_gt_to_input_name(gt_path.stem)
                # Get output data, and average over folds
                fold_outputs = []

                for fold_index in self.folds:
                    seed_str = f"seed:{seed}"
                    fold_str = f"fold:{fold_index}"
                    experiment_dir = [
                        d for d in experiment_dirs
                        if (seed_str in str(d)) and (fold_str in str(d))
                    ]

                    if len(experiment_dir) == 1:
                        experiment_dir = experiment_dir[0]
                    else:
                        print(f"Seed {seed} and Fold {fold_index} not found: {len(experiment_dir)}\n")
                        continue

                    # Read output data
                    output_path = experiment_dir / self.dataset_name / f"{image_name}.pt"
                    output_logits = torch.load(output_path, map_location=self.device)
                    # output_logits = output_logits.squeeze(dim=0)
                    # Collect
                    fold_outputs.append(output_logits)

                # Get average over folds
                if len(fold_outputs) > 1:
                    seed_output = torch.mean(torch.stack(fold_outputs, dim=0), dim=0)
                elif len(fold_outputs) == 1:
                    seed_output = fold_outputs[0]
                else:
                    continue

                # seed_output = self.metrics_collector.to_class_output(seed_output)
                self.visualize_results(gt_path, self.gt_masks[path_index], seed_output, setting, seed)
                output_masks.append(seed_output)

            # Compute metrics
            # output_masks = torch.cat(output_masks, dim=0)
            # gt_masks = torch.cat(self.gt_masks, dim=0)
            print(len(output_masks))
            if len(output_masks) != len(self.gt_masks):
                print(f"Not enought samples {len(output_masks)} < {len(self.gt_masks)}. Skip!")
                continue

            for i, ts in enumerate(output_masks):
                print(f'Shape of output {i}: {ts.shape}')

            for i, ts in enumerate(self.gt_masks):
                print(f'Shape of gt {i}: {ts.shape}')
            self.metrics_collector.compute(output_masks, self.gt_masks)

            # self.metrics_collector.compute(output_masks, gt_masks)
            metrics_dict[seed] = self.metrics_collector.mean()
            n_seeds += 1

        # Compute statistics over runs
        summary_dict = {"mean": dict(), "std": dict(), "std_error": dict()}
        for metric_name in SemanticSegmentationMetricsCollector.METRIC_NAMES:
            if metric_name in metrics_dict[seed]:
                values = torch.tensor([metrics_dict[seed][metric_name] for seed in metrics_dict.keys()])
                summary_dict["mean"][metric_name] = torch.mean(values).item()
                summary_dict["std"][metric_name] = torch.std(values, unbiased=False).item()
                summary_dict["std_error"][metric_name] = torch.std(values, unbiased=False).item() / np.sqrt(n_seeds)

        summary_dict["seeds"] = list(metrics_dict.keys())

        return summary_dict

    def run(self):
        os.makedirs(self.visual_dir, exist_ok=True)
        # Get list of hyperparam settings, independent of seed and fold
        settings = set()

        for experiment_dir in self.experiment_dirs:
            basename = experiment_dir.name
            seed_index = basename.find("_seed:")
            if seed_index < 0:
                continue

            setting = basename[:seed_index]
            settings.add(setting)

        self.settings = list(settings)
        print(f"Settings: {len(self.settings)}", self.settings[:2])
        eval_results = []

        # Loop over settings
        for setting in self.settings:
            print(setting)
            with torch.no_grad():
                metrics = self.compute_metrics_on_setting(setting)
            eval_results.append([setting, metrics])
            print(metrics)
            print("\n")

        # Log best result
        best_result = max(eval_results, key=lambda x: x[1]["mean"][self.key_metric])
        method_name = re.findall(f"_method:(.*?)_", best_result[0])[0]
        output_fullname = Path(self.log_dir) / f"{self.dataset_name}_{method_name}.json"
        print(f"Writing resuls to {output_fullname}")
        json.dump(
            {
                "method": method_name,
                "dataset": self.dataset_name,
                "setting": best_result[0],
                "mean": best_result[1]["mean"],
                "std": best_result[1]["std"],
                "std_error": best_result[1]["std_error"],
            },
            open(output_fullname, "w+"),
            indent=4)
        return


@click.command()
@click.option("--config")
@click.option("--output_dir")
@click.option("--log_dir")
@click.option("--visual_dir")
@click.option("--metadata_path")
@click.option("--dataset_name")
@click.option("--seeds", default="12345,28966", type=str)
@click.option("--folds", default="0")
def main(
    config: str,
    output_dir: str, log_dir: str, visual_dir: str,
    metadata_path: str,
    dataset_name: str,
    seeds: str, folds: int,
):
    # Get config
    cwd = Path().cwd()
    conf_path = cwd / "config" / "experiment" / f"{config}.yaml"
    cfg = OmegaConf.load(str(conf_path))

    if metadata_path.endswith(".yaml"):
        metadata = OmegaConf.load(metadata_path)
    elif metadata_path.endswith(".pkl"):
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

    seeds = [int(x) for x in seeds.split(",")]
    folds = [int(x) for x in folds.split(",")]

    evaluator = Evaluator(
        output_dir=output_dir,
        log_dir=log_dir,
        visual_dir=visual_dir,
        metadata=metadata,
        cfg=cfg,
        dataset_name=dataset_name,
        seeds=seeds,
        folds=folds,
    )
    evaluator.run()


if __name__ == "__main__":
    main()