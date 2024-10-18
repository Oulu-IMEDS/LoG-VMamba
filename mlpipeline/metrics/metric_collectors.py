import warnings
import numpy as np
import torch
import torch.nn.functional as functional
import torch.distributed as dist
import segmentation_models_pytorch as smp
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from mlpipeline.metrics import binary
import pandas as pd


def mask_list(l, masks):
    return [l[i].item() if isinstance(l, torch.Tensor) else l[i] for i in range(masks.shape[0]) if masks[i]]


def calculate_metric(metric_func, y_true, y_pred, **kwargs):
    result = None
    if len(y_pred) == len(y_true) and len(y_pred) > 0:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = metric_func(y_true, y_pred, **kwargs)
        except ValueError:
            # print(f'Metric: {metric_func}\n\tlabel: {set(y_true)}\n\tpredictions: {set(y_pred)}.')
            result = -1.0
        # print(f'Metric: {metric_func}\n\tlabel: {set(y_true)}\n\tpredictions: {set(y_pred)}.')
        # result = metric_func(y_true, y_pred, **kwargs)
    return result


class MultiLossesCollector(object):
    def __init__(self, local_rank, cfg):
        self.cfg = cfg
        self.local_rank = local_rank
        self.world_size = cfg.world_size
        self.distributed =  self.world_size > 1

        self.init()

    def init(self):
        self.losses = {}
        self.iter = 0

    def update(self, losses):
        for name in losses:
            if name in self.losses:
                self.losses[name] += losses[name]
            else:
                self.losses[name] = losses[name]

        self.iter += 1

    def compute(self):
        losses = {}
        for name in self.losses:
            if self.iter > 0:
                losses[name] = self.losses[name] / self.iter
        return losses

    def compute_on_epoch_end(self):
        cum_losses = self.compute()
        if self.distributed:
            for name in cum_losses:
                dist.all_reduce(cum_losses[name])
        for name in cum_losses:
            cum_losses[name] = cum_losses[name] / self.world_size
        return cum_losses


class EoMetrics(object):
    def __init__(self, cfg, eps=1e-6):
        self.n_classes = cfg.metrics.n_classes
        self.threshold = max(cfg.metrics.threshold, 0.0)
        self.eps = eps
        self.reset()

    def reset(self):
        self.dice_scores = np.empty((0, self.n_classes))
        self.hd95_scores = np.empty((0, self.n_classes))
        self.count = np.zeros((self.n_classes))

    def update(self, preds, targets):
        preds = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
        targets = targets.cpu().numpy() if isinstance(targets, torch.Tensor) else targets

        batch_size = targets.shape[0]
        dsc = np.empty((batch_size, self.n_classes))
        hd = np.empty((batch_size, self.n_classes))

        for b, c in np.ndindex(batch_size, self.n_classes):
            edges_pred, edges_gt = preds[b, c], targets[b, c]
            edges_pred = (edges_pred > self.threshold)
            if not np.any(edges_gt):
                warnings.warn(f"the ground truth of class {c} is all 0, this may result in nan distance.")
            if not np.any(edges_pred):
                warnings.warn(f"the prediction of class {c} is all 0, this may result in nan distance.")

            if (edges_pred.sum() > 0 and edges_gt.sum() > 0):
                dice = binary.dc(edges_pred, edges_gt)
                distance = binary.hd95(edges_pred, edges_gt)
                dsc[b, c] = dice
                hd[b, c] = distance
                self.count[c] += 1
            elif (edges_pred.sum() == 0 and edges_gt.sum() == 0):
                dsc[b, c] = 1
                hd[b, c] = 0
                self.count[c] += 1
            else:
                dsc[b, c] = 0
                hd[b, c] = 0
                self.count[c] += self.eps

        self.dice_scores = np.concatenate([self.dice_scores, dsc], axis=0)
        self.hd95_scores = np.concatenate([self.hd95_scores, hd], axis=0)

    def mean(self):
        mean_dice_scores = np.sum(self.dice_scores, axis=0) / self.count
        mean_hd95_scores = np.sum(self.hd95_scores, axis=0) / self.count
        return mean_dice_scores, mean_hd95_scores


class SemanticSegmentationMetricsCollector(object):
    METRIC_NAMES = [
        "IoU", "Sens", "Spec", "BA", "F1", "Accuracy",
        "EDice", "EDice1", "EDice2", "EDice3",
        "HD95", "HD95_1", "HD95_2", "HD95_3",
    ]

    def __init__(self, local_rank, cfg):
        self.distributed = cfg.train.distributed
        self.local_rank = local_rank
        self.world_size = cfg.world_size

        self.mode = cfg.metrics.mode
        self.output_mode = cfg.metrics.mode if (cfg.metrics.output_mode is None) else cfg.metrics.output_mode
        self.n_classes = cfg.metrics.n_classes
        self.threshold = cfg.metrics.threshold
        self.reduction = cfg.metrics.reduction if cfg.metrics.reduction is not None else "micro"
        self.eo_metrics = EoMetrics(cfg)

        self.cfg = cfg
        self.reset()

    def reset(self):
        # SS results data
        if "imagewise" in self.reduction:
            self.tp = torch.tensor([]).cuda(self.local_rank, non_blocking=True)
            self.fp = torch.tensor([]).cuda(self.local_rank, non_blocking=True)
            self.fn = torch.tensor([]).cuda(self.local_rank, non_blocking=True)
            self.tn = torch.tensor([]).cuda(self.local_rank, non_blocking=True)
        else:
            self.tp = torch.tensor(0.0).cuda(self.local_rank, non_blocking=True)
            self.fp = torch.tensor(0.0).cuda(self.local_rank, non_blocking=True)
            self.fn = torch.tensor(0.0).cuda(self.local_rank, non_blocking=True)
            self.tn = torch.tensor(0.0).cuda(self.local_rank, non_blocking=True)
        # self.auroc = AUROC(num_classes=self.n_classes, thresholds=None, average="micro", task="binary").cuda()
        self.eo_metrics.reset()

    def sigmoid_scores_to_classes(self, output, dim=None):
        dim = 1 if dim is None else dim
        output_scores = torch.max(output, dim=dim)[0]
        output_class_indices = torch.argmax(output, dim=dim) + 1
        output = torch.where(
            output_scores >= self.threshold,
            output_class_indices,
            torch.zeros_like(output_class_indices),
        )
        return output

    def to_class_output(self, preds, dim=None, threshoding=True):
        if self.cfg.model.params.cfg.arch == "SkelCon":
            return preds
        elif (self.output_mode == "multiclass") and (self.n_classes > 1):
            dim = 1 if dim is None else dim
            probs = functional.softmax(preds, dim=dim)
            class_indices = torch.argmax(probs, dim=dim)
            return class_indices.unsqueeze(dim=dim)
        elif self.output_mode == "tanh":
            probs = torch.tanh(preds)
            if not threshoding:
                return probs
            class_indices = (probs >= self.threshold).long()
            return class_indices
        output = torch.sigmoid(preds)
        if self.mode == "multiclass":
            output = self.sigmoid_scores_to_classes(output)
        return output

    def compute_segmentation_results(self, preds, targets, batch_size=1):
        threshold = 0.5 if self.output_mode == "tanh" else self.threshold
        kwargs = {
            "threshold": None if self.mode == "multiclass" else threshold,
            "num_classes": self.n_classes if self.mode == "multiclass" else None,
        }

        if isinstance(preds, list) or isinstance(preds, tuple):
            tp, fp, fn, tn = [], [], [], []
            for start_index in range(0, len(preds), batch_size):
                end_index = min(start_index + batch_size, len(preds))
                batch_preds = torch.cat(preds[start_index:end_index], dim=0)
                batch_targets = torch.cat(targets[start_index:end_index], dim=0)

                batch_preds = self.to_class_output(batch_preds)
                batch_preds = batch_preds.squeeze(dim=1)
                batch_targets = batch_targets.long() if self.mode == "multiclass" else batch_targets
                # batch_targets = batch_targets[:, :1, :, :] if self.mode == "tanh" else batch_targets
                batch_targets = batch_targets.squeeze(dim=1)
                # correct_count = torch.sum(batch_preds == batch_targets)
                # accuracy_score = correct_count / batch_targets.numel()
                # print("Accuracy:", accuracy_score)

                if (self.mode == "binary") and (batch_preds.shape[1] != self.n_classes or batch_targets.shape[1] != self.n_classes):
                    batch_preds = batch_preds.unsqueeze(dim=1)
                    batch_targets = batch_targets.unsqueeze(dim=1)
                print(batch_preds.shape, batch_targets.shape, batch_preds.max(), batch_targets.max())
                (batch_tp, batch_fp, batch_fn, batch_tn) = smp.metrics.get_stats(
                    batch_preds, batch_targets, mode=self.mode, **kwargs)
                self.eo_metrics.update(batch_preds, batch_targets)
                # self.auroc.update(batch_preds.flatten(), batch_targets.flatten())

                tp.append(batch_tp)
                fp.append(batch_fp)
                fn.append(batch_fn)
                tn.append(batch_tn)

            tp = torch.cat(tp, dim=0)
            fp = torch.cat(fp, dim=0)
            fn = torch.cat(fn, dim=0)
            tn = torch.cat(tn, dim=0)
        else:
            preds = self.to_class_output(preds)
            preds = preds.squeeze(dim=1)
            targets = targets.long() if self.mode == "multiclass" else targets
            # targets = targets[:, :1, :, :] if self.output_mode == "tanh" else targets
            targets = targets.squeeze(dim=1)
            """
            if ignore_index is not None:
                preds = preds + ignore_index
                targets = targets + ignore_index
            """

            if (self.mode == "binary") and (targets.shape[1] != self.n_classes or preds.shape[1] != self.n_classes):
                preds = preds.unsqueeze(dim=1)
                targets = targets.unsqueeze(dim=1)
            # print(preds.shape, targets.shape, preds.max(), targets.max())
            (tp, fp, fn, tn) = smp.metrics.get_stats(preds, targets, mode=self.mode, **kwargs)
            self.eo_metrics.update(preds, targets)
            # self.auroc.update(preds.flatten(), targets.flatten())

        if "imagewise" in self.reduction:
            self.tp = torch.cat([self.tp, tp], dim=0) if len(self.tp) else tp
            self.fp = torch.cat([self.fp, fp], dim=0) if len(self.fp) else fp
            self.fn = torch.cat([self.fn, fn], dim=0) if len(self.fn) else fn
            self.tn = torch.cat([self.tn, tn], dim=0) if len(self.tn) else tn
        else:
            self.tp += tp.sum()
            self.fp += fp.sum()
            self.fn += fn.sum()
            self.tn += tn.sum()
        return

    def compute(self, preds, targets):
        self.compute_segmentation_results(preds, targets)

    def all_reduce(self):
        if self.distributed:
            dist.all_reduce(self.tp, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.fp, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.fn, op=dist.ReduceOp.SUM)
            dist.all_reduce(self.tn, op=dist.ReduceOp.SUM)
            # dist.all_reduce(self.auroc, op=dist.ReduceOp.SUM)
        return

    def mean(self):
        mean_metrics = {
            "IoU": smp.metrics.iou_score(self.tp, self.fp, self.fn, self.tn, reduction=self.reduction),
            "Sens": smp.metrics.sensitivity(self.tp, self.fp, self.fn, self.tn, reduction=self.reduction),
            "Spec": smp.metrics.specificity(self.tp, self.fp, self.fn, self.tn, reduction=self.reduction),
            "BA": smp.metrics.balanced_accuracy(self.tp, self.fp, self.fn, self.tn, reduction=self.reduction),
            "F1": smp.metrics.f1_score(self.tp, self.fp, self.fn, self.tn, reduction=self.reduction),
            "Accuracy": smp.metrics.accuracy(self.tp, self.fp, self.fn, self.tn, reduction=self.reduction),
            # "AUROC": self.auroc.compute(),
        }
        eo_dice, eo_hd95 = self.eo_metrics.mean()
        mean_metrics.update({
            "EDice": eo_dice.mean(),
            "HD95": eo_hd95.mean(),
            **{f"EDice{i + 1}": eo_dice[i] for i in range(eo_dice.shape[0])},
            **{f"HD95_{i + 1}": eo_hd95[i] for i in range(eo_hd95.shape[0])},
        })
        print("Metrics:", mean_metrics["F1"], mean_metrics["EDice"], mean_metrics["HD95"])
        print("Metrics:", mean_metrics["IoU"], mean_metrics["Sens"], mean_metrics["Spec"])
        return mean_metrics
