import torch
import operator
import os


class Checkpointer:
    def __init__(self, pipeline):
        self.cache_dir = pipeline.cache_dir
        self.model = pipeline.model
        self.model_name = pipeline.cfg.model.name
        self.logger = pipeline.logger
        self.keep_old = pipeline.cfg.checkpointer.keep_old
        self.comparator = getattr(operator, pipeline.cfg.checkpointer.comparator)
        self.pipeline = pipeline

        self.optimizer = pipeline.optimizer
        self.lr_scheduler = pipeline.lr_scheduler

        self.best_snapshot_fname = None
        self.best_val_metric = None

    def save_state(self, metric_val):
        epoch = self.pipeline.epoch

        ckpt_cand_name = self.cache_dir / f'epoch_{epoch:03d}_{self.model_name}_{metric_val:.4f}.pth'

        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        state = {
            "model": model_state,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": {"lr": self.lr_scheduler.lr, "epoch": self.lr_scheduler.epoch},
        }

        if self.best_snapshot_fname is None:
            self.logger.info(f'Saving the first snapshot [{os.path.abspath(ckpt_cand_name)}]')
            torch.save(state, ckpt_cand_name)
            self.best_snapshot_fname = ckpt_cand_name
            self.best_val_metric = metric_val
        else:
            if self.comparator(metric_val, self.best_val_metric):
                if (not self.keep_old) and (self.best_snapshot_fname.exists()):
                    self.best_snapshot_fname.unlink()
                self.logger.info(f'Saving model [{os.path.abspath(ckpt_cand_name)}]')
                self.best_snapshot_fname = ckpt_cand_name
                self.best_val_metric = metric_val
                torch.save(state, ckpt_cand_name)
        return
