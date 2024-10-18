import numpy as np


class LRScheduler:
    def __init__(self, cfg, optimizer, epoch):
        self.cfg = cfg
        self.optimizer = optimizer
        self.epoch = epoch
        self.lr = self.cfg.optimizer.params.lr

    def lr_schedule_epoch(self):
        if self.cfg.optimizer.scheduler.type == 'milestones':
            lr = self.lr
            scheduler_conf = self.cfg.optimizer.scheduler
            for epoch_drop in scheduler_conf.milestones:
                if self.epoch == epoch_drop:
                    lr *= scheduler_conf.gamma
        elif self.cfg.optimizer.scheduler.type == 'annealing':
            # replication of the schedule from SWA paper
            t = self.epoch / self.cfg.train.num_epochs
            lr_scaler = self.cfg.optimizer.scheduler.lr_scaler
            t1 = self.cfg.optimizer.scheduler.t1
            t2 = self.cfg.optimizer.scheduler.t2

            # We run a constant high lr for 50% of the time
            # For the remaining 40%, we will linearly decrease it
            # The last 10% of the training time, we will just run training with low LR
            if t <= t1:
                factor = 1.0
            elif t <= t2:
                factor = 1.0 - (1.0 - lr_scaler) * (t - t1) / (t2 - t1)
            else:
                factor = lr_scaler

            lr = self.cfg.optimizer.params.lr * factor
        else:
            raise NotImplementedError('Unknown scheduler')
        return lr

    def step(self, epoch=None):
        if epoch is None:
            self.epoch += 1
        else:
            self.epoch = epoch
        new_lr = self.lr_schedule_epoch()

        if new_lr != self.lr:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr

        # This assumes that we have only one lr for all the parameter groups!
        lrs = [pg["lr"] for pg in self.optimizer.param_groups]
        lrs = np.unique(lrs)
        assert lrs.shape[0] == 1
        self.lr = lrs[0]
