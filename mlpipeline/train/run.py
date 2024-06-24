import random
import gc

import numpy as np
import cv2
import os
import logging
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import monai
import socket
import pathlib
from omegaconf import OmegaConf

from mlpipeline.utils.common import init_dist_url, init_conf, init_obj_cls
from mlpipeline.train.pipeline import MLPipeline

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def worker(local_rank, dist_url, cfg):
    # Calculate global rank and world size
    global_rank = cfg.node_rank * cfg.n_gpus + local_rank
    world_size = cfg.n_gpus * cfg.num_nodes

    cfg.global_rank = global_rank
    cfg.world_size = world_size

    logger = logging.getLogger(__name__)

    # Init process group
    if dist_url is not None:
        dist.init_process_group(
            backend='nccl', init_method=dist_url,
            world_size=world_size, rank=local_rank)

    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()
    gc.collect()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    cv2.setRNGSeed(cfg.seed)
    os.environ["CUDNN_DETERMINISTIC"] = "1"
    os.environ["PYTHONHASHSEED"] = str(cfg.seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True
    monai.utils.set_determinism(cfg.seed)

    pipeline = init_obj_cls(cfg.pipeline_cls)
    print(f'[INFO] Starting worker local rank {local_rank} of node {cfg.node_rank}. Global rank {global_rank}/{world_size - 1}.')
    # cfg.model.params.cfg.local_rank = local_rank
    ppl = pipeline(cfg, local_rank, global_rank)
    if "eval_only" in cfg and cfg.eval_only:
        ppl.eval()
    else:
        ppl.train()
        ppl.infer()
    return


def main():
    cwd = pathlib.Path().cwd()
    conf_path = cwd / "config" / "config.yaml"

    cfg = init_conf(cwd, conf_path)
    print(OmegaConf.to_yaml(cfg))

    if cfg.train.distributed:
        dist_url = init_dist_url(socket.gethostname())
        cfg.n_gpus = torch.cuda.device_count()

        node_rank = 0
        if 'SLURM_NODEID' in os.environ:
            node_rank = int(os.environ['SLURM_NODEID'])
        cfg.node_rank = node_rank

        n_nodes = 1
        if 'SLURM_JOB_NUM_NODES' in os.environ:
            n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
        cfg.num_nodes = n_nodes

        mp.spawn(worker, nprocs=cfg.n_gpus, args=(dist_url, cfg))

    else:
        cfg.n_gpus = 1
        worker(0, None, cfg)
    return


if __name__ == "__main__":
    main()
