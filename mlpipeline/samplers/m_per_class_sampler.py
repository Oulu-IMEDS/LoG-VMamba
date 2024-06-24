import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import DistributedSampler
from mlpipeline.samplers import utils as c_f
from typing import TypeVar, Optional, Iterator
from torch.utils.data.dataset import Dataset

# modified from
class DistributedMPerClassSampler(DistributedSampler):
    """
    At every iteration, this will return m samples per class. For example,
    if dataloader's batchsize is 100, and m = 5, then 20 classes with 5 samples
    each will be returned
    """

    def __init__(self, dataset: Dataset, labels, m, batch_size=None, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False,
                 length_before_new_iter=100000):
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.m_per_class = int(m)
        self.batch_size = int(batch_size) if batch_size is not None else batch_size
        self.labels_to_indices = c_f.get_labels_to_indices(labels)
        self.labels = list(self.labels_to_indices.keys())
        self.length_of_single_pass = self.m_per_class * len(self.labels) // self.num_replicas
        length_before_new_iter_per_proc = length_before_new_iter // self.num_replicas
        self.list_size_per_proc = length_before_new_iter_per_proc
        if self.batch_size is None:
            if self.length_of_single_pass < self.list_size_per_proc:
                self.list_size_per_proc -= (self.list_size_per_proc) % (self.length_of_single_pass)
        else:
            assert self.list_size_per_proc >= self.batch_size
            assert (
                self.length_of_single_pass >= self.batch_size
            ), "m * (number of unique labels) must be >= batch_size"
            assert (
                self.batch_size % self.m_per_class
            ) == 0, "m_per_class must divide batch_size without any remainder"
            self.list_size_per_proc -= self.list_size_per_proc % self.batch_size

    def __len__(self):
        return self.list_size_per_proc

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.labels), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.labels)))  # type: ignore[arg-type]

        labels = [self.labels[i] for i in indices]

        idx_list = [0] * self.list_size_per_proc
        i = 0
        num_iters = self.calculate_num_iters()
        labels_per_proc = labels[self.rank:num_iters:self.num_replicas]
        for _ in range(num_iters):
            c_f.NUMPY_RANDOM.shuffle(labels_per_proc)
            if self.batch_size is None:
                curr_label_set = labels_per_proc
            else:
                curr_label_set = labels_per_proc[: self.batch_size // self.m_per_class]
            for label in curr_label_set:
                t = self.labels_to_indices[label]
                idx_list[i : i + self.m_per_class] = c_f.safe_random_choice(
                    t, size=self.m_per_class
                )
                i += self.m_per_class
        return iter(idx_list)

    def calculate_num_iters(self):
        divisor = (
            self.length_of_single_pass if self.batch_size is None else self.batch_size
        )
        return self.list_size_per_proc // divisor if divisor < self.list_size_per_proc else 1
