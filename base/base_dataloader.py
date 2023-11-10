import numpy as np
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers, val_split=0.0):
        self.shuffle = shuffle
        self.dataset = dataset
        self.nbr_examples = len(dataset)
        if val_split:
            self.train_sampler, self.val_sampler = self._split_sampler(val_split)
        else:
            self.train_sampler, self.val_sampler = None, None

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers,
            'pin_memory': True
        }
        super(BaseDataLoader, self).__init__(sampler=self.train_sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        self.shuffle = False

        split_indx = int(self.nbr_examples * split)
        np.random.seed(0)

        indxs = np.arange(self.nbr_examples)
        np.random.shuffle(indxs)
        train_indxs = indxs[split_indx:]
        val_indxs = indxs[:split_indx]
        self.nbr_examples = len(train_indxs)

        train_sampler = SubsetRandomSampler(train_indxs)
        val_sampler = SubsetRandomSampler(val_indxs)
        return train_sampler, val_sampler

    def get_val_loader(self):
        if self.val_sampler is None:
            return None
        # self.init_kwargs['batch_size'] = 1
        return DataLoader(sampler=self.val_sampler, **self.init_kwargs)


class DataPrefetcher(object):
    def __init__(self, loader, device, stop_after=None, unlabelled=False):
        self.loader = loader
        self.dataset = loader.dataset
        self.stream = torch.cuda.Stream()
        self.stop_after = stop_after
        self.next_input = None
        self.next_target = None
        self.next_target_edges = None
        self.next_target_centers = None
        self.next_name = None
        self.device = device
        self.return_id = self.dataset.return_id
        self.unlabelled = unlabelled
        self.edges = self.dataset.edges

    def __len__(self):
        return len(self.loader)

    def preload(self):
        try:
            if self.unlabelled:
                self.next_input, self.next_target_edges, self.next_name, self.next_idx = next(self.loaditer)
            else:
                if self.edges:
                    if self.return_id:
                        self.next_input, self.next_target, self.next_target_edges, _, self.next_idx = next(
                            self.loaditer)
                    else:
                        self.next_input, self.next_target, self.next_target_edges = next(self.loaditer)
                else:
                    if self.return_id:
                        self.next_input, self.next_target, self.next_idx, _ = next(self.loaditer)
                    else:
                        self.next_input, self.next_target = next(self.loaditer)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_target_edges = None
            self.next_target_centers = None
            self.next_name = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(device=self.device, non_blocking=True)
            if self.unlabelled:
                self.next_target_edges = self.next_target_edges.cuda(device=self.device, non_blocking=True)
            else:
                self.next_target = self.next_target.cuda(device=self.device, non_blocking=True)
            if self.edges:
                self.next_target_edges = self.next_target_edges.cuda(device=self.device, non_blocking=True)

    def __iter__(self):
        count = 0
        self.loaditer = iter(self.loader)
        self.preload()
        while self.next_input is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            input = self.next_input
            target = self.next_target
            if self.unlabelled:
                target_edges = self.next_target_edges
                names = self.next_name
            if self.edges:
                target_edges = self.next_target_edges
            if self.return_id:
                idx = self.next_idx
            self.preload()
            count += 1
            if self.unlabelled:
                yield input, target_edges, names, idx
            else:
                if self.edges:
                    if self.return_id:
                        yield input, target, target_edges, idx
                    else:
                        yield input, target, target_edges
                else:
                    if self.return_id:
                        yield input, target, idx
                    else:
                        yield input, target
            if type(self.stop_after) is int and (count > self.stop_after):
                break