from collections import Counter
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from .filter_dataset import FilterDataset


class WeightedDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset
                 ):
        self.base_dataset = dataset

    def __getitem__(self, item):
        rec = self.base_dataset[item]
        rec = ({k: FilterDataset.to_torch(v) for k, v in rec[0].items()}, rec[1])
        return rec

    def __len__(self):
        return len(self.base_dataset)

    @staticmethod
    def get_weighted_random_sampler(dataset: List[Tuple[Dict, int]]):
        y_train = list(map(lambda x: x[1], dataset))
        count = Counter(y_train)
        class_count = np.array([count[0], count[1]])
        weight = 1. / class_count
        samples_weight = np.array([weight[t] for t in y_train])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        return sampler