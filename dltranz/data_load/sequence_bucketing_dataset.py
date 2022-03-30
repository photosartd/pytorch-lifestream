import time
from typing import List
from collections import defaultdict

import numpy as np
import torch

from .filter_dataset import FilterDataset
from dltranz.trx_encoder import PaddedBatch


class SequenceBucketingFilterDataset(FilterDataset):
    """
    Класс, который дополнительно делает SequenceBucketing, чтобы снизить паддинг
    """

    def __init__(self, dataset, post_processing=None, shuffle_files=False, shuffle_seed=42, print_info=False,
                 do_sequence_bucketing=True):
        """
        Parameters:
            dataset: List[Tuple[Dict, int]]
            do_sequence_bucketing: нужно ли делать sequence_bucketing; иначе просто сеплирует последовательно
        """
        super().__init__(dataset, post_processing, shuffle_files, shuffle_seed)
        self.seq_len = 'seq_len'
        self.print_info = print_info
        if do_sequence_bucketing:
            self.base_dataset = self._sequence_bucketing(dataset)

    def _get_gen(self, my_ids):
        for ind in my_ids:
            rec = self.base_dataset[ind]
            rec = ({k: self.to_torch(v) for k, v in rec[0].items()}, rec[1])
            yield rec

    def _sequence_bucketing(self, dataset: List):
        start_time = time.time()
        dataset_len = len(dataset)
        for idx in range(dataset_len):
            dictionary = dataset[idx][0]
            for key, value in dictionary.items():
                # if not id (-> np.ndarray with feature)
                if not (isinstance(value, int) or isinstance(value, float)):
                    dictionary[self.seq_len] = len(value)
                    break
        dataset.sort(key=lambda tup: tup[0][self.seq_len])
        for tup in dataset:
            try:
                del tup[0][self.seq_len]
            except Exception as e:
                print(f'No {self.seq_len} key while sequence bucketing')
        if self.print_info:
            print(f'Done sequence bucketing in: {time.time() - start_time}')
        return dataset


def padded_collate(batch):
    """
    Функция, которая делает padding в батче; должна передаваться в torch.Dataloader в collate_fn
    """
    new_x_ = defaultdict(list)
    for x, _ in batch:
        for k, v in x.items():
            if not (isinstance(v, int) or isinstance(v, float)):
                new_x_[k].append(v)

    lengths = torch.IntTensor([len(e) for e in next(iter(new_x_.values()))
                               if not (isinstance(e, int) or isinstance(e, float))])

    new_x = {k: torch.nn.utils.rnn.pad_sequence(v, batch_first=True) for k, v in new_x_.items()
             if not (isinstance(v, int) or isinstance(v, float))}
    new_y = np.array([y for _, y in batch])
    if new_y.dtype.kind in ('i', 'f'):
        new_y = torch.from_numpy(new_y).float()  # зависит от таргета

    return PaddedBatch(new_x, lengths), new_y
