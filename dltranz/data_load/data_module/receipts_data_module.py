from typing import List, Dict

import torch.utils.data
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split

from dltranz.data_load.sequence_bucketing_dataset import SequenceBucketingFilterDataset, padded_collate
from dltranz.data_load.weighted_dataset import WeightedDataset


class ReceiptsTrainDataModule(pl.LightningDataModule):
    """
    A.k.a RegressionTrainDataModule in my notebooks
    """
    def __init__(self,
                 dataset: List[Dict],
                 test_dataset: List[Dict],
                 val_size: float = 0.1,
                 train_num_workers: int = 0,
                 train_batch_size: int = 512,
                 valid_num_workers: int = 0,
                 valid_batch_size: int = 512,
                 random_state=42,
                 weighted=False
                 ):
        """
        Parameters:
            weighted: нужно ли семплировать классы по их весам
        """
        super().__init__()
        self.dataset = dataset
        self.test = test_dataset
        self.val_size = val_size
        self.train_num_workers = train_num_workers
        self.valid_num_workers = valid_num_workers
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.weighted = weighted

        train, valid = train_test_split(dataset, test_size=val_size, random_state=random_state)
        self.train_data = train
        self.valid_data = valid
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        if not self.weighted:
            self.train_dataset = SequenceBucketingFilterDataset(self.train_data)
        else:
            self.train_dataset = WeightedDataset(self.train_data)
        self.valid_dataset = SequenceBucketingFilterDataset(self.valid_data)
        self.test_dataset = SequenceBucketingFilterDataset(self.test)

    def train_dataloader(self):
        if not self.weighted:
            return torch.utils.data.DataLoader(
                dataset=self.train_dataset,
                collate_fn=padded_collate,
                num_workers=self.train_num_workers,
                batch_size=self.train_batch_size
            )
        else:
            return torch.utils.data.DataLoader(
                dataset=self.train_dataset,
                collate_fn=padded_collate,
                num_workers=self.train_num_workers,
                batch_size=self.train_batch_size,
                sampler=WeightedDataset.get_weighted_random_sampler(self.train_data)
            )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.valid_dataset,
            collate_fn=padded_collate,
            num_workers=self.valid_num_workers,
            batch_size=self.valid_batch_size
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_dataset,
            collate_fn=padded_collate,
            num_workers=self.valid_num_workers,
            batch_size=self.valid_batch_size
        )