from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from src.data.components.anomrecon_dataset import IndexAnomaly


class AnomReconDataModule(LightningDataModule):
    """Data Module for AnomRec model"""

    def __init__(
            self, 
            indexes_paths: list = ['../data',],
            anomalies_path: str ='../data',
            land_sea_mask_path: str ='../data',
            orography_path: str ='../data',
            months: list = [],
            num_indexes: list = [15,],
            num_pca: int = 0,
            train_last_date: str = '2010-12-01',
            val_last_date: str = '2014-12-01',
            batch_size: int = 1,
            num_workers: int = 1,
            )-> None:
    
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: str) -> None:

        if not self.data_train or not self.data_val or not self.data_test:
            dataset = IndexAnomaly(
                indexes_paths=self.hparams.indexes_paths,
                anomalies_path=self.hparams.anomalies_path,
                land_sea_mask_path=self.hparams.land_sea_mask_path,
                orography_path=self.hparams.orography_path,
                num_indexes=self.hparams.num_indexes,
                num_pca=self.hparams.num_pca,
                train_last_date=self.hparams.train_last_date,
                val_last_date=self.hparams.val_last_date
            )

            time = dataset.anomalies.time
            train_last_date = pd.to_datetime(self.hparams.train_last_date, format='%Y-%m-%d')
            val_last_date   = pd.to_datetime(self.hparams.val_last_date, format='%Y-%m-%d')
            
            train_start_idx = 0
            train_end_idx   = sum(time.data <= train_last_date)
            train_indexes   = np.arange(
                train_start_idx,
                train_end_idx
            )
            if self.hparams.months != []:
                train_indexes = train_indexes[np.isin(time[train_indexes].dt.month, self.hparams.months)]

            self.data_train = torch.utils.data.Subset(
                dataset=dataset,
                indices=train_indexes
            )

            val_start_idx = train_end_idx
            val_end_idx   = sum(time.data <= val_last_date)
            val_indexes   = np.arange(
                val_start_idx,
                val_end_idx
            )
            if self.hparams.months != []:
                val_indexes = val_indexes[np.isin(time[val_indexes].dt.month, self.hparams.months)]
            
            self.data_val = torch.utils.data.Subset(
                dataset=dataset,
                indices=val_indexes
            )
            
            test_start_idx = val_end_idx
            test_end_idx   = len(time)
            test_indexes   = np.arange(
                test_start_idx,
                test_end_idx
            )
            if self.hparams.months != []:
                test_indexes = test_indexes[np.isin(time[test_indexes].dt.month, self.hparams.months)]
            
            self.data_test = torch.utils.data.Subset(
                dataset=dataset,
                indices=test_indexes
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=1,
            num_workers=1,
            shuffle=False,
        )
        