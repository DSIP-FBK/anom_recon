import torch
import numpy as np
import pandas as pd
import xarray as xr
import xeofs as xe
from datetime import datetime
import einops
from omegaconf import OmegaConf
from torch.utils.data import Dataset

class IndexAnomaly(Dataset):
    def __init__(
            self, 
            indexes_paths: list,
            anomalies_path: str,
            land_sea_mask_path: str,
            orography_path: str,
            num_indexes: list,
            num_pca: int,
            train_last_date: str
            ):
        self.indexes_paths = indexes_paths
        self.anomalies_path = anomalies_path
        self.land_sea_mask_path = land_sea_mask_path
        self.orography_path = orography_path
        self.num_indexes = num_indexes
        self.num_pca = num_pca
        self.train_last_date = train_last_date

        # --------
        # indexes
        # --------
        indexes_list = list()
        for n, path in zip(self.num_indexes, self.indexes_paths):
            indexes_list.append(
                    xr.open_dataarray(path, engine='netcdf4').sel(mode=slice(None, n))
            )
        
        # concatenate and reformat the 'mode' column
        self.indexes = xr.concat(indexes_list, dim='mode', coords='minimal', compat='override')
        self.indexes['mode'] = np.arange(1, np.sum(num_indexes) + 1)

        # nan to zero
        # self.indexes = self.indexes.fillna(0)

        # normalize indexes (on the training set only)
        # self.indexes = (self.indexes - self.indexes.sel(time=slice(None, train_last_date)).mean()) #/ self.indexes.sel(time=slice(None, train_last_date).std()

        # ----------
        # anomalies
        # ----------
        self.anomalies = xr.open_dataarray(anomalies_path, engine='netcdf4')

        # normalize anomalies (on the training set only)
        # self.anomalies = self.anomalies - self.anomalies.sel(time=slice(None, train_last_date)).mean() #/ self.anomalies.std()
        
        # reduce to common time
        self.start     = max(self.indexes.time.min(), self.anomalies.time.min())
        self.end       = min(self.indexes.time.max(), self.anomalies.time.max())
        self.indexes   = self.indexes.sel(time=slice(self.start, self.end))
        self.anomalies = self.anomalies.sel(time=slice(self.start, self.end))
        
        # latitude, longitude and static data
        self.static_data = self.get_static()


    def __len__(self):
        return self.anomalies.time.count().values

    def __getitem__(self, idx):
        
        index = torch.tensor(
            self.indexes[:, idx].data.flatten(),
            dtype=torch.float32
            )
            
        # anomaly (ground truth)
        anom = self.anomalies[idx]

        # date (year, month)
        year  = torch.tensor(anom.time.dt.year.data - self.start.dt.year.data, dtype=torch.float32)
        month = torch.tensor(anom.time.dt.month.data, dtype=torch.float32)

        return (
            (year, month, index, self.static_data),
            torch.tensor(anom.data, dtype=torch.float32),
        )   
    
    def get_static(self):

        # load lsm, dtm and lat, lon
        lsm_dset = xr.load_dataarray(self.land_sea_mask_path)
        lats = torch.from_numpy(normalize_lat(lsm_dset.latitude.data)).to(torch.float32)
        lons = torch.from_numpy(normalize_lon(lsm_dset.longitude.data)).to(torch.float32)
        lsm = torch.from_numpy(lsm_dset.values).to(torch.float32)
        dtm = torch.from_numpy(normalize_dtm(xr.load_dataarray(self.orography_path)).data).to(torch.float32)
        
        # reshape lats, lons, lsm and dtm
        h, w = len(lats), len(lons)
        reshaped_lats = einops.repeat(lats, 'h -> h w', w=w)
        reshaped_lons = einops.repeat(lons, 'w -> h w', h=h)

        # concatenate lats, lons, lsm and dtm
        return torch.stack((reshaped_lats, reshaped_lons, lsm, dtm))

# we can implement sin and cosine for latitude
def normalize_lon(lons):
    return lons / np.max(abs(lons))

def normalize_lat(lats):
    return lats / np.max(abs(lats))

def normalize_dtm(dtm, pow=1./3):
    dtm_scaled = xr.where(dtm >= 0, dtm ** pow, -abs(dtm) ** pow)
    return (dtm_scaled - dtm_scaled.mean()) / dtm_scaled.std()

def datetime_from_index_row(row):
    year = row.name[0]
    month = row.name[1]
    date_str = "%s/%s" % (year, month)
    if isinstance(month, str):
        return datetime.strptime(date_str,'%Y/%b')
    return datetime.strptime(date_str, "%Y/%m")