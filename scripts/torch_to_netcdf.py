import xarray as xr

# custom functions
from functions import *

# suppress warnings
import warnings
warnings.filterwarnings("ignore")


# paths
path_7wr = '../logs/t2m_ERA5/multiruns/7wr_2011'
path_seas5_idx = '../data/SEAS5_monthly_z500_7wr_noa_bias1981-2010_201101-202412.nc'

# load model, anomalies and indexes
torch_model, datamodule, config = get_torch_models_infos(path_7wr)
anom  = xr.open_dataarray(config['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
idxs  = xr.open_dataarray(config['data']['indexes_paths'][0]).sel(mode=slice(1, config['data']['num_indexes'][0]))
model = get_models_out(torch_model, idxs, anom, datamodule)

# load model with seas5 indexes
idxs_seas5  = xr.open_dataarray(path_seas5_idx)
model_seas5 = models_with_SEAS5_indexes(torch_model, idxs_seas5, anom, datamodule)

# save
model.to_netcdf('models/model_7wr_2mt.nc')
anom.to_netcdf('models/anom_2mt.nc')
idxs.to_netcdf('models/7wr.nc')
model_seas5.to_netcdf('models/model_7wr_seas5_2mt.nc')
