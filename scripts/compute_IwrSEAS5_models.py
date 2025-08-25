import argparse
import numpy as np
import xarray as xr

# custom functions
from functions import *

# suppress warnings
import warnings
warnings.filterwarnings("ignore")


# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--Iwr_SEAS5", type=str, help="path to the NetCDF containing the WR index computed from SEAS5 forecasts")
parser.add_argument("--seas5_temp", type=str, help="path to the NetCDF containing the SEAS5 temperature normalized anomalies forecasts")
parser.add_argument("--seas5_prec", type=str, help="path to the NetCDF containing the SEAS5 precipitation normalized anomalies forecasts")
parser.add_argument("--model_temp", type=str, help="path to the folder containing the model trained on temperature")
parser.add_argument("--model_prec", type=str, help="path to the folder containing the model trained on precipitation")
parser.add_argument("--start", type=str, help="start date of the analysis")
args = parser.parse_args()

# seasons
winter_months = (12,1,2)
summer_months = (6,7,8)


# ---------------------
# Load models and SEAS5
# ---------------------
print('Loading models and SEAS5...')

# WR computed from SEAS5
idxs_seas5_7wr = xr.open_dataarray(args.Iwr_SEAS5)

# models on temprature
torch_models, datamodule, config = get_torch_models_infos(args.model_temp)
anomT                = xr.open_dataarray(config['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
model_seas5_7wrT     = models_with_SEAS5_indexes(torch_models, idxs_seas5_7wr, anomT, datamodule)

# models on precipitation
torch_models, datamodule, config = get_torch_models_infos(args.model_prec)
anomP                = xr.open_dataarray(config['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
model_seas5_7wrP     = models_with_SEAS5_indexes(torch_models, idxs_seas5_7wr, anomP, datamodule)

# save
model_seas5_7wrT.to_netcdf(f'data/model_seas5_7wrT_{args.start}.nc')
model_seas5_7wrP.to_netcdf(f'data/model_seas5_7wrP_{args.start}.nc')