import argparse
import numpy as np
import xarray as xr

# plotting
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

# custom functions
from functions import *

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("-wr7_temp", type=str, help="path to the folder containing the model trained with 7WR to reconstruct temperature" )
parser.add_argument("-wr7_prec", type=str, help="path to the folder containing the model trained with 7WR to reconstruct precipitation" )

parser.add_argument("-wr4_temp_winter", type=str, help="path to the folder containing the model trained with winter's 4WR to reconstruct temperature" )
parser.add_argument("-wr4_temp_summer", type=str, help="path to the folder containing the model trained with summers's 4WR to reconstruct temperature" )
parser.add_argument("-wr4_prec_winter", type=str, help="path to the folder containing the model trained with winter's 4WR to reconstruct precipitation" )
parser.add_argument("-wr4_prec_summer", type=str, help="path to the folder containing the model trained with summer's 4WR to reconstruct precipitation" )

parser.add_argument("-NAO_temp_winter", type=str, help="path to the folder containing the model trained with winter's NAO to reconstruct temperature" )
parser.add_argument("-NAO_temp_summer", type=str, help="path to the folder containing the model trained with summer's NAO to reconstruct temperature" )
parser.add_argument("-NAO_prec_winter", type=str, help="path to the folder containing the model trained with winter's NAO to reconstruct precipitation" )
parser.add_argument("-NAO_prec_summer", type=str, help="path to the folder containing the model trained with summer's NAO to reconstruct precipitation" )

parser.add_argument("-wr0_temp", type=str, help="path to the folder containing the model trained with 0WR to reconstruct temperature" )
parser.add_argument("-wr0_prec", type=str, help="path to the folder containing the model trained with 0WR to reconstruct precipitation" )
args = parser.parse_args()

# parameters
lat_min, lat_max = 35, 70
lon_min, lon_max = -20, 30
winter_months = (12,1,2)
summer_months = (6,7,8)

# land sea mask
lsm = xr.open_dataarray('../data/lsm_regrid_shift_europe.nc')

# ---------------------
# Load models and SEAS5
# ---------------------
print('Loading models...')

# 7WR on temprature
torch_model, datamodule, config = get_torch_models_infos(args.wr7_temp)
idxs_7wr_temp          = xr.open_dataarray(config['data']['indexes_paths'][0]).sel(mode=slice(1, config['data']['num_indexes'][0]))
anom_7wr_temp          = xr.open_dataarray(config['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
model_7wr_temp         = get_models_out(torch_model, idxs_7wr_temp, anom_7wr_temp, datamodule)

# 7WR on precipitation
torch_model, datamodule, config = get_torch_models_infos(args.wr7_prec)
idxs_7wr_prec          = xr.open_dataarray(config['data']['indexes_paths'][0]).sel(mode=slice(1, config['data']['num_indexes'][0]))
anom_7wr_prec          = xr.open_dataarray(config['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
model_7wr_prec         = get_models_out(torch_model, idxs_7wr_prec, anom_7wr_prec, datamodule)

# 4WR on temprature (winter)
torch_model, datamodule, config = get_torch_models_infos(args.wr4_temp_winter)
idxs_4wr_temp_winter            = xr.open_dataarray(config['data']['indexes_paths'][0]).sel(mode=slice(1, config['data']['num_indexes'][0]))
anom_4wr_temp_winter            = xr.open_dataarray(config['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
model_4wr_temp_winter           = get_models_out(torch_model, idxs_4wr_temp_winter, anom_4wr_temp_winter, datamodule)

# 4WR on temprature (summer)
torch_model, datamodule, config = get_torch_models_infos(args.wr4_temp_summer)
idxs_4wr_temp_summer            = xr.open_dataarray(config['data']['indexes_paths'][0]).sel(mode=slice(1, config['data']['num_indexes'][0]))
anom_4wr_temp_summer            = xr.open_dataarray(config['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
model_4wr_temp_summer           = get_models_out(torch_model, idxs_4wr_temp_summer, anom_4wr_temp_summer, datamodule)

# 4WR on precipitation (winter)
torch_model, datamodule, config = get_torch_models_infos(args.wr4_prec_winter)
idxs_4wr_prec_winter            = xr.open_dataarray(config['data']['indexes_paths'][0]).sel(mode=slice(1, config['data']['num_indexes'][0]))
anom_4wr_prec_winter            = xr.open_dataarray(config['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
model_4wr_prec_winter           = get_models_out(torch_model, idxs_4wr_prec_winter, anom_4wr_prec_winter, datamodule)

# 4WR on precipitation (summer)
torch_model, datamodule, config = get_torch_models_infos(args.wr4_prec_summer)
idxs_4wr_prec_summer            = xr.open_dataarray(config['data']['indexes_paths'][0]).sel(mode=slice(1, config['data']['num_indexes'][0]))
anom_4wr_prec_summer            = xr.open_dataarray(config['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
model_4wr_prec_summer           = get_models_out(torch_model, idxs_4wr_prec_summer, anom_4wr_prec_summer, datamodule)

# NAO on temprature (winter)
torch_model, datamodule, config = get_torch_models_infos(args.NAO_temp_winter)
idxs_NAO_temp_winter            = xr.open_dataarray(config['data']['indexes_paths'][0]).sel(mode=slice(1, config['data']['num_indexes'][0]))
anom_NAO_temp_winter            = xr.open_dataarray(config['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
model_NAO_temp_winter           = get_models_out(torch_model, idxs_NAO_temp_winter, anom_NAO_temp_winter, datamodule)

# NAO on temprature (summer)
torch_model, datamodule, config = get_torch_models_infos(args.NAO_temp_summer)
idxs_NAO_temp_summer            = xr.open_dataarray(config['data']['indexes_paths'][0]).sel(mode=slice(1, config['data']['num_indexes'][0]))
anom_NAO_temp_summer            = xr.open_dataarray(config['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
model_NAO_temp_summer           = get_models_out(torch_model, idxs_NAO_temp_summer, anom_NAO_temp_summer, datamodule)

# NAO on precipitation (winter)
torch_model, datamodule, config = get_torch_models_infos(args.NAO_prec_winter)
idxs_NAO_prec_winter            = xr.open_dataarray(config['data']['indexes_paths'][0]).sel(mode=slice(1, config['data']['num_indexes'][0]))
anom_NAO_prec_winter            = xr.open_dataarray(config['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
model_NAO_prec_winter           = get_models_out(torch_model, idxs_NAO_prec_winter, anom_NAO_prec_winter, datamodule)

# NAO on precipitation (summer)
torch_model, datamodule, config = get_torch_models_infos(args.NAO_prec_summer)
idxs_NAO_prec_summer            = xr.open_dataarray(config['data']['indexes_paths'][0]).sel(mode=slice(1, config['data']['num_indexes'][0]))
anom_NAO_prec_summer            = xr.open_dataarray(config['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
model_NAO_prec_summer           = get_models_out(torch_model, idxs_NAO_prec_summer, anom_NAO_prec_summer, datamodule)

# 0WR on temprature
torch_model, datamodule, config = get_torch_models_infos(args.wr0_temp)
idxs_0wr_temp          = xr.open_dataarray(config['data']['indexes_paths'][0]).sel(mode=slice(1, config['data']['num_indexes'][0]))
anom_0wr_temp          = xr.open_dataarray(config['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
model_0wr_temp         = get_models_out(torch_model, idxs_0wr_temp, anom_0wr_temp, datamodule)

# 0WR on precipitation
torch_model, datamodule, config = get_torch_models_infos(args.wr0_prec)
idxs_0wr_prec          = xr.open_dataarray(config['data']['indexes_paths'][0]).sel(mode=slice(1, config['data']['num_indexes'][0]))
anom_0wr_prec          = xr.open_dataarray(config['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
model_0wr_prec         = get_models_out(torch_model, idxs_0wr_prec, anom_0wr_prec, datamodule)

# crop to europe
anom_7wr_temp = anom_7wr_temp.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
model_7wr_temp = model_7wr_temp.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
anom_7wr_prec = anom_7wr_prec.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
model_7wr_prec = model_7wr_prec.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))

anom_4wr_temp_winter = anom_4wr_temp_winter.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
model_4wr_temp_winter = model_4wr_temp_winter.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
anom_4wr_prec_winter = anom_4wr_prec_winter.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
model_4wr_prec_winter = model_4wr_prec_winter.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))

anom_NAO_temp_winter = anom_NAO_temp_winter.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
model_NAO_temp_winter = model_NAO_temp_winter.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
anom_NAO_prec_winter = anom_NAO_prec_winter.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
model_NAO_prec_winter = model_NAO_prec_winter.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))

anom_0wr_temp = anom_0wr_temp.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
model_0wr_temp = model_0wr_temp.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
anom_0wr_prec = anom_0wr_prec.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
model_0wr_prec = model_0wr_prec.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))

# seasons
model_7wr_temp_winter  = model_7wr_temp[np.isin(model_7wr_temp.time.dt.month, winter_months)]
anom_7wr_temp_winter   = anom_7wr_temp[np.isin(anom_7wr_temp.time.dt.month, winter_months)]
model_7wr_temp_summer  = model_7wr_temp[np.isin(model_7wr_temp.time.dt.month, summer_months)]
anom_7wr_temp_summer   = anom_7wr_temp[np.isin(anom_7wr_temp.time.dt.month, summer_months)]

model_7wr_prec_winter  = model_7wr_prec[np.isin(model_7wr_prec.time.dt.month, winter_months)]
anom_7wr_prec_winter   = anom_7wr_prec[np.isin(anom_7wr_prec.time.dt.month, winter_months)]
model_7wr_prec_summer  = model_7wr_prec[np.isin(model_7wr_prec.time.dt.month, summer_months)]
anom_7wr_prec_summer   = anom_7wr_prec[np.isin(anom_7wr_prec.time.dt.month, summer_months)]

anom_4wr_temp_winter   = anom_4wr_temp_winter[np.isin(anom_4wr_temp_winter.time.dt.month, winter_months)]
anom_4wr_temp_summer   = anom_4wr_temp_summer[np.isin(anom_4wr_temp_summer.time.dt.month, summer_months)]
anom_4wr_prec_winter   = anom_4wr_prec_winter[np.isin(anom_4wr_prec_winter.time.dt.month, winter_months)]
anom_4wr_prec_summer   = anom_4wr_prec_summer[np.isin(anom_4wr_prec_summer.time.dt.month, summer_months)]

anom_NAO_temp_winter   = anom_NAO_temp_winter[np.isin(anom_NAO_temp_winter.time.dt.month, winter_months)]
anom_NAO_temp_summer   = anom_NAO_temp_summer[np.isin(anom_NAO_temp_summer.time.dt.month, summer_months)]
anom_NAO_prec_winter   = anom_NAO_prec_winter[np.isin(anom_NAO_prec_winter.time.dt.month, winter_months)]
anom_NAO_prec_summer   = anom_NAO_prec_summer[np.isin(anom_NAO_prec_summer.time.dt.month, summer_months)]

model_0wr_temp_winter  = model_0wr_temp[np.isin(model_0wr_temp.time.dt.month, winter_months)]
anom_0wr_temp_winter   = anom_0wr_temp[np.isin(anom_0wr_temp.time.dt.month, winter_months)]
model_0wr_temp_summer  = model_0wr_temp[np.isin(model_0wr_temp.time.dt.month, summer_months)]
anom_0wr_temp_summer   = anom_0wr_temp[np.isin(anom_0wr_temp.time.dt.month, summer_months)]

model_0wr_prec_winter  = model_0wr_prec[np.isin(model_0wr_prec.time.dt.month, winter_months)]
anom_0wr_prec_winter   = anom_0wr_prec[np.isin(anom_0wr_prec.time.dt.month, winter_months)]
model_0wr_prec_summer  = model_0wr_prec[np.isin(model_0wr_prec.time.dt.month, summer_months)]
anom_0wr_prec_summer   = anom_0wr_prec[np.isin(anom_0wr_prec.time.dt.month, summer_months)]

# reduce all variables to common time-range
start, end            = '2011', '2024'
anom_7wr_temp_winter  = anom_7wr_temp_winter.sel(time=slice(start, end))
anom_4wr_temp_winter  = anom_4wr_temp_winter.sel(time=slice(start, end))
anom_NAO_temp_winter  = anom_NAO_temp_winter.sel(time=slice(start, end))
anom_0wr_temp_winter  = anom_0wr_temp_winter.sel(time=slice(start, end))
anom_7wr_prec_winter  = anom_7wr_prec_winter.sel(time=slice(start, end))
anom_4wr_prec_winter  = anom_4wr_prec_winter.sel(time=slice(start, end))
anom_NAO_prec_winter  = anom_NAO_prec_winter.sel(time=slice(start, end))
anom_0wr_prec_winter  = anom_0wr_prec_winter.sel(time=slice(start, end))
model_7wr_temp_winter = model_7wr_temp_winter.sel(time=slice(start, end))
model_4wr_temp_winter = model_4wr_temp_winter.sel(time=slice(start, end))
model_NAO_temp_winter = model_NAO_temp_winter.sel(time=slice(start, end))
model_0wr_temp_winter = model_0wr_temp_winter.sel(time=slice(start, end))
model_7wr_prec_winter = model_7wr_prec_winter.sel(time=slice(start, end))
model_4wr_prec_winter = model_4wr_prec_winter.sel(time=slice(start, end))
model_NAO_prec_winter = model_NAO_prec_winter.sel(time=slice(start, end))
model_0wr_prec_winter = model_0wr_prec_winter.sel(time=slice(start, end))

anom_7wr_temp_summer  = anom_7wr_temp_summer.sel(time=slice(start, end))
anom_4wr_temp_summer  = anom_4wr_temp_summer.sel(time=slice(start, end))
anom_NAO_temp_summer  = anom_NAO_temp_summer.sel(time=slice(start, end))
anom_0wr_temp_summer  = anom_0wr_temp_summer.sel(time=slice(start, end))
anom_7wr_prec_summer  = anom_7wr_prec_summer.sel(time=slice(start, end))
anom_4wr_prec_summer  = anom_4wr_prec_summer.sel(time=slice(start, end))
anom_NAO_prec_summer  = anom_NAO_prec_summer.sel(time=slice(start, end))
anom_0wr_prec_summer  = anom_0wr_prec_summer.sel(time=slice(start, end))
model_7wr_temp_summer = model_7wr_temp_summer.sel(time=slice(start, end))
model_4wr_temp_summer = model_4wr_temp_summer.sel(time=slice(start, end))
model_NAO_temp_summer = model_NAO_temp_summer.sel(time=slice(start, end))
model_0wr_temp_summer = model_0wr_temp_summer.sel(time=slice(start, end))
model_7wr_prec_summer = model_7wr_prec_summer.sel(time=slice(start, end))
model_4wr_prec_summer = model_4wr_prec_summer.sel(time=slice(start, end))
model_NAO_prec_summer = model_NAO_prec_summer.sel(time=slice(start, end))
model_0wr_prec_summer = model_0wr_prec_summer.sel(time=slice(start, end))


# --------------
# Compute skills
# --------------
print('Computing skills...')

# mean absolute error
model_7wrT_winter_MAE = abs(anom_7wr_temp_winter - model_7wr_temp_winter.mean(dim='number')).mean(dim='time')
model_7wrT_summer_MAE = abs(anom_7wr_temp_summer - model_7wr_temp_summer.mean(dim='number')).mean(dim='time')
model_4wrT_winter_MAE = abs(anom_4wr_temp_winter - model_4wr_temp_winter.mean(dim='number')).mean(dim='time')
model_4wrT_summer_MAE = abs(anom_4wr_temp_summer - model_4wr_temp_summer.mean(dim='number')).mean(dim='time')
model_NAOT_winter_MAE = abs(anom_NAO_temp_winter - model_NAO_temp_winter.mean(dim='number')).mean(dim='time')
model_NAOT_summer_MAE = abs(anom_NAO_temp_summer - model_NAO_temp_summer.mean(dim='number')).mean(dim='time')
model_0wrT_winter_MAE = abs(anom_0wr_temp_winter - model_0wr_temp_winter.mean(dim='number')).mean(dim='time')
model_0wrT_summer_MAE = abs(anom_0wr_temp_summer - model_0wr_temp_summer.mean(dim='number')).mean(dim='time')

model_7wrP_winter_MAE = abs(anom_7wr_prec_winter - model_7wr_prec_winter.mean(dim='number')).mean(dim='time') * 100 # m to cm
model_7wrP_summer_MAE = abs(anom_7wr_prec_summer - model_7wr_prec_summer.mean(dim='number')).mean(dim='time') * 100 # m to cm
model_4wrP_winter_MAE = abs(anom_4wr_prec_winter - model_4wr_prec_winter.mean(dim='number')).mean(dim='time') * 100 # m to cm
model_4wrP_summer_MAE = abs(anom_4wr_prec_summer - model_4wr_prec_summer.mean(dim='number')).mean(dim='time') * 100 # m to cm
model_NAOP_winter_MAE = abs(anom_NAO_prec_winter - model_NAO_prec_winter.mean(dim='number')).mean(dim='time') * 100 # m to cm
model_NAOP_summer_MAE = abs(anom_NAO_prec_summer - model_NAO_prec_summer.mean(dim='number')).mean(dim='time') * 100 # m to cm
model_0wrP_winter_MAE = abs(anom_0wr_prec_winter - model_0wr_prec_winter.mean(dim='number')).mean(dim='time') * 100 # m to cm
model_0wrP_summer_MAE = abs(anom_0wr_prec_summer - model_0wr_prec_summer.mean(dim='number')).mean(dim='time') * 100 # m to cm

# anomaly correlation coefficient
model_7wrT_winter_ACC = xr.corr(anom_7wr_temp_winter, model_7wr_temp_winter.mean(dim='number'), dim='time')
model_7wrT_summer_ACC = xr.corr(anom_7wr_temp_summer, model_7wr_temp_summer.mean(dim='number'), dim='time')
model_4wrT_winter_ACC = xr.corr(anom_4wr_temp_winter, model_4wr_temp_winter.mean(dim='number'), dim='time')
model_4wrT_summer_ACC = xr.corr(anom_4wr_temp_summer, model_4wr_temp_summer.mean(dim='number'), dim='time')
model_NAOT_winter_ACC = xr.corr(anom_NAO_temp_winter, model_NAO_temp_winter.mean(dim='number'), dim='time')
model_NAOT_summer_ACC = xr.corr(anom_NAO_temp_summer, model_NAO_temp_summer.mean(dim='number'), dim='time')
model_0wrT_winter_ACC = xr.corr(anom_0wr_temp_winter, model_0wr_temp_winter.mean(dim='number'), dim='time')
model_0wrT_summer_ACC = xr.corr(anom_0wr_temp_summer, model_0wr_temp_summer.mean(dim='number'), dim='time')

model_7wrP_winter_ACC = xr.corr(anom_7wr_prec_winter, model_7wr_prec_winter.mean(dim='number'), dim='time')
model_7wrP_summer_ACC = xr.corr(anom_7wr_prec_summer, model_7wr_prec_summer.mean(dim='number'), dim='time')
model_4wrP_winter_ACC = xr.corr(anom_4wr_prec_winter, model_4wr_prec_winter.mean(dim='number'), dim='time')
model_4wrP_summer_ACC = xr.corr(anom_4wr_prec_summer, model_4wr_prec_summer.mean(dim='number'), dim='time')
model_NAOP_winter_ACC = xr.corr(anom_NAO_prec_winter, model_NAO_prec_winter.mean(dim='number'), dim='time')
model_NAOP_summer_ACC = xr.corr(anom_NAO_prec_summer, model_NAO_prec_summer.mean(dim='number'), dim='time')
model_0wrP_winter_ACC = xr.corr(anom_0wr_prec_winter, model_0wr_prec_winter.mean(dim='number'), dim='time')
model_0wrP_summer_ACC = xr.corr(anom_0wr_prec_summer, model_0wr_prec_summer.mean(dim='number'), dim='time')

# coefficient of efficacy
model_7wrT_winter_CE = get_ce(anom_7wr_temp_winter, model_7wr_temp_winter.mean(dim='number'))
model_7wrT_summer_CE = get_ce(anom_7wr_temp_summer, model_7wr_temp_summer.mean(dim='number'))
model_4wrT_winter_CE = get_ce(anom_4wr_temp_winter, model_4wr_temp_winter.mean(dim='number'))
model_4wrT_summer_CE = get_ce(anom_4wr_temp_summer, model_4wr_temp_summer.mean(dim='number'))
model_NAOT_winter_CE = get_ce(anom_NAO_temp_winter, model_NAO_temp_winter.mean(dim='number'))
model_NAOT_summer_CE = get_ce(anom_NAO_temp_summer, model_NAO_temp_summer.mean(dim='number'))
model_0wrT_winter_CE = get_ce(anom_0wr_temp_winter, model_0wr_temp_winter.mean(dim='number'))
model_0wrT_summer_CE = get_ce(anom_0wr_temp_summer, model_0wr_temp_summer.mean(dim='number'))

model_7wrP_winter_CE = get_ce(anom_7wr_prec_winter, model_7wr_prec_winter.mean(dim='number'))
model_7wrP_summer_CE = get_ce(anom_7wr_prec_summer, model_7wr_prec_summer.mean(dim='number'))
model_4wrP_winter_CE = get_ce(anom_4wr_prec_winter, model_4wr_prec_winter.mean(dim='number'))
model_4wrP_summer_CE = get_ce(anom_4wr_prec_summer, model_4wr_prec_summer.mean(dim='number'))
model_NAOP_winter_CE = get_ce(anom_NAO_prec_winter, model_NAO_prec_winter.mean(dim='number'))
model_NAOP_summer_CE = get_ce(anom_NAO_prec_summer, model_NAO_prec_summer.mean(dim='number'))
model_0wrP_winter_CE = get_ce(anom_0wr_prec_winter, model_0wr_prec_winter.mean(dim='number'))
model_0wrP_summer_CE = get_ce(anom_0wr_prec_summer, model_0wr_prec_summer.mean(dim='number'))

# models uncertainty MAE
mae = abs(anom_7wr_temp_winter - model_7wr_temp_winter).mean(dim='time').median(dim=['lat', 'lon'])
model_7wrT_winter_MAE_err = (mae.max() - mae.min()).values / 2 * 100
mae = abs(anom_4wr_temp_winter - model_4wr_temp_winter).mean(dim='time').median(dim=['lat', 'lon'])
model_4wrT_winter_MAE_err = (mae.max() - mae.min()).values / 2 * 100
mae = abs(anom_NAO_temp_winter - model_NAO_temp_winter).mean(dim='time').median(dim=['lat', 'lon'])
model_NAOT_winter_MAE_err = (mae.max() - mae.min()).values / 2 * 100
mae = abs(anom_0wr_temp_winter - model_0wr_temp_winter).mean(dim='time').median(dim=['lat', 'lon'])
model_0wrT_winter_MAE_err = (mae.max() - mae.min()).values / 2 * 100
mae = abs(anom_7wr_temp_summer - model_7wr_temp_summer).mean(dim='time').median(dim=['lat', 'lon'])
model_7wrT_summer_MAE_err = (mae.max() - mae.min()).values / 2 * 100
mae = abs(anom_4wr_temp_summer - model_4wr_temp_summer).mean(dim='time').median(dim=['lat', 'lon'])
model_4wrT_summer_MAE_err = (mae.max() - mae.min()).values / 2 * 100
mae = abs(anom_NAO_temp_summer - model_NAO_temp_summer).mean(dim='time').median(dim=['lat', 'lon'])
model_NAOT_summer_MAE_err = (mae.max() - mae.min()).values / 2 * 100
mae = abs(anom_0wr_temp_summer - model_0wr_temp_summer).mean(dim='time').median(dim=['lat', 'lon'])
model_0wrT_summer_MAE_err = (mae.max() - mae.min()).values / 2 * 100

mae = abs(anom_7wr_prec_winter - model_7wr_prec_winter).mean(dim='time').median(dim=['lat', 'lon']) * 100 # m to cm
model_7wrP_winter_MAE_err = (mae.max() - mae.min()).values / 2 * 100
mae = abs(anom_4wr_prec_winter - model_4wr_prec_winter).mean(dim='time').median(dim=['lat', 'lon']) * 100 # m to cm
model_4wrP_winter_MAE_err = (mae.max() - mae.min()).values / 2 * 100
mae = abs(anom_NAO_prec_winter - model_NAO_prec_winter).mean(dim='time').median(dim=['lat', 'lon']) * 100 # m to cm
model_NAOP_winter_MAE_err = (mae.max() - mae.min()).values / 2 * 100
mae = abs(anom_0wr_prec_winter - model_0wr_prec_winter).mean(dim='time').median(dim=['lat', 'lon']) * 100 # m to cm
model_0wrP_winter_MAE_err = (mae.max() - mae.min()).values / 2 * 100
mae = abs(anom_7wr_prec_summer - model_7wr_prec_summer).mean(dim='time').median(dim=['lat', 'lon']) * 100 # m to cm
model_7wrP_summer_MAE_err = (mae.max() - mae.min()).values / 2 * 100
mae = abs(anom_4wr_prec_summer - model_4wr_prec_summer).mean(dim='time').median(dim=['lat', 'lon']) * 100 # m to cm
model_4wrP_summer_MAE_err = (mae.max() - mae.min()).values / 2 * 100
mae = abs(anom_NAO_prec_summer - model_NAO_prec_summer).mean(dim='time').median(dim=['lat', 'lon']) * 100 # m to cm
model_NAOP_summer_MAE_err = (mae.max() - mae.min()).values / 2 * 100
mae = abs(anom_0wr_prec_summer - model_0wr_prec_summer).mean(dim='time').median(dim=['lat', 'lon']) * 100 # m to cm
model_0wrP_summer_MAE_err = (mae.max() - mae.min()).values / 2 * 100

# models uncertainty ACC
corr = xr.corr(anom_7wr_temp_winter, model_7wr_temp_winter, dim='time').median(dim=['lat', 'lon'])
model_7wrT_winter_ACC_err = (corr.max() - corr.min()).values / 2 * 100
corr = xr.corr(anom_4wr_temp_winter, model_4wr_temp_winter, dim='time').median(dim=['lat', 'lon'])
model_4wrT_winter_ACC_err = (corr.max() - corr.min()).values / 2 * 100
corr = xr.corr(anom_NAO_temp_winter, model_NAO_temp_winter, dim='time').median(dim=['lat', 'lon'])
model_NAOT_winter_ACC_err = (corr.max() - corr.min()).values / 2 * 100
corr = xr.corr(anom_0wr_temp_winter, model_0wr_temp_winter, dim='time').median(dim=['lat', 'lon'])
model_0wrT_winter_ACC_err = (corr.max() - corr.min()).values / 2 * 100
corr = xr.corr(anom_7wr_temp_summer, model_7wr_temp_summer, dim='time').median(dim=['lat', 'lon'])
model_7wrT_summer_ACC_err = (corr.max() - corr.min()).values / 2 * 100
corr = xr.corr(anom_4wr_temp_summer, model_4wr_temp_summer, dim='time').median(dim=['lat', 'lon'])
model_4wrT_summer_ACC_err = (corr.max() - corr.min()).values / 2 * 100
corr = xr.corr(anom_NAO_temp_summer, model_NAO_temp_summer, dim='time').median(dim=['lat', 'lon'])
model_NAOT_summer_ACC_err = (corr.max() - corr.min()).values / 2 * 100
corr = xr.corr(anom_0wr_temp_summer, model_0wr_temp_summer, dim='time').median(dim=['lat', 'lon'])
model_0wrT_summer_ACC_err = (corr.max() - corr.min()).values / 2 * 100

corr = xr.corr(anom_7wr_prec_winter, model_7wr_prec_winter, dim='time').median(dim=['lat', 'lon'])
model_7wrP_winter_ACC_err = (corr.max() - corr.min()).values / 2 * 100
corr = xr.corr(anom_4wr_prec_winter, model_4wr_prec_winter, dim='time').median(dim=['lat', 'lon'])
model_4wrP_winter_ACC_err = (corr.max() - corr.min()).values / 2 * 100
corr = xr.corr(anom_NAO_prec_winter, model_NAO_prec_winter, dim='time').median(dim=['lat', 'lon'])
model_NAOP_winter_ACC_err = (corr.max() - corr.min()).values / 2 * 100
corr = xr.corr(anom_0wr_prec_winter, model_0wr_prec_winter, dim='time').median(dim=['lat', 'lon'])
model_0wrP_winter_ACC_err = (corr.max() - corr.min()).values / 2 * 100
corr = xr.corr(anom_7wr_prec_summer, model_7wr_prec_summer, dim='time').median(dim=['lat', 'lon'])
model_7wrP_summer_ACC_err = (corr.max() - corr.min()).values / 2 * 100
corr = xr.corr(anom_4wr_prec_summer, model_4wr_prec_summer, dim='time').median(dim=['lat', 'lon'])
model_4wrP_summer_ACC_err = (corr.max() - corr.min()).values / 2 * 100
corr = xr.corr(anom_NAO_prec_summer, model_NAO_prec_summer, dim='time').median(dim=['lat', 'lon'])
model_NAOP_summer_ACC_err = (corr.max() - corr.min()).values / 2 * 100
corr = xr.corr(anom_0wr_prec_summer, model_0wr_prec_summer, dim='time').median(dim=['lat', 'lon'])
model_0wrP_summer_ACC_err = (corr.max() - corr.min()).values / 2 * 100

# models uncertainty CE
ce = get_ce(anom_7wr_temp_winter, model_7wr_temp_winter).median(dim=['lat', 'lon'])
anom_7wr_temp_winter_CE_err = (ce.max() - ce.min()).values / 2 * 100
ce = get_ce(anom_4wr_temp_winter, model_4wr_temp_winter).median(dim=['lat', 'lon'])
anom_4wr_temp_winter_CE_err = (ce.max() - ce.min()).values / 2 * 100
ce = get_ce(anom_NAO_temp_winter, model_NAO_temp_winter).median(dim=['lat', 'lon'])
anom_NAO_temp_winter_CE_err = (ce.max() - ce.min()).values / 2 * 100
ce = get_ce(anom_0wr_temp_winter, model_0wr_temp_winter).median(dim=['lat', 'lon'])
anom_0wr_temp_winter_CE_err = (ce.max() - ce.min()).values / 2 * 100
ce = get_ce(anom_7wr_temp_summer, model_7wr_temp_summer).median(dim=['lat', 'lon'])
anom_7wr_temp_summer_CE_err = (ce.max() - ce.min()).values / 2 * 100
ce = get_ce(anom_4wr_temp_summer, model_4wr_temp_summer).median(dim=['lat', 'lon'])
anom_4wr_temp_summer_CE_err = (ce.max() - ce.min()).values / 2 * 100
ce = get_ce(anom_NAO_temp_summer, model_NAO_temp_summer).median(dim=['lat', 'lon'])
anom_NAO_temp_summer_CE_err = (ce.max() - ce.min()).values / 2 * 100
ce = get_ce(anom_0wr_temp_summer, model_0wr_temp_summer).median(dim=['lat', 'lon'])
anom_0wr_temp_summer_CE_err = (ce.max() - ce.min()).values / 2 * 100

ce = get_ce(anom_7wr_prec_winter, model_7wr_prec_winter).median(dim=['lat', 'lon'])
anom_7wr_prec_winter_CE_err = (ce.max() - ce.min()).values / 2 * 100
ce = get_ce(anom_4wr_prec_winter, model_4wr_prec_winter).median(dim=['lat', 'lon'])
anom_4wr_prec_winter_CE_err = (ce.max() - ce.min()).values / 2 * 100
ce = get_ce(anom_NAO_prec_winter, model_NAO_prec_winter).median(dim=['lat', 'lon'])
anom_NAO_prec_winter_CE_err = (ce.max() - ce.min()).values / 2 * 100
ce = get_ce(anom_0wr_prec_winter, model_0wr_prec_winter).median(dim=['lat', 'lon'])
anom_0wr_prec_winter_CE_err = (ce.max() - ce.min()).values / 2 * 100
ce = get_ce(anom_7wr_prec_summer, model_7wr_prec_summer).median(dim=['lat', 'lon'])
anom_7wr_prec_summer_CE_err = (ce.max() - ce.min()).values / 2 * 100
ce = get_ce(anom_4wr_prec_summer, model_4wr_prec_summer).median(dim=['lat', 'lon'])
anom_4wr_prec_summer_CE_err = (ce.max() - ce.min()).values / 2 * 100
ce = get_ce(anom_NAO_prec_summer, model_NAO_prec_summer).median(dim=['lat', 'lon'])
anom_NAO_prec_summer_CE_err = (ce.max() - ce.min()).values / 2 * 100
ce = get_ce(anom_0wr_prec_summer, model_0wr_prec_summer).median(dim=['lat', 'lon'])
anom_0wr_prec_summer_CE_err = (ce.max() - ce.min()).values / 2 * 100


# mask outside land
model_7wrT_winter_MAE = model_7wrT_winter_MAE.where(lsm > .8)
model_7wrT_summer_MAE = model_7wrT_summer_MAE.where(lsm > .8)
model_4wrT_winter_MAE = model_4wrT_winter_MAE.where(lsm > .8)
model_4wrT_summer_MAE = model_4wrT_summer_MAE.where(lsm > .8)
model_NAOT_winter_MAE = model_NAOT_winter_MAE.where(lsm > .8)
model_NAOT_summer_MAE = model_NAOT_summer_MAE.where(lsm > .8)
model_0wrT_winter_MAE = model_0wrT_winter_MAE.where(lsm > .8)
model_0wrT_summer_MAE = model_0wrT_summer_MAE.where(lsm > .8)

model_7wrP_winter_MAE = model_7wrP_winter_MAE.where(lsm > .8)
model_7wrP_summer_MAE = model_7wrP_summer_MAE.where(lsm > .8)
model_4wrP_winter_MAE = model_4wrP_winter_MAE.where(lsm > .8)
model_4wrP_summer_MAE = model_4wrP_summer_MAE.where(lsm > .8)
model_NAOP_winter_MAE = model_NAOP_winter_MAE.where(lsm > .8)
model_NAOP_summer_MAE = model_NAOP_summer_MAE.where(lsm > .8)
model_0wrP_winter_MAE = model_0wrP_winter_MAE.where(lsm > .8)
model_0wrP_summer_MAE = model_0wrP_summer_MAE.where(lsm > .8)

model_7wrT_winter_ACC = model_7wrT_winter_ACC.where(lsm > .8)
model_7wrT_summer_ACC = model_7wrT_summer_ACC.where(lsm > .8)
model_4wrT_winter_ACC = model_4wrT_winter_ACC.where(lsm > .8)
model_4wrT_summer_ACC = model_4wrT_summer_ACC.where(lsm > .8)
model_NAOT_winter_ACC = model_NAOT_winter_ACC.where(lsm > .8)
model_NAOT_summer_ACC = model_NAOT_summer_ACC.where(lsm > .8)
model_0wrT_winter_ACC = model_0wrT_winter_ACC.where(lsm > .8)
model_0wrT_summer_ACC = model_0wrT_summer_ACC.where(lsm > .8)

model_7wrP_winter_ACC = model_7wrP_winter_ACC.where(lsm > .8)
model_7wrP_summer_ACC = model_7wrP_summer_ACC.where(lsm > .8)
model_4wrP_winter_ACC = model_4wrP_winter_ACC.where(lsm > .8)
model_4wrP_summer_ACC = model_4wrP_summer_ACC.where(lsm > .8)
model_NAOP_winter_ACC = model_NAOP_winter_ACC.where(lsm > .8)
model_NAOP_summer_ACC = model_NAOP_summer_ACC.where(lsm > .8)
model_0wrP_winter_ACC = model_0wrP_winter_ACC.where(lsm > .8)
model_0wrP_summer_ACC = model_0wrP_summer_ACC.where(lsm > .8)

model_7wrT_winter_CE = model_7wrT_winter_CE.where(lsm > .8)
model_7wrT_summer_CE = model_7wrT_summer_CE.where(lsm > .8)
model_4wrT_winter_CE = model_4wrT_winter_CE.where(lsm > .8)
model_4wrT_summer_CE = model_4wrT_summer_CE.where(lsm > .8)
model_NAOT_winter_CE = model_NAOT_winter_CE.where(lsm > .8)
model_NAOT_summer_CE = model_NAOT_summer_CE.where(lsm > .8)
model_0wrT_winter_CE = model_0wrT_winter_CE.where(lsm > .8)
model_0wrT_summer_CE = model_0wrT_summer_CE.where(lsm > .8)

model_7wrP_winter_CE = model_7wrP_winter_CE.where(lsm > .8)
model_7wrP_summer_CE = model_7wrP_summer_CE.where(lsm > .8)
model_4wrP_winter_CE = model_4wrP_winter_CE.where(lsm > .8)
model_4wrP_summer_CE = model_4wrP_summer_CE.where(lsm > .8)
model_NAOP_winter_CE = model_NAOP_winter_CE.where(lsm > .8)
model_NAOP_summer_CE = model_NAOP_summer_CE.where(lsm > .8)
model_0wrP_winter_CE = model_0wrP_winter_CE.where(lsm > .8)
model_0wrP_summer_CE = model_0wrP_summer_CE.where(lsm > .8)


# -------------------
# Write table to file
# -------------------
f = open('table_skills.tex', 'w')
f.write(r'\begin{table*}' + '\n')
f.write(r'    \centering' + '\n')
f.write(r'    \begin{tabular}{lcccccccccccc}' + '\n')
f.write(r'    \toprule' + '\n')
f.write(r'    &\multicolumn{12}{c}{two-meter temperature}\\' + '\n')
f.write(r'    &\multicolumn{4}{c}{MAE}    &\multicolumn{4}{c}{ACC}    &\multicolumn{4}{c}{CE}\\' + '\n')
f.write(r'    \cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13}' + '\n')
f.write(r'              &7 WR  &4 WR  &NAO  &No idx  &7 WR  &4 WR  &NAO  &No idx  &7 WR  &4 WR  &NAO  &No idx\\' + '\n')

f.write(r'    DJF    &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f) \\' % 
        (
            model_7wrT_winter_MAE.median().values, model_7wrT_winter_MAE_err, model_4wrT_winter_MAE.median().values, model_4wrT_winter_MAE_err, model_NAOT_winter_MAE.median().values, model_NAOT_winter_MAE_err, model_0wrT_winter_MAE.median().values, model_0wrT_winter_MAE_err, \
            model_7wrT_winter_ACC.median().values, model_7wrT_winter_ACC_err, model_4wrT_winter_ACC.median().values, model_4wrT_winter_ACC_err, model_NAOT_winter_ACC.median().values, model_NAOT_winter_ACC_err, model_0wrT_winter_ACC.median().values, model_0wrT_winter_ACC_err, \
            model_7wrT_winter_CE.median().values, anom_7wr_temp_winter_CE_err, model_4wrT_winter_CE.median().values, anom_4wr_temp_winter_CE_err, model_NAOT_winter_CE.median().values, anom_NAO_temp_winter_CE_err, model_0wrT_winter_CE.median().values, anom_0wr_temp_winter_CE_err
         )
)
f.write('\n')
f.write(r'    JJA    &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f) \\' % \
      (
            model_7wrT_summer_MAE.median().values, model_7wrT_summer_MAE_err, model_4wrT_summer_MAE.median().values, model_4wrT_summer_MAE_err, model_NAOT_summer_MAE.median().values, model_NAOT_summer_MAE_err, model_0wrT_summer_MAE.median().values, model_0wrT_summer_MAE_err, \
            model_7wrT_summer_ACC.median().values, model_7wrT_summer_ACC_err, model_4wrT_summer_ACC.median().values, model_4wrT_summer_ACC_err, model_NAOT_summer_ACC.median().values, model_NAOT_summer_ACC_err, model_0wrT_summer_ACC.median().values, model_0wrT_summer_ACC_err, \
            model_7wrT_summer_CE.median().values, anom_7wr_temp_summer_CE_err, model_4wrT_summer_CE.median().values, anom_4wr_temp_summer_CE_err, model_NAOT_summer_CE.median().values, anom_NAO_temp_summer_CE_err, model_0wrT_summer_CE.median().values, anom_0wr_temp_summer_CE_err
      )
)
f.write('\n')
f.write(r'    \cmidrule(lr){2-13} \\' + '\n')
f.write(r'    &\multicolumn{12}{c}{total precipitation}\\' + '\n')
f.write(r'    &\multicolumn{4}{c}{MAE}    &\multicolumn{4}{c}{ACC}    &\multicolumn{4}{c}{CE}\\' + '\n')
f.write(r'    \cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13}' + '\n')

f.write(r'    DJF    &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f) \\' % \
      (
            model_7wrP_winter_MAE.median().values, model_7wrP_winter_MAE_err, model_4wrP_winter_MAE.median().values, model_4wrP_winter_MAE_err, model_NAOP_winter_MAE.median().values, model_NAOP_winter_MAE_err, model_0wrP_winter_MAE.median().values, model_0wrP_winter_MAE_err, \
            model_7wrP_winter_ACC.median().values, model_7wrP_winter_ACC_err, model_4wrP_winter_ACC.median().values, model_4wrP_winter_ACC_err, model_NAOP_winter_ACC.median().values, model_NAOP_winter_ACC_err, model_0wrP_winter_ACC.median().values, model_0wrP_winter_ACC_err, \
            model_7wrP_winter_CE.median().values, anom_7wr_prec_winter_CE_err, model_4wrP_winter_CE.median().values, anom_4wr_prec_winter_CE_err, model_NAOP_winter_CE.median().values, anom_NAO_prec_winter_CE_err, model_0wrP_winter_CE.median().values, anom_0wr_prec_winter_CE_err
      )
)
f.write('\n')
f.write(r'    JJA    &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f)  &%.2f(%.0f) \\' % \
      (
            model_7wrP_summer_MAE.median().values, model_7wrP_summer_MAE_err, model_4wrP_summer_MAE.median().values, model_4wrP_summer_MAE_err, model_NAOP_summer_MAE.median().values, model_NAOP_summer_MAE_err, model_0wrP_summer_MAE.median().values, model_0wrP_summer_MAE_err, \
            model_7wrP_summer_ACC.median().values, model_7wrP_summer_ACC_err, model_4wrP_summer_ACC.median().values, model_4wrP_summer_ACC_err, model_NAOP_summer_ACC.median().values, model_NAOP_summer_ACC_err, model_0wrP_summer_ACC.median().values, model_0wrP_summer_ACC_err, \
            model_7wrP_summer_CE.median().values, anom_7wr_prec_summer_CE_err, model_4wrP_summer_CE.median().values, anom_4wr_prec_summer_CE_err, model_NAOP_summer_CE.median().values, anom_NAO_prec_summer_CE_err, model_0wrP_summer_CE.median().values, anom_0wr_prec_summer_CE_err
      )
)
f.write('\n')
f.write(r'    \bottomrule' + '\n')
f.write(r'    \end{tabular}' + '\n')
f.write(r'    \caption{Median spatial values of MAE, ACC, and CE for reconstructed monthly mean two-meter temperature and total precipitation during winter (DJF) and summer (JJA), using seven \ac{WR}, four 4 \ac{WR}, NAO, or no indices as input to the AI model. Uncertainties (in parentheses) denote the semi-range (half the difference between maximum and minimum) of the skill metrics’ medians across multiple model initializations with different random seeds.}' + '\n')
f.write(r'    \label{tab:numWR}' + '\n')
f.write(r'\end{table*}')
f.close()
