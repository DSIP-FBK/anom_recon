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

def plot_boxplot(ax, data, position, color, marker):
    bplot = ax.boxplot(
        data[~np.isnan(data)], 
        positions=position,
        patch_artist=True,
        showmeans=False, boxprops={'color': color},
        medianprops={'linewidth': 1, 'color': color}, 
        #medianprops={'marker': marker, 'markeredgecolor': color, 'markerfacecolor': color, 'markersize': 4}, 
        showfliers=False,
        whiskerprops={'color': color}, capprops = {'color': color},
        widths=[.5,]
    )
    [patch.set(facecolor=color, alpha=.2) for patch in bplot['boxes']]
    return bplot

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("-wr7_temp", type=str, help="path to the folder containing the model trained with 7WR to reconstruct temperature" )
parser.add_argument("-wr4_temp", type=str, help="path to the folder containing the model trained with 4WR to reconstruct temperature" )
parser.add_argument("-wr2_temp", type=str, help="path to the folder containing the model trained with 2WR to reconstruct temperature" )
parser.add_argument("-wr1_temp", type=str, help="path to the folder containing the model trained with 1WR to reconstruct temperature" )
parser.add_argument("-wr0_temp", type=str, help="path to the folder containing the model trained with 0WR to reconstruct temperature" )
parser.add_argument("-wr7_prec", type=str, help="path to the folder containing the model trained with 7WR to reconstruct precipitation" )
parser.add_argument("-wr4_prec", type=str, help="path to the folder containing the model trained with 4WR to reconstruct precipitation" )
parser.add_argument("-wr2_prec", type=str, help="path to the folder containing the model trained with 2WR to reconstruct precipitation" )
parser.add_argument("-wr1_prec", type=str, help="path to the folder containing the model trained with 1WR to reconstruct precipitation" )
parser.add_argument("-wr0_prec", type=str, help="path to the folder containing the model trained with 0WR to reconstruct precipitation" )
args = parser.parse_args()

# seasons
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
model_7wr_temp_winter  = model_7wr_temp[np.isin(model_7wr_temp.time.dt.month, winter_months)]
anom_7wr_temp_winter   = anom_7wr_temp[np.isin(anom_7wr_temp.time.dt.month, winter_months)]
model_7wr_temp_summer  = model_7wr_temp[np.isin(model_7wr_temp.time.dt.month, summer_months)]
anom_7wr_temp_summer   = anom_7wr_temp[np.isin(anom_7wr_temp.time.dt.month, summer_months)]

# 4WR on temprature
torch_model, datamodule, config = get_torch_models_infos(args.wr4_temp)
idxs_4wr_temp          = xr.open_dataarray(config['data']['indexes_paths'][0]).sel(mode=slice(1, config['data']['num_indexes'][0]))
anom_4wr_temp          = xr.open_dataarray(config['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
model_4wr_temp         = get_models_out(torch_model, idxs_4wr_temp, anom_4wr_temp, datamodule)
model_4wr_temp_winter  = model_4wr_temp[np.isin(model_4wr_temp.time.dt.month, winter_months)]
anom_4wr_temp_winter   = anom_4wr_temp[np.isin(anom_4wr_temp.time.dt.month, winter_months)]
model_4wr_temp_summer  = model_4wr_temp[np.isin(model_4wr_temp.time.dt.month, summer_months)]
anom_4wr_temp_summer   = anom_4wr_temp[np.isin(anom_4wr_temp.time.dt.month, summer_months)]

# 2WR on temprature
torch_model, datamodule, config = get_torch_models_infos(args.wr2_temp)
idxs_2wr_temp          = xr.open_dataarray(config['data']['indexes_paths'][0]).sel(mode=slice(1, config['data']['num_indexes'][0]))
anom_2wr_temp          = xr.open_dataarray(config['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
model_2wr_temp         = get_models_out(torch_model, idxs_2wr_temp, anom_2wr_temp, datamodule)
model_2wr_temp_winter  = model_2wr_temp[np.isin(model_2wr_temp.time.dt.month, winter_months)]
anom_2wr_temp_winter   = anom_2wr_temp[np.isin(anom_2wr_temp.time.dt.month, winter_months)]
model_2wr_temp_summer  = model_2wr_temp[np.isin(model_2wr_temp.time.dt.month, summer_months)]
anom_2wr_temp_summer   = anom_2wr_temp[np.isin(anom_2wr_temp.time.dt.month, summer_months)]

# 1WR on temprature
torch_model, datamodule, config = get_torch_models_infos(args.wr1_temp)
idxs_1wr_temp          = xr.open_dataarray(config['data']['indexes_paths'][0]).sel(mode=slice(1, config['data']['num_indexes'][0]))
anom_1wr_temp          = xr.open_dataarray(config['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
model_1wr_temp         = get_models_out(torch_model, idxs_1wr_temp, anom_1wr_temp, datamodule)
model_1wr_temp_winter  = model_1wr_temp[np.isin(model_1wr_temp.time.dt.month, winter_months)]
anom_1wr_temp_winter   = anom_1wr_temp[np.isin(anom_1wr_temp.time.dt.month, winter_months)]
model_1wr_temp_summer  = model_1wr_temp[np.isin(model_1wr_temp.time.dt.month, summer_months)]
anom_1wr_temp_summer   = anom_1wr_temp[np.isin(anom_1wr_temp.time.dt.month, summer_months)]

# 0WR on temprature
torch_model, datamodule, config = get_torch_models_infos(args.wr0_temp)
idxs_0wr_temp          = xr.open_dataarray(config['data']['indexes_paths'][0]).sel(mode=slice(1, config['data']['num_indexes'][0]))
anom_0wr_temp          = xr.open_dataarray(config['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
model_0wr_temp         = get_models_out(torch_model, idxs_0wr_temp, anom_0wr_temp, datamodule)
model_0wr_temp_winter  = model_0wr_temp[np.isin(model_0wr_temp.time.dt.month, winter_months)]
anom_0wr_temp_winter   = anom_0wr_temp[np.isin(anom_0wr_temp.time.dt.month, winter_months)]
model_0wr_temp_summer  = model_0wr_temp[np.isin(model_0wr_temp.time.dt.month, summer_months)]
anom_0wr_temp_summer   = anom_0wr_temp[np.isin(anom_0wr_temp.time.dt.month, summer_months)]

# 7WR on precipitation
torch_model, datamodule, config = get_torch_models_infos(args.wr7_prec)
idxs_7wr_prec          = xr.open_dataarray(config['data']['indexes_paths'][0]).sel(mode=slice(1, config['data']['num_indexes'][0]))
anom_7wr_prec          = xr.open_dataarray(config['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
model_7wr_prec         = get_models_out(torch_model, idxs_7wr_prec, anom_7wr_prec, datamodule)
model_7wr_prec_winter  = model_7wr_prec[np.isin(model_7wr_prec.time.dt.month, winter_months)]
anom_7wr_prec_winter   = anom_7wr_prec[np.isin(anom_7wr_prec.time.dt.month, winter_months)]
model_7wr_prec_summer  = model_7wr_prec[np.isin(model_7wr_prec.time.dt.month, summer_months)]
anom_7wr_prec_summer   = anom_7wr_prec[np.isin(anom_7wr_prec.time.dt.month, summer_months)]

# 4WR on precipitation
torch_model, datamodule, config = get_torch_models_infos(args.wr4_prec)
idxs_4wr_prec          = xr.open_dataarray(config['data']['indexes_paths'][0]).sel(mode=slice(1, config['data']['num_indexes'][0]))
anom_4wr_prec          = xr.open_dataarray(config['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
model_4wr_prec         = get_models_out(torch_model, idxs_4wr_prec, anom_4wr_prec, datamodule)
model_4wr_prec_winter  = model_4wr_prec[np.isin(model_4wr_prec.time.dt.month, winter_months)]
anom_4wr_prec_winter   = anom_4wr_prec[np.isin(anom_4wr_prec.time.dt.month, winter_months)]
model_4wr_prec_summer  = model_4wr_prec[np.isin(model_4wr_prec.time.dt.month, summer_months)]
anom_4wr_prec_summer   = anom_4wr_prec[np.isin(anom_4wr_prec.time.dt.month, summer_months)]

# 2WR on precipitation
torch_model, datamodule, config = get_torch_models_infos(args.wr2_prec)
idxs_2wr_prec          = xr.open_dataarray(config['data']['indexes_paths'][0]).sel(mode=slice(1, config['data']['num_indexes'][0]))
anom_2wr_prec          = xr.open_dataarray(config['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
model_2wr_prec         = get_models_out(torch_model, idxs_2wr_prec, anom_2wr_prec, datamodule)
model_2wr_prec_winter  = model_2wr_prec[np.isin(model_2wr_prec.time.dt.month, winter_months)]
anom_2wr_prec_winter   = anom_2wr_prec[np.isin(anom_2wr_prec.time.dt.month, winter_months)]
model_2wr_prec_summer  = model_2wr_prec[np.isin(model_2wr_prec.time.dt.month, summer_months)]
anom_2wr_prec_summer   = anom_2wr_prec[np.isin(anom_2wr_prec.time.dt.month, summer_months)]

# 1WR on precipitation
torch_model, datamodule, config = get_torch_models_infos(args.wr1_prec)
idxs_1wr_prec          = xr.open_dataarray(config['data']['indexes_paths'][0]).sel(mode=slice(1, config['data']['num_indexes'][0]))
anom_1wr_prec          = xr.open_dataarray(config['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
model_1wr_prec         = get_models_out(torch_model, idxs_1wr_prec, anom_1wr_prec, datamodule)
model_1wr_prec_winter  = model_1wr_prec[np.isin(model_1wr_prec.time.dt.month, winter_months)]
anom_1wr_prec_winter   = anom_1wr_prec[np.isin(anom_1wr_prec.time.dt.month, winter_months)]
model_1wr_prec_summer  = model_1wr_prec[np.isin(model_1wr_prec.time.dt.month, summer_months)]
anom_1wr_prec_summer   = anom_1wr_prec[np.isin(anom_1wr_prec.time.dt.month, summer_months)]

# 0WR on precipitation
torch_model, datamodule, config = get_torch_models_infos(args.wr0_prec)
idxs_0wr_prec          = xr.open_dataarray(config['data']['indexes_paths'][0]).sel(mode=slice(1, config['data']['num_indexes'][0]))
anom_0wr_prec          = xr.open_dataarray(config['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
model_0wr_prec         = get_models_out(torch_model, idxs_0wr_prec, anom_0wr_prec, datamodule)
model_0wr_prec_winter  = model_0wr_prec[np.isin(model_0wr_prec.time.dt.month, winter_months)]
anom_0wr_prec_winter   = anom_0wr_prec[np.isin(anom_0wr_prec.time.dt.month, winter_months)]
model_0wr_prec_summer  = model_0wr_prec[np.isin(model_0wr_prec.time.dt.month, summer_months)]
anom_0wr_prec_summer   = anom_0wr_prec[np.isin(anom_0wr_prec.time.dt.month, summer_months)]

# reduce all variables to common time-range
start, end            = '2011', '2024'
anom_7wr_temp_winter  = anom_7wr_temp_winter.sel(time=slice(start, end))
anom_4wr_temp_winter  = anom_4wr_temp_winter.sel(time=slice(start, end))
anom_2wr_temp_winter  = anom_2wr_temp_winter.sel(time=slice(start, end))
anom_1wr_temp_winter  = anom_1wr_temp_winter.sel(time=slice(start, end))
anom_0wr_temp_winter  = anom_0wr_temp_winter.sel(time=slice(start, end))
anom_7wr_prec_winter  = anom_7wr_prec_winter.sel(time=slice(start, end))
anom_4wr_prec_winter  = anom_4wr_prec_winter.sel(time=slice(start, end))
anom_2wr_prec_winter  = anom_2wr_prec_winter.sel(time=slice(start, end))
anom_1wr_prec_winter  = anom_1wr_prec_winter.sel(time=slice(start, end))
anom_0wr_prec_winter  = anom_0wr_prec_winter.sel(time=slice(start, end))
model_7wr_temp_winter = model_7wr_temp_winter.sel(time=slice(start, end))
model_4wr_temp_winter = model_4wr_temp_winter.sel(time=slice(start, end))
model_2wr_temp_winter = model_2wr_temp_winter.sel(time=slice(start, end))
model_1wr_temp_winter = model_1wr_temp_winter.sel(time=slice(start, end))
model_0wr_temp_winter = model_0wr_temp_winter.sel(time=slice(start, end))
model_7wr_prec_winter = model_7wr_prec_winter.sel(time=slice(start, end))
model_4wr_prec_winter = model_4wr_prec_winter.sel(time=slice(start, end))
model_2wr_prec_winter = model_2wr_prec_winter.sel(time=slice(start, end))
model_1wr_prec_winter = model_1wr_prec_winter.sel(time=slice(start, end))
model_0wr_prec_winter = model_0wr_prec_winter.sel(time=slice(start, end))

anom_7wr_temp_summer  = anom_7wr_temp_summer.sel(time=slice(start, end))
anom_4wr_temp_summer  = anom_4wr_temp_summer.sel(time=slice(start, end))
anom_2wr_temp_summer  = anom_2wr_temp_summer.sel(time=slice(start, end))
anom_1wr_temp_summer  = anom_1wr_temp_summer.sel(time=slice(start, end))
anom_0wr_temp_summer  = anom_0wr_temp_summer.sel(time=slice(start, end))
anom_7wr_prec_summer  = anom_7wr_prec_summer.sel(time=slice(start, end))
anom_4wr_prec_summer  = anom_4wr_prec_summer.sel(time=slice(start, end))
anom_2wr_prec_summer  = anom_2wr_prec_summer.sel(time=slice(start, end))
anom_1wr_prec_summer  = anom_1wr_prec_summer.sel(time=slice(start, end))
anom_0wr_prec_summer  = anom_0wr_prec_summer.sel(time=slice(start, end))
model_7wr_temp_summer = model_7wr_temp_summer.sel(time=slice(start, end))
model_4wr_temp_summer = model_4wr_temp_summer.sel(time=slice(start, end))
model_2wr_temp_summer = model_2wr_temp_summer.sel(time=slice(start, end))
model_1wr_temp_summer = model_1wr_temp_summer.sel(time=slice(start, end))
model_0wr_temp_summer = model_0wr_temp_summer.sel(time=slice(start, end))
model_7wr_prec_summer = model_7wr_prec_summer.sel(time=slice(start, end))
model_4wr_prec_summer = model_4wr_prec_summer.sel(time=slice(start, end))
model_2wr_prec_summer = model_2wr_prec_summer.sel(time=slice(start, end))
model_1wr_prec_summer = model_1wr_prec_summer.sel(time=slice(start, end))
model_0wr_prec_summer = model_0wr_prec_summer.sel(time=slice(start, end))


# --------------
# Compute skills
# --------------
print('Computing skills...')

# mean absolute error
# model_7wrT_winter_MAE = abs(anom_7wr_temp_winter - model_7wr_temp_winter.mean(dim='number')).mean(dim='time')
# model_7wrT_summer_MAE = abs(anom_7wr_temp_summer - model_7wr_temp_summer.mean(dim='number')).mean(dim='time')
# model_4wrT_winter_MAE = abs(anom_4wr_temp_winter - model_4wr_temp_winter.mean(dim='number')).mean(dim='time')
# model_4wrT_summer_MAE = abs(anom_4wr_temp_summer - model_4wr_temp_summer.mean(dim='number')).mean(dim='time')
# model_2wrT_winter_MAE = abs(anom_2wr_temp_winter - model_2wr_temp_winter.mean(dim='number')).mean(dim='time')
# model_2wrT_summer_MAE = abs(anom_2wr_temp_summer - model_2wr_temp_summer.mean(dim='number')).mean(dim='time')
# model_0wrT_winter_MAE = abs(anom_0wr_temp_winter - model_0wr_temp_winter.mean(dim='number')).mean(dim='time')
# model_0wrT_summer_MAE = abs(anom_0wr_temp_summer - model_0wr_temp_summer.mean(dim='number')).mean(dim='time')

# model_7wrP_winter_MAE = abs(anom_7wr_prec_winter - model_7wr_prec_winter.mean(dim='number')).mean(dim='time')
# model_7wrP_summer_MAE = abs(anom_7wr_prec_summer - model_7wr_prec_summer.mean(dim='number')).mean(dim='time')
# model_4wrP_winter_MAE = abs(anom_4wr_prec_winter - model_4wr_prec_winter.mean(dim='number')).mean(dim='time')
# model_4wrP_summer_MAE = abs(anom_4wr_prec_summer - model_4wr_prec_summer.mean(dim='number')).mean(dim='time')
# model_2wrP_winter_MAE = abs(anom_2wr_prec_winter - model_2wr_prec_winter.mean(dim='number')).mean(dim='time')
# model_2wrP_summer_MAE = abs(anom_2wr_prec_summer - model_2wr_prec_summer.mean(dim='number')).mean(dim='time')
# model_0wrP_winter_MAE = abs(anom_0wr_prec_winter - model_0wr_prec_winter.mean(dim='number')).mean(dim='time')
# model_0wrP_summer_MAE = abs(anom_0wr_prec_summer - model_0wr_prec_summer.mean(dim='number')).mean(dim='time')

# anomaly correlation coefficient
model_7wrT_winter_ACC = xr.corr(anom_7wr_temp_winter, model_7wr_temp_winter.mean(dim='number'), dim='time')
model_7wrT_summer_ACC = xr.corr(anom_7wr_temp_summer, model_7wr_temp_summer.mean(dim='number'), dim='time')
model_4wrT_winter_ACC = xr.corr(anom_4wr_temp_winter, model_4wr_temp_winter.mean(dim='number'), dim='time')
model_4wrT_summer_ACC = xr.corr(anom_4wr_temp_summer, model_4wr_temp_summer.mean(dim='number'), dim='time')
model_2wrT_winter_ACC = xr.corr(anom_2wr_temp_winter, model_2wr_temp_winter.mean(dim='number'), dim='time')
model_2wrT_summer_ACC = xr.corr(anom_2wr_temp_summer, model_2wr_temp_summer.mean(dim='number'), dim='time')
model_1wrT_winter_ACC = xr.corr(anom_1wr_temp_winter, model_1wr_temp_winter.mean(dim='number'), dim='time')
model_1wrT_summer_ACC = xr.corr(anom_1wr_temp_summer, model_1wr_temp_summer.mean(dim='number'), dim='time')
model_0wrT_winter_ACC = xr.corr(anom_0wr_temp_winter, model_0wr_temp_winter.mean(dim='number'), dim='time')
model_0wrT_summer_ACC = xr.corr(anom_0wr_temp_summer, model_0wr_temp_summer.mean(dim='number'), dim='time')

model_7wrP_winter_ACC = xr.corr(anom_7wr_prec_winter, model_7wr_prec_winter.mean(dim='number'), dim='time')
model_7wrP_summer_ACC = xr.corr(anom_7wr_prec_summer, model_7wr_prec_summer.mean(dim='number'), dim='time')
model_4wrP_winter_ACC = xr.corr(anom_4wr_prec_winter, model_4wr_prec_winter.mean(dim='number'), dim='time')
model_4wrP_summer_ACC = xr.corr(anom_4wr_prec_summer, model_4wr_prec_summer.mean(dim='number'), dim='time')
model_2wrP_winter_ACC = xr.corr(anom_2wr_prec_winter, model_2wr_prec_winter.mean(dim='number'), dim='time')
model_2wrP_summer_ACC = xr.corr(anom_2wr_prec_summer, model_2wr_prec_summer.mean(dim='number'), dim='time')
model_1wrP_winter_ACC = xr.corr(anom_1wr_prec_winter, model_1wr_prec_winter.mean(dim='number'), dim='time')
model_1wrP_summer_ACC = xr.corr(anom_1wr_prec_summer, model_1wr_prec_summer.mean(dim='number'), dim='time')
model_0wrP_winter_ACC = xr.corr(anom_0wr_prec_winter, model_0wr_prec_winter.mean(dim='number'), dim='time')
model_0wrP_summer_ACC = xr.corr(anom_0wr_prec_summer, model_0wr_prec_summer.mean(dim='number'), dim='time')

# coefficient of efficacy
model_7wrT_winter_CE = get_ce(anom_7wr_temp_winter, model_7wr_temp_winter.mean(dim='number'))
model_7wrT_summer_CE = get_ce(anom_7wr_temp_summer, model_7wr_temp_summer.mean(dim='number'))
model_4wrT_winter_CE = get_ce(anom_4wr_temp_winter, model_4wr_temp_winter.mean(dim='number'))
model_4wrT_summer_CE = get_ce(anom_4wr_temp_summer, model_4wr_temp_summer.mean(dim='number'))
model_2wrT_winter_CE = get_ce(anom_2wr_temp_winter, model_2wr_temp_winter.mean(dim='number'))
model_2wrT_summer_CE = get_ce(anom_2wr_temp_summer, model_2wr_temp_summer.mean(dim='number'))
model_1wrT_winter_CE = get_ce(anom_1wr_temp_winter, model_1wr_temp_winter.mean(dim='number'))
model_1wrT_summer_CE = get_ce(anom_1wr_temp_summer, model_1wr_temp_summer.mean(dim='number'))
model_0wrT_winter_CE = get_ce(anom_0wr_temp_winter, model_0wr_temp_winter.mean(dim='number'))
model_0wrT_summer_CE = get_ce(anom_0wr_temp_summer, model_0wr_temp_summer.mean(dim='number'))

model_7wrP_winter_CE = get_ce(anom_7wr_prec_winter, model_7wr_prec_winter.mean(dim='number'))
model_7wrP_summer_CE = get_ce(anom_7wr_prec_summer, model_7wr_prec_summer.mean(dim='number'))
model_4wrP_winter_CE = get_ce(anom_4wr_prec_winter, model_4wr_prec_winter.mean(dim='number'))
model_4wrP_summer_CE = get_ce(anom_4wr_prec_summer, model_4wr_prec_summer.mean(dim='number'))
model_2wrP_winter_CE = get_ce(anom_2wr_prec_winter, model_2wr_prec_winter.mean(dim='number'))
model_2wrP_summer_CE = get_ce(anom_2wr_prec_summer, model_2wr_prec_summer.mean(dim='number'))
model_1wrP_winter_CE = get_ce(anom_1wr_prec_winter, model_1wr_prec_winter.mean(dim='number'))
model_1wrP_summer_CE = get_ce(anom_1wr_prec_summer, model_1wr_prec_summer.mean(dim='number'))
model_0wrP_winter_CE = get_ce(anom_0wr_prec_winter, model_0wr_prec_winter.mean(dim='number'))
model_0wrP_summer_CE = get_ce(anom_0wr_prec_summer, model_0wr_prec_summer.mean(dim='number'))

# models uncertainty ACC
corr = xr.corr(anom_7wr_temp_winter, model_7wr_temp_winter, dim='time').median(dim=['lat', 'lon'])
anom_7wr_temp_winter_ACC_err = (corr.max() - corr.min()).values / 2
corr = xr.corr(anom_4wr_temp_winter, model_4wr_temp_winter, dim='time').median(dim=['lat', 'lon'])
anom_4wr_temp_winter_ACC_err = (corr.max() - corr.min()).values / 2
corr = xr.corr(anom_2wr_temp_winter, model_2wr_temp_winter, dim='time').median(dim=['lat', 'lon'])
anom_2wr_temp_winter_ACC_err = (corr.max() - corr.min()).values / 2
corr = xr.corr(anom_1wr_temp_winter, model_1wr_temp_winter, dim='time').median(dim=['lat', 'lon'])
anom_1wr_temp_winter_ACC_err = (corr.max() - corr.min()).values / 2
corr = xr.corr(anom_0wr_temp_winter, model_0wr_temp_winter, dim='time').median(dim=['lat', 'lon'])
anom_0wr_temp_winter_ACC_err = (corr.max() - corr.min()).values / 2
corr = xr.corr(anom_7wr_temp_summer, model_7wr_temp_summer, dim='time').median(dim=['lat', 'lon'])
anom_7wr_temp_summer_ACC_err = (corr.max() - corr.min()).values / 2
corr = xr.corr(anom_4wr_temp_summer, model_4wr_temp_summer, dim='time').median(dim=['lat', 'lon'])
anom_4wr_temp_summer_ACC_err = (corr.max() - corr.min()).values / 2
corr = xr.corr(anom_2wr_temp_summer, model_2wr_temp_summer, dim='time').median(dim=['lat', 'lon'])
anom_2wr_temp_summer_ACC_err = (corr.max() - corr.min()).values / 2
corr = xr.corr(anom_1wr_temp_summer, model_1wr_temp_summer, dim='time').median(dim=['lat', 'lon'])
anom_1wr_temp_summer_ACC_err = (corr.max() - corr.min()).values / 2
corr = xr.corr(anom_0wr_temp_summer, model_0wr_temp_summer, dim='time').median(dim=['lat', 'lon'])
anom_0wr_temp_summer_ACC_err = (corr.max() - corr.min()).values / 2
corr = xr.corr(anom_7wr_prec_winter, model_7wr_prec_winter, dim='time').median(dim=['lat', 'lon'])

anom_7wr_prec_winter_ACC_err = (corr.max() - corr.min()).values / 2
corr = xr.corr(anom_4wr_prec_winter, model_4wr_prec_winter, dim='time').median(dim=['lat', 'lon'])
anom_4wr_prec_winter_ACC_err = (corr.max() - corr.min()).values / 2
corr = xr.corr(anom_2wr_prec_winter, model_2wr_prec_winter, dim='time').median(dim=['lat', 'lon'])
anom_2wr_prec_winter_ACC_err = (corr.max() - corr.min()).values / 2
corr = xr.corr(anom_1wr_prec_winter, model_1wr_prec_winter, dim='time').median(dim=['lat', 'lon'])
anom_1wr_prec_winter_ACC_err = (corr.max() - corr.min()).values / 2
corr = xr.corr(anom_0wr_prec_winter, model_0wr_prec_winter, dim='time').median(dim=['lat', 'lon'])
anom_0wr_prec_winter_ACC_err = (corr.max() - corr.min()).values / 2
corr = xr.corr(anom_7wr_prec_summer, model_7wr_prec_summer, dim='time').median(dim=['lat', 'lon'])
anom_7wr_prec_summer_ACC_err = (corr.max() - corr.min()).values / 2
corr = xr.corr(anom_4wr_prec_summer, model_4wr_prec_summer, dim='time').median(dim=['lat', 'lon'])
anom_4wr_prec_summer_ACC_err = (corr.max() - corr.min()).values / 2
corr = xr.corr(anom_2wr_prec_summer, model_2wr_prec_summer, dim='time').median(dim=['lat', 'lon'])
anom_2wr_prec_summer_ACC_err = (corr.max() - corr.min()).values / 2
corr = xr.corr(anom_1wr_prec_summer, model_1wr_prec_summer, dim='time').median(dim=['lat', 'lon'])
anom_1wr_prec_summer_ACC_err = (corr.max() - corr.min()).values / 2
corr = xr.corr(anom_0wr_prec_summer, model_0wr_prec_summer, dim='time').median(dim=['lat', 'lon'])
anom_0wr_prec_summer_ACC_err = (corr.max() - corr.min()).values / 2

# models uncertainty CE
ce = get_ce(anom_7wr_temp_winter, model_7wr_temp_winter).median(dim=['lat', 'lon'])
anom_7wr_temp_winter_CE_err = (ce.max() - ce.min()).values / 2
ce = get_ce(anom_4wr_temp_winter, model_4wr_temp_winter).median(dim=['lat', 'lon'])
anom_4wr_temp_winter_CE_err = (ce.max() - ce.min()).values / 2
ce = get_ce(anom_2wr_temp_winter, model_2wr_temp_winter).median(dim=['lat', 'lon'])
anom_2wr_temp_winter_CE_err = (ce.max() - ce.min()).values / 2
ce = get_ce(anom_1wr_temp_winter, model_1wr_temp_winter).median(dim=['lat', 'lon'])
anom_1wr_temp_winter_CE_err = (ce.max() - ce.min()).values / 2
ce = get_ce(anom_0wr_temp_winter, model_0wr_temp_winter).median(dim=['lat', 'lon'])
anom_0wr_temp_winter_CE_err = (ce.max() - ce.min()).values / 2
ce = get_ce(anom_7wr_temp_summer, model_7wr_temp_summer).median(dim=['lat', 'lon'])
anom_7wr_temp_summer_CE_err = (ce.max() - ce.min()).values / 2
ce = get_ce(anom_4wr_temp_summer, model_4wr_temp_summer).median(dim=['lat', 'lon'])
anom_4wr_temp_summer_CE_err = (ce.max() - ce.min()).values / 2
ce = get_ce(anom_2wr_temp_summer, model_2wr_temp_summer).median(dim=['lat', 'lon'])
anom_2wr_temp_summer_CE_err = (ce.max() - ce.min()).values / 2
ce = get_ce(anom_1wr_temp_summer, model_1wr_temp_summer).median(dim=['lat', 'lon'])
anom_1wr_temp_summer_CE_err = (ce.max() - ce.min()).values / 2
ce = get_ce(anom_0wr_temp_summer, model_0wr_temp_summer).median(dim=['lat', 'lon'])
anom_0wr_temp_summer_CE_err = (ce.max() - ce.min()).values / 2

ce = get_ce(anom_7wr_prec_winter, model_7wr_prec_winter).median(dim=['lat', 'lon'])
anom_7wr_prec_winter_CE_err = (ce.max() - ce.min()).values / 2
ce = get_ce(anom_4wr_prec_winter, model_4wr_prec_winter).median(dim=['lat', 'lon'])
anom_4wr_prec_winter_CE_err = (ce.max() - ce.min()).values / 2
ce = get_ce(anom_2wr_prec_winter, model_2wr_prec_winter).median(dim=['lat', 'lon'])
anom_2wr_prec_winter_CE_err = (ce.max() - ce.min()).values / 2
ce = get_ce(anom_1wr_prec_winter, model_1wr_prec_winter).median(dim=['lat', 'lon'])
anom_1wr_prec_winter_CE_err = (ce.max() - ce.min()).values / 2
ce = get_ce(anom_0wr_prec_winter, model_0wr_prec_winter).median(dim=['lat', 'lon'])
anom_0wr_prec_winter_CE_err = (ce.max() - ce.min()).values / 2
ce = get_ce(anom_7wr_prec_summer, model_7wr_prec_summer).median(dim=['lat', 'lon'])
anom_7wr_prec_summer_CE_err = (ce.max() - ce.min()).values / 2
ce = get_ce(anom_4wr_prec_summer, model_4wr_prec_summer).median(dim=['lat', 'lon'])
anom_4wr_prec_summer_CE_err = (ce.max() - ce.min()).values / 2
ce = get_ce(anom_2wr_prec_summer, model_2wr_prec_summer).median(dim=['lat', 'lon'])
anom_2wr_prec_summer_CE_err = (ce.max() - ce.min()).values / 2
ce = get_ce(anom_1wr_prec_summer, model_1wr_prec_summer).median(dim=['lat', 'lon'])
anom_1wr_prec_summer_CE_err = (ce.max() - ce.min()).values / 2
ce = get_ce(anom_0wr_prec_summer, model_0wr_prec_summer).median(dim=['lat', 'lon'])
anom_0wr_prec_summer_CE_err = (ce.max() - ce.min()).values / 2


# mask ACC with low p-values
# ptresh = .05
# model_7wrT_winter_ACC = np.where(model_7wrT_winter_ACC.pvalue < ptresh, model_7wrT_winter_ACC.statistic, np.nan)
# model_7wrT_summer_ACC = np.where(model_7wrT_summer_ACC.pvalue < ptresh, model_7wrT_summer_ACC.statistic, np.nan)
# model_4wrT_winter_ACC = np.where(model_4wrT_winter_ACC.pvalue < ptresh, model_4wrT_winter_ACC.statistic, np.nan)
# model_4wrT_summer_ACC = np.where(model_4wrT_summer_ACC.pvalue < ptresh, model_4wrT_summer_ACC.statistic, np.nan)
# model_2wrT_winter_ACC = np.where(model_2wrT_winter_ACC.pvalue < ptresh, model_2wrT_winter_ACC.statistic, np.nan)
# model_2wrT_summer_ACC = np.where(model_2wrT_summer_ACC.pvalue < ptresh, model_2wrT_summer_ACC.statistic, np.nan)
# model_0wrT_winter_ACC = np.where(model_0wrT_winter_ACC.pvalue < ptresh, model_0wrT_winter_ACC.statistic, np.nan)
# model_0wrT_summer_ACC = np.where(model_0wrT_summer_ACC.pvalue < ptresh, model_0wrT_summer_ACC.statistic, np.nan)

# model_7wrP_winter_ACC = np.where(model_7wrP_winter_ACC.pvalue < ptresh, model_7wrP_winter_ACC.statistic, np.nan)
# model_7wrP_summer_ACC = np.where(model_7wrP_summer_ACC.pvalue < ptresh, model_7wrP_summer_ACC.statistic, np.nan)
# model_4wrP_winter_ACC = np.where(model_4wrP_winter_ACC.pvalue < ptresh, model_4wrP_winter_ACC.statistic, np.nan)
# model_4wrP_summer_ACC = np.where(model_4wrP_summer_ACC.pvalue < ptresh, model_4wrP_summer_ACC.statistic, np.nan)
# model_2wrP_winter_ACC = np.where(model_2wrP_winter_ACC.pvalue < ptresh, model_2wrP_winter_ACC.statistic, np.nan)
# model_2wrP_summer_ACC = np.where(model_2wrP_summer_ACC.pvalue < ptresh, model_2wrP_summer_ACC.statistic, np.nan)
# model_0wrP_winter_ACC = np.where(model_0wrP_winter_ACC.pvalue < ptresh, model_0wrP_winter_ACC.statistic, np.nan)
# model_0wrP_summer_ACC = np.where(model_0wrP_summer_ACC.pvalue < ptresh, model_0wrP_summer_ACC.statistic, np.nan)

# mask outside land
# model_7wrT_winter_MAE = model_7wrT_winter_MAE.where(lsm > .8)
# model_7wrT_summer_MAE = model_7wrT_summer_MAE.where(lsm > .8)
# model_4wrT_winter_MAE = model_4wrT_winter_MAE.where(lsm > .8)
# model_4wrT_summer_MAE = model_4wrT_summer_MAE.where(lsm > .8)
# model_2wrT_winter_MAE = model_2wrT_winter_MAE.where(lsm > .8)
# model_2wrT_summer_MAE = model_2wrT_summer_MAE.where(lsm > .8)
# model_0wrT_winter_MAE = model_0wrT_winter_MAE.where(lsm > .8)
# model_0wrT_summer_MAE = model_0wrT_summer_MAE.where(lsm > .8)

# model_7wrP_winter_MAE = model_7wrP_winter_MAE.where(lsm > .8)
# model_7wrP_summer_MAE = model_7wrP_summer_MAE.where(lsm > .8)
# model_4wrP_winter_MAE = model_4wrP_winter_MAE.where(lsm > .8)
# model_4wrP_summer_MAE = model_4wrP_summer_MAE.where(lsm > .8)
# model_2wrP_winter_MAE = model_2wrP_winter_MAE.where(lsm > .8)
# model_2wrP_summer_MAE = model_2wrP_summer_MAE.where(lsm > .8)
# model_0wrP_winter_MAE = model_0wrP_winter_MAE.where(lsm > .8)
# model_0wrP_summer_MAE = model_0wrP_summer_MAE.where(lsm > .8)

model_7wrT_winter_ACC = model_7wrT_winter_ACC.where(lsm > .8)
model_7wrT_summer_ACC = model_7wrT_summer_ACC.where(lsm > .8)
model_4wrT_winter_ACC = model_4wrT_winter_ACC.where(lsm > .8)
model_4wrT_summer_ACC = model_4wrT_summer_ACC.where(lsm > .8)
model_2wrT_winter_ACC = model_2wrT_winter_ACC.where(lsm > .8)
model_2wrT_summer_ACC = model_2wrT_summer_ACC.where(lsm > .8)
model_1wrT_winter_ACC = model_1wrT_winter_ACC.where(lsm > .8)
model_1wrT_summer_ACC = model_1wrT_summer_ACC.where(lsm > .8)
model_0wrT_winter_ACC = model_0wrT_winter_ACC.where(lsm > .8)
model_0wrT_summer_ACC = model_0wrT_summer_ACC.where(lsm > .8)

model_7wrP_winter_ACC = model_7wrP_winter_ACC.where(lsm > .8)
model_7wrP_summer_ACC = model_7wrP_summer_ACC.where(lsm > .8)
model_4wrP_winter_ACC = model_4wrP_winter_ACC.where(lsm > .8)
model_4wrP_summer_ACC = model_4wrP_summer_ACC.where(lsm > .8)
model_2wrP_winter_ACC = model_2wrP_winter_ACC.where(lsm > .8)
model_2wrP_summer_ACC = model_2wrP_summer_ACC.where(lsm > .8)
model_1wrP_winter_ACC = model_1wrP_winter_ACC.where(lsm > .8)
model_1wrP_summer_ACC = model_1wrP_summer_ACC.where(lsm > .8)
model_0wrP_winter_ACC = model_0wrP_winter_ACC.where(lsm > .8)
model_0wrP_summer_ACC = model_0wrP_summer_ACC.where(lsm > .8)

model_7wrT_winter_CE = model_7wrT_winter_CE.where(lsm > .8)
model_7wrT_summer_CE = model_7wrT_summer_CE.where(lsm > .8)
model_4wrT_winter_CE = model_4wrT_winter_CE.where(lsm > .8)
model_4wrT_summer_CE = model_4wrT_summer_CE.where(lsm > .8)
model_2wrT_winter_CE = model_2wrT_winter_CE.where(lsm > .8)
model_2wrT_summer_CE = model_2wrT_summer_CE.where(lsm > .8)
model_1wrT_winter_CE = model_1wrT_winter_CE.where(lsm > .8)
model_1wrT_summer_CE = model_1wrT_summer_CE.where(lsm > .8)
model_0wrT_winter_CE = model_0wrT_winter_CE.where(lsm > .8)
model_0wrT_summer_CE = model_0wrT_summer_CE.where(lsm > .8)

model_7wrP_winter_CE = model_7wrP_winter_CE.where(lsm > .8)
model_7wrP_summer_CE = model_7wrP_summer_CE.where(lsm > .8)
model_4wrP_winter_CE = model_4wrP_winter_CE.where(lsm > .8)
model_4wrP_summer_CE = model_4wrP_summer_CE.where(lsm > .8)
model_2wrP_winter_CE = model_2wrP_winter_CE.where(lsm > .8)
model_2wrP_summer_CE = model_2wrP_summer_CE.where(lsm > .8)
model_1wrP_winter_CE = model_1wrP_winter_CE.where(lsm > .8)
model_1wrP_summer_CE = model_1wrP_summer_CE.where(lsm > .8)
model_0wrP_winter_CE = model_0wrP_winter_CE.where(lsm > .8)
model_0wrP_summer_CE = model_0wrP_summer_CE.where(lsm > .8)


# ----------------
# Print skills
# ----------------
print('Skills computed:')
print('-------------------------------------------------------------------------------------------------------------------------------------')
print('                                                                  Temperature:')
print('-------------------------------------------------------------------------------------------------------------------------------------')
print('                                         ACC                                                         CE')
print('                 7WR         4WR         2WR         1WR         0WR         7WR         4WR         2WR         1WR         0WR')
print('winter        %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f' %\
       (model_7wrT_winter_ACC.median().values, anom_7wr_temp_winter_ACC_err, model_4wrT_winter_ACC.median().values, anom_4wr_temp_winter_ACC_err, model_2wrT_winter_ACC.median().values, anom_2wr_temp_winter_ACC_err, model_1wrT_winter_ACC.median().values, anom_1wr_temp_winter_ACC_err, model_0wrT_winter_ACC.median().values, anom_0wr_temp_winter_ACC_err, \
        model_7wrT_winter_CE.median().values, anom_7wr_temp_winter_CE_err, model_4wrT_winter_CE.median().values, anom_4wr_temp_winter_CE_err, model_2wrT_winter_CE.median().values, anom_2wr_temp_winter_CE_err, model_1wrT_winter_CE.median().values, anom_1wr_temp_winter_CE_err, model_0wrT_winter_CE.median().values, anom_0wr_temp_winter_CE_err)
)
print('summer        %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f' % \
      (model_7wrT_summer_ACC.median().values, anom_7wr_temp_summer_ACC_err, model_4wrT_summer_ACC.median().values, anom_4wr_temp_summer_ACC_err, model_2wrT_summer_ACC.median().values, anom_2wr_temp_summer_ACC_err, model_1wrT_summer_ACC.median().values, anom_1wr_temp_summer_ACC_err, model_0wrT_summer_ACC.median().values, anom_0wr_temp_summer_ACC_err, \
      model_7wrT_summer_CE.median().values, anom_7wr_temp_summer_CE_err, model_4wrT_summer_CE.median().values, anom_4wr_temp_summer_CE_err, model_2wrT_summer_CE.median().values, anom_2wr_temp_summer_CE_err, model_1wrT_summer_CE.median().values, anom_1wr_temp_summer_CE_err, model_0wrT_summer_CE.median().values, anom_0wr_temp_summer_CE_err,)
                                                                        )
print('-------------------------------------------------------------------------------------------------------------------------------------')
print('                                                                  Precipitation:')
print('-------------------------------------------------------------------------------------------------------------------------------------')
print('                                         ACC                                                         CE')
print('                 7WR         4WR         2WR         1WR         0WR         7WR         4WR         2WR         1WR         0WR')
print('winter        %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f' % \
      (model_7wrP_winter_ACC.median().values, anom_7wr_prec_winter_ACC_err, model_4wrP_winter_ACC.median().values, anom_4wr_prec_winter_ACC_err, model_2wrP_winter_ACC.median().values, anom_2wr_prec_winter_ACC_err, model_1wrP_winter_ACC.median().values, anom_1wr_prec_winter_ACC_err, model_0wrP_winter_ACC.median().values, anom_0wr_prec_winter_ACC_err, \
      model_7wrP_winter_CE.median().values, anom_7wr_prec_winter_CE_err, model_4wrP_winter_CE.median().values, anom_4wr_prec_winter_CE_err, model_2wrP_winter_CE.median().values, anom_2wr_prec_winter_CE_err, model_1wrP_winter_CE.median().values, anom_1wr_prec_winter_CE_err, model_0wrP_winter_CE.median().values, anom_0wr_prec_winter_CE_err)
                                                                        )
print('summer        %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f  %.2f+-%.2f' % \
      (model_7wrP_summer_ACC.median().values, anom_7wr_prec_summer_ACC_err, model_4wrP_summer_ACC.median().values, anom_4wr_prec_summer_ACC_err, model_2wrP_summer_ACC.median().values, anom_2wr_prec_summer_ACC_err, model_1wrP_summer_ACC.median().values, anom_1wr_prec_summer_ACC_err, model_0wrP_summer_ACC.median().values, anom_0wr_prec_summer_ACC_err, \
       model_7wrP_summer_CE.median().values, anom_7wr_prec_summer_CE_err, model_4wrP_summer_CE.median().values, anom_4wr_prec_summer_CE_err, model_2wrP_summer_CE.median().values, anom_2wr_prec_summer_CE_err, model_1wrP_summer_CE.median().values, anom_1wr_prec_summer_CE_err, model_0wrP_summer_CE.median().values, anom_0wr_prec_summer_CE_err)
)
print('-------------------------------------------------------------------------------------------------------------------------------------')

# -----------
# Plot skills
# -----------
print('Plotting...')

columnwidth = 236# 248.
wr7_color = '#b81b22'
wr4_color = '#fdaa31'
wr2_color = '#6bb4e1'
wr1_color = '#204487'
wr0_color = "#111111"
fig, axs_sx = plt.subplots(
    2, 2, figsize=set_figsize(columnwidth, 1, subplots=(2,2)),
    layout="constrained", sharex=True, sharey=True, gridspec_kw={'wspace': 0, 'hspace': 0.6}
)

axs_dx = np.array([[axs_sx[0,0].twinx(), axs_sx[0,1].twinx()], [axs_sx[1,0].twinx(), axs_sx[1,1].twinx()]])
#axs_sx[0,0].set_ylabel('MAE')
#axs_sx[1,0].set_ylabel('MAE')
axs_sx[0,0].set_ylabel('ACC')
axs_sx[1,0].set_ylabel('ACC')
axs_dx[0,1].set_ylabel('CE')
axs_dx[1,1].set_ylabel('CE')
axs_sx[0,0].text(0.45, 1.3, 'two-meter temperature', fontsize=9.5, transform=axs_sx[0,0].transAxes)
axs_sx[0,0].set_title('winter (DJF)', fontsize=8.5)
axs_sx[0,1].set_title('summer (JJA)', fontsize=8.5)
axs_sx[1,0].text(0.7, 1.3, 'precipitation', fontsize=9.5, transform=axs_sx[1,0].transAxes)
axs_sx[1,0].set_title('winter (DJF)', fontsize=8.5)
axs_sx[1,1].set_title('summer (JJA)', fontsize=8.5)
axs_dx[0,0].set_xlim(-0.6, 6)
#[axs_sx[i,j].set_ylim(0, 1.5) for i in range(2) for j in range(2)]
[axs_sx[i,j].set_ylim(-.4, 1) for i in range(2) for j in range(2)]
[axs_dx[i,j].set_ylim(-.4, 1) for i in range(2) for j in range(2)]

# ----------------
# Plot temperature
# ----------------
# 7WR winter
#plot_boxplot(axs_sx[0,0], model_7wrT_winter_MAE.data.flatten(), [-.9,], wr7_color, '*')
plot_boxplot(axs_sx[0,0], model_7wrT_winter_ACC.data.flatten(), [0,], wr7_color, '*')
plot_boxplot(axs_dx[0,0], model_7wrT_winter_CE.data.flatten(), [3,], wr7_color, '*')

# 4WR winter
#plot_boxplot(axs_sx[0,0], model_4wrT_winter_MAE.data.flatten(), [-.3,], wr4_color, '*')
plot_boxplot(axs_sx[0,0], model_4wrT_winter_ACC.data.flatten(), [.6,], wr4_color, '*')
plot_boxplot(axs_dx[0,0], model_4wrT_winter_CE.data.flatten(), [3.6,], wr4_color, '*')

# 2WR winter
#plot_boxplot(axs_sx[0,0], model_2wrT_winter_MAE.data.flatten(), [0.3,], wr2_color, '*')
plot_boxplot(axs_sx[0,0], model_2wrT_winter_ACC.data.flatten(), [1.2,], wr2_color, '*')
plot_boxplot(axs_dx[0,0], model_2wrT_winter_CE.data.flatten(), [4.2,], wr2_color, '*')

# 1WR winter
#plot_boxplot(axs_sx[0,0], model_2wrT_winter_MAE.data.flatten(), [0.3,], wr2_color, '*')
plot_boxplot(axs_sx[0,0], model_1wrT_winter_ACC.data.flatten(), [1.8,], wr1_color, '*')
plot_boxplot(axs_dx[0,0], model_1wrT_winter_CE.data.flatten(), [4.8,], wr1_color, '*')

# 0WR winter
#plot_boxplot(axs_sx[0,0], model_0wrT_winter_MAE.data.flatten(), [.9,], wr0_color, '*')
plot_boxplot(axs_sx[0,0], model_0wrT_winter_ACC.data.flatten(), [2.4,], wr0_color, '*')
plot_boxplot(axs_dx[0,0], model_0wrT_winter_CE.data.flatten(), [5.6,], wr0_color, '*')

# 7WR summer
#plot_boxplot(axs_sx[0,1], model_7wrT_summer_MAE.data.flatten(), [-0.9,], wr7_color, '*')
plot_boxplot(axs_sx[0,1], model_7wrT_summer_ACC.data.flatten(), [0,], wr7_color, '*')
plot_boxplot(axs_dx[0,1], model_7wrT_summer_CE.data.flatten(), [3,], wr7_color, '*')

# 4WR summer
#plot_boxplot(axs_sx[0,1], model_4wrT_summer_MAE.data.flatten(), [-.3,], wr4_color, '*')
plot_boxplot(axs_sx[0,1], model_4wrT_summer_ACC.data.flatten(), [0.6,], wr4_color, '*')
plot_boxplot(axs_dx[0,1], model_4wrT_summer_CE.data.flatten(), [3.6,], wr4_color, '*')

# 2WR summer
#plot_boxplot(axs_sx[0,1], model_2wrT_summer_MAE.data.flatten(), [0.3,], wr2_color, '*')
plot_boxplot(axs_sx[0,1], model_2wrT_summer_ACC.data.flatten(), [1.2,], wr2_color, '*')
plot_boxplot(axs_dx[0,1], model_2wrT_summer_CE.data.flatten(), [4.2,], wr2_color, '*')

# 1WR summer
#plot_boxplot(axs_sx[0,1], model_1wrT_summer_MAE.data.flatten(), [0.9,], wr1_color, '*')
plot_boxplot(axs_sx[0,1], model_1wrT_summer_ACC.data.flatten(), [1.8,], wr1_color, '*')
plot_boxplot(axs_dx[0,1], model_1wrT_summer_CE.data.flatten(), [4.8,], wr1_color, '*')

# 0WR summer
#plot_boxplot(axs_sx[0,1], model_0wrT_summer_MAE.data.flatten(), [0.9,], wr0_color, '*')
plot_boxplot(axs_sx[0,1], model_0wrT_summer_ACC.data.flatten(), [2.4,], wr0_color, '*')
plot_boxplot(axs_dx[0,1], model_0wrT_summer_CE.data.flatten(), [5.4,], wr0_color, '*')

# ------------------
# Plot precipitation
# ------------------
# 7WR winter
#plot_boxplot(axs_sx[1,0], model_7wrP_winter_MAE.data.flatten(), [-0.9,], wr7_color, '*')
plot_boxplot(axs_sx[1,0], model_7wrP_winter_ACC.data.flatten(), [0,], wr7_color, '*')
plot_boxplot(axs_dx[1,0], model_7wrP_winter_CE.data.flatten(), [3,], wr7_color, '*')

# 4WR winter
#plot_boxplot(axs_sx[1,0], model_4wrP_winter_MAE.data.flatten(), [-.3,], wr4_color, '*')
plot_boxplot(axs_sx[1,0], model_4wrP_winter_ACC.data.flatten(), [0.6,], wr4_color, '*')
plot_boxplot(axs_dx[1,0], model_4wrP_winter_CE.data.flatten(), [3.6,], wr4_color, '*')

# 2WR winter
#plot_boxplot(axs_sx[1,0], model_2wrP_winter_MAE.data.flatten(), [0.3,], wr2_color, '*')
plot_boxplot(axs_sx[1,0], model_2wrP_winter_ACC.data.flatten(), [1.2,], wr2_color, '*')
plot_boxplot(axs_dx[1,0], model_2wrP_winter_CE.data.flatten(), [4.2,], wr2_color, '*')

# 1WR winter
#plot_boxplot(axs_sx[1,0], model_1wrP_winter_MAE.data.flatten(), [0.9,], wr1_color, '*')
plot_boxplot(axs_sx[1,0], model_1wrP_winter_ACC.data.flatten(), [1.8,], wr1_color, '*')
plot_boxplot(axs_dx[1,0], model_1wrP_winter_CE.data.flatten(), [4.8,], wr1_color, '*')

# 0WR winter
#plot_boxplot(axs_sx[1,0], model_0wrP_winter_MAE.data.flatten(), [0.9,], wr0_color, '*')
plot_boxplot(axs_sx[1,0], model_0wrP_winter_ACC.data.flatten(), [2.4,], wr0_color, '*')
plot_boxplot(axs_dx[1,0], model_0wrP_winter_CE.data.flatten(), [5.4,], wr0_color, '*')

# 7WR summer
#plot_boxplot(axs_sx[1,1], model_7wrP_summer_MAE.data.flatten(), [-0.9,], wr7_color, '*')
plot_boxplot(axs_sx[1,1], model_7wrP_summer_ACC.data.flatten(), [0,], wr7_color, '*')
plot_boxplot(axs_dx[1,1], model_7wrP_summer_CE.data.flatten(), [3,], wr7_color, '*')

# 4WR summer
#plot_boxplot(axs_sx[1,1], model_4wrP_summer_MAE.data.flatten(), [-0.3,], wr4_color, '*')
plot_boxplot(axs_sx[1,1], model_4wrP_summer_ACC.data.flatten(), [0.6,], wr4_color, '*')
plot_boxplot(axs_dx[1,1], model_4wrP_summer_CE.data.flatten(), [3.6,], wr4_color, '*')

# 2WR summer
#plot_boxplot(axs_sx[1,1], model_2wrP_summer_MAE.data.flatten(), [0.3,], wr2_color, '*')
plot_boxplot(axs_sx[1,1], model_2wrP_summer_ACC.data.flatten(), [1.2,], wr2_color, '*')
plot_boxplot(axs_dx[1,1], model_2wrP_summer_CE.data.flatten(), [4.2,], wr2_color, '*')

# 1WR summer
#plot_boxplot(axs_sx[1,1], model_1wrP_summer_MAE.data.flatten(), [0.9,], wr1_color, '*')
plot_boxplot(axs_sx[1,1], model_1wrP_summer_ACC.data.flatten(), [1.8,], wr1_color, '*')
plot_boxplot(axs_dx[1,1], model_1wrP_summer_CE.data.flatten(), [4.8,], wr1_color, '*')

# 0WR summer
#plot_boxplot(axs_sx[1,1], model_0wrP_summer_MAE.data.flatten(), [0.9,], wr0_color, '*')
plot_boxplot(axs_sx[1,1], model_0wrP_summer_ACC.data.flatten(), [2.4,], wr0_color, '*')
plot_boxplot(axs_dx[1,1], model_0wrP_summer_CE.data.flatten(), [5.4,], wr0_color, '*')

# legend
legend_elements = [
    Patch(color=wr7_color, label='AI-model with 7 WR'),
    Patch(color=wr4_color, label='AI-model with 4 WR'),
    Patch(color=wr2_color, label='AI-model with 2 WR'),
    Patch(color=wr1_color, label='AI-model with 1 WR'),
    Patch(color=wr0_color, label='AI-model with 0 WR'),
]
fig.legend(handles=legend_elements, ncol=2, loc='upper left',  bbox_to_anchor=(-.3, 1.95), bbox_transform=axs_sx[0,0].transAxes)

# set some ticks options
axs_sx[0,0].set_xticks([1.2, 4.2])
axs_sx[0,1].set_xticks([1.2, 4.2])
axs_sx[1,0].set_xticks([1.2, 4.2])
axs_sx[1,1].set_xticks([1.2, 4.2])
axs_sx[0,0].set_xticklabels(['ACC', 'CE'])
axs_sx[0,1].set_xticklabels(['ACC', 'CE'])
axs_sx[1,0].set_xticklabels(['ACC', 'CE'])
axs_sx[1,1].set_xticklabels(['ACC', 'CE'])
[axs_sx[i,j].tick_params(axis='x', which='both', top=False) for i in range(2) for j in range(2)]
[axs_sx[i,j].tick_params(axis='x', which='minor', bottom=False) for i in range(2) for j in range(2)]
axs_dx[0,0].set_yticklabels([])
axs_dx[1,0].set_yticklabels([])


# save
plt.tight_layout()
plt.savefig(f'plots/numWR_skills_test_1WR.pdf', bbox_inches='tight')
