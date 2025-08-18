import argparse
import numpy as np
import xarray as xr
from scipy.stats import pearsonr

# plotting
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.1
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# custom functions
from functions import *

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("-temp_model", type=str, help="path to the folder containing the trained model for temperature")
parser.add_argument("-prec_model", type=str, help="path to the folder containing the trained model for precipitation")
args = parser.parse_args()

# models for temperature
torch_model, datamodule, config = get_torch_models_infos(args.temp_model)
anom_temp  = xr.open_dataarray(config['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
idxs_temp  = xr.open_dataarray(config['data']['indexes_paths'][0]).sel(mode=slice(1, config['data']['num_indexes'][0]))
model_temp = get_models_out(torch_model, idxs_temp, anom_temp, datamodule)

# models for precipitation
torch_model, datamodule, config = get_torch_models_infos(args.prec_model)
anom_prec  = xr.open_dataarray(config['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
idxs_prec  = xr.open_dataarray(config['data']['indexes_paths'][0]).sel(mode=slice(1, config['data']['num_indexes'][0]))
model_prec = get_models_out(torch_model, idxs_prec, anom_prec, datamodule)

# seasons
winter = (12,1,2)
summer = (6,7,8)
model_temp_winter = model_temp[np.isin(model_temp.time.dt.month, winter)]
anom_temp_winter  = anom_temp[np.isin(anom_temp.time.dt.month, winter)]
model_temp_summer = model_temp[np.isin(model_temp.time.dt.month, summer)]
anom_temp_summer  = anom_temp[np.isin(anom_temp.time.dt.month, summer)]
model_prec_winter = model_prec[np.isin(model_prec.time.dt.month, winter)]
anom_prec_winter  = anom_prec[np.isin(anom_prec.time.dt.month, winter)]
model_prec_summer = model_prec[np.isin(model_prec.time.dt.month, summer)]
anom_prec_summer  = anom_prec[np.isin(anom_prec.time.dt.month, summer)]

# -----------------
# 2011 - 2024
# -----------------
start, end   = '2011', '2024'
anom_temp_winter  = anom_temp_winter.sel(time=slice(start, end))
anom_temp_summer  = anom_temp_summer.sel(time=slice(start, end))
model_temp_winter = model_temp_winter.sel(time=slice(start, end))
model_temp_summer = model_temp_summer.sel(time=slice(start, end))
anom_prec_winter  = anom_prec_winter.sel(time=slice(start, end))
anom_prec_summer  = anom_prec_summer.sel(time=slice(start, end))
model_prec_winter = model_prec_winter.sel(time=slice(start, end))
model_prec_summer = model_prec_summer.sel(time=slice(start, end))

# mean absolute error 
model_temp_winter_mae_2011 = abs(anom_temp_winter - model_temp_winter.mean(dim='number')).mean(dim='time')
model_temp_summer_mae_2011 = abs(anom_temp_summer - model_temp_summer.mean(dim='number')).mean(dim='time')
model_prec_winter_mae_2011 = abs(anom_prec_winter - model_prec_winter.mean(dim='number')).mean(dim='time')
model_prec_summer_mae_2011 = abs(anom_prec_summer - model_prec_summer.mean(dim='number')).mean(dim='time')

# anomaly correlation coefficient
model_temp_winter_acc_2011 = pearsonr(anom_temp_winter, model_temp_winter.mean(dim='number'), axis=0)
model_temp_summer_acc_2011 = pearsonr(anom_temp_summer, model_temp_summer.mean(dim='number'), axis=0)
model_prec_winter_acc_2011 = pearsonr(anom_prec_winter, model_prec_winter.mean(dim='number'), axis=0)
model_prec_summer_acc_2011 = pearsonr(anom_prec_summer, model_prec_summer.mean(dim='number'), axis=0)

# coefficient of efficacy
model_temp_winter_ce_2011 = get_ce(anom_temp_winter, model_temp_winter.mean(dim='number'))
model_temp_summer_ce_2011 = get_ce(anom_temp_summer, model_temp_summer.mean(dim='number'))
model_prec_winter_ce_2011 = get_ce(anom_prec_winter, model_prec_winter.mean(dim='number'))
model_prec_summer_ce_2011 = get_ce(anom_prec_summer, model_prec_summer.mean(dim='number'))


# -----------------
# 2019 - 2024
# -----------------
start, end   = '2019', '2024'
anom_temp_winter  = anom_temp_winter.sel(time=slice(start, end))
anom_temp_summer  = anom_temp_summer.sel(time=slice(start, end))
model_temp_winter = model_temp_winter.sel(time=slice(start, end))
model_temp_summer = model_temp_summer.sel(time=slice(start, end))
anom_prec_winter  = anom_prec_winter.sel(time=slice(start, end))
anom_prec_summer  = anom_prec_summer.sel(time=slice(start, end))
model_prec_winter = model_prec_winter.sel(time=slice(start, end))
model_prec_summer = model_prec_summer.sel(time=slice(start, end))

# mean absolute error 
model_temp_winter_mae_2019 = abs(anom_temp_winter - model_temp_winter.mean(dim='number')).mean(dim='time')
model_temp_summer_mae_2019 = abs(anom_temp_summer - model_temp_summer.mean(dim='number')).mean(dim='time')
model_prec_winter_mae_2019 = abs(anom_prec_winter - model_prec_winter.mean(dim='number')).mean(dim='time')
model_prec_summer_mae_2019 = abs(anom_prec_summer - model_prec_summer.mean(dim='number')).mean(dim='time')

# anomaly correlation coefficient
model_temp_winter_acc_2019 = pearsonr(anom_temp_winter, model_temp_winter.mean(dim='number'), axis=0)
model_temp_summer_acc_2019 = pearsonr(anom_temp_summer, model_temp_summer.mean(dim='number'), axis=0)
model_prec_winter_acc_2019 = pearsonr(anom_prec_winter, model_prec_winter.mean(dim='number'), axis=0)
model_prec_summer_acc_2019 = pearsonr(anom_prec_summer, model_prec_summer.mean(dim='number'), axis=0)

# coefficient of efficacy
model_temp_winter_ce_2019 = get_ce(anom_temp_winter, model_temp_winter.mean(dim='number'))
model_temp_summer_ce_2019 = get_ce(anom_temp_summer, model_temp_summer.mean(dim='number'))
model_prec_winter_ce_2019 = get_ce(anom_prec_winter, model_prec_winter.mean(dim='number'))
model_prec_summer_ce_2019 = get_ce(anom_prec_summer, model_prec_summer.mean(dim='number'))


# ---------------
# differences 
# ---------------
model_temp_winter_mae_diff = model_temp_winter_mae_2011 - model_temp_winter_mae_2019
model_temp_summer_mae_diff = model_temp_summer_mae_2011 - model_temp_summer_mae_2019
model_prec_winter_mae_diff = model_prec_winter_mae_2011 - model_prec_winter_mae_2019
model_prec_summer_mae_diff = model_prec_summer_mae_2011 - model_prec_summer_mae_2019


model_temp_winter_acc_diff = model_temp_winter_acc_2011.statistic - model_temp_winter_acc_2019.statistic
model_temp_summer_acc_diff = model_temp_summer_acc_2011.statistic - model_temp_summer_acc_2019.statistic
model_prec_winter_acc_diff = model_prec_winter_acc_2011.statistic - model_prec_winter_acc_2019.statistic
model_prec_summer_acc_diff = model_prec_summer_acc_2011.statistic - model_prec_summer_acc_2019.statistic


model_temp_winter_ce_diff = model_temp_winter_ce_2011 - model_temp_winter_ce_2019
model_temp_summer_ce_diff = model_temp_summer_ce_2011 - model_temp_summer_ce_2019
model_prec_winter_ce_diff = model_prec_winter_ce_2011 - model_prec_winter_ce_2019
model_prec_summer_ce_diff = model_prec_summer_ce_2011 - model_prec_summer_ce_2019

print('Temp   Median MAE    Median ACC    Median CE')
print('Winter    %.2f        %.2f         %.2f' % (model_temp_winter_mae_diff.median(), np.median(model_temp_winter_acc_diff), model_temp_winter_ce_diff.median()))
print('Summer    %.2f        %.2f         %.2f' % (model_temp_summer_mae_diff.median(), np.median(model_temp_summer_acc_diff), model_temp_summer_ce_diff.median()))

print('Prec   Median MAE    Median ACC    Mean CE')
print('Winter    %.2f        %.2f         %.2f' % (model_prec_winter_mae_diff.median(), np.median(model_prec_winter_acc_diff), model_prec_winter_ce_diff.median()))
print('Summer    %.2f        %.2f         %.2f' % (model_prec_summer_mae_diff.median(), np.median(model_prec_summer_acc_diff), model_prec_summer_ce_diff.median()))

