import argparse
import numpy as np
import xarray as xr

# plotting
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

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
parser.add_argument("--anom_temp", type=str, help="path to the NetCDF containing the ERA5 temperature normalized anomalies" )
parser.add_argument("--anom_prec", type=str, help="path to the NetCDF containing the ERA5 precipitation normalized anomalies" )
parser.add_argument("--seas5_temp", type=str, help="path to the NetCDF containing the SEAS5 temperature normalized anomalies forecasts" )
parser.add_argument("--seas5_prec", type=str, help="path to the NetCDF containing the SEAS5 precipitation normalized anomalies forecasts" )
parser.add_argument("--model_seas5_temp", type=str, help="path to the file containing the model trained on temperature with SEAS5 indexes" )
parser.add_argument("--model_seas5_prec", type=str, help="path to the file containing the model trained on precipitation with SEAS5 indexes" )
parser.add_argument("--start", type=str, default='2011', help="start date of the analysis (default 2011)")
args = parser.parse_args()

# perameters
lat_min, lat_max = 35, 70
lon_min, lon_max = -20, 30
winter_months = (12,1,2)
summer_months = (6,7,8)


# ---------------------
# Load models and SEAS5
# ---------------------
print('Loading models and SEAS5...')

# ERA5 anomalies
anomT      = xr.open_dataarray(args.anom_temp).rename({'latitude': 'lat', 'longitude': 'lon'})
anomP      = xr.open_dataarray(args.anom_prec).rename({'latitude': 'lat', 'longitude': 'lon'})

# SEAS5 temperature and precipitation
seas5T      = xr.open_dataarray(args.seas5_temp).rename({'latitude': 'lat', 'longitude': 'lon'})
seas5P      = xr.open_dataarray(args.seas5_prec).rename({'latitude': 'lat', 'longitude': 'lon'})

# model with SEAS5 indices
model_seas5T = xr.open_dataarray(args.model_seas5_temp).mean(dim='number').rename({'ensemble_member': 'number'})
model_seas5P = xr.open_dataarray(args.model_seas5_prec).mean(dim='number').rename({'ensemble_member': 'number'})

# land sea mask
lsm = xr.open_dataarray('../data/lsm_regrid_shift_europe.nc').rename({'latitude': 'lat', 'longitude': 'lon'})

# crop to european region
anomT        = anomT.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
anomP        = anomP.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
seas5T       = seas5T.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
seas5P       = seas5P.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
model_seas5T = model_seas5T.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
model_seas5P = model_seas5P.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
lsm          = lsm.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))

# seasons
seas5T_DJF  = get_SEAS5_season(seas5T, 'winter')
seas5T_JJA  = get_SEAS5_season(seas5T, 'summer')
seas5P_DJF  = get_SEAS5_season(seas5P, 'winter')
seas5P_JJA  = get_SEAS5_season(seas5P, 'summer')

anomT_DJF            = anomT[np.isin(anomT.time.dt.month, winter_months)]
anomT_JJA            = anomT[np.isin(anomT.time.dt.month, summer_months)]
model_seas5T_DJF = get_SEAS5_season(model_seas5T, 'winter')
model_seas5T_JJA = get_SEAS5_season(model_seas5T, 'summer')

anomP_DJF            = anomP[np.isin(anomP.time.dt.month, winter_months)]
anomP_JJA            = anomP[np.isin(anomP.time.dt.month, summer_months)]
model_seas5P_DJF = get_SEAS5_season(model_seas5P, 'winter')
model_seas5P_JJA = get_SEAS5_season(model_seas5P, 'summer')

# reduce all variables to common time-range
start, end = args.start, '2024'
common_time_DJF = np.intersect1d(
    anomT_DJF.sel(time=slice(args.start, end)).time.values, 
    seas5T_DJF.sel(time=slice(args.start, end)).time.values
    )
anomT_DJF            = anomT_DJF.sel(time=common_time_DJF)
seas5T_DJF           = seas5T_DJF.sel(time=common_time_DJF)
model_seas5T_DJF = model_seas5T_DJF.sel(time=common_time_DJF)
anomP_DJF            = anomP_DJF.sel(time=common_time_DJF)
seas5P_DJF           = seas5P_DJF.sel(time=common_time_DJF)
model_seas5P_DJF = model_seas5P_DJF.sel(time=common_time_DJF)

common_time_JJA = np.intersect1d(
    anomT_JJA.sel(time=slice(args.start, end)).time.values, 
    seas5T_JJA.sel(time=slice(args.start, end)).time.values
    )
anomT_JJA            = anomT_JJA.sel(time=common_time_JJA)
seas5T_JJA           = seas5T_JJA.sel(time=common_time_JJA)
model_seas5T_JJA = model_seas5T_JJA.sel(time=common_time_JJA)
anomP_JJA            = anomP_JJA.sel(time=common_time_JJA)
seas5P_JJA           = seas5P_JJA.sel(time=common_time_JJA)
model_seas5P_JJA = model_seas5P_JJA.sel(time=common_time_JJA)

# -------------------
# Sanity check: times
# -------------------
# DJF
assert anomT_DJF.time.equals(seas5T_DJF.time), "Time mismatch: anomT_DJF vs seas5T_DJF"
assert anomT_DJF.time.equals(model_seas5T_DJF.time), "Time mismatch: anomT_DJF vs model_seas5T_DJF"
assert anomP_DJF.time.equals(seas5P_DJF.time), "Time mismatch: anomP_DJF vs seas5P_DJF"
assert anomP_DJF.time.equals(model_seas5P_DJF.time), "Time mismatch: anomP_DJF vs model_seas5P_DJF"

# JJA
assert anomT_JJA.time.equals(seas5T_JJA.time), "Time mismatch: anomT_JJA vs seas5T_JJA"
assert anomT_JJA.time.equals(model_seas5T_JJA.time), "Time mismatch: anomT_JJA vs model_seas5T_JJA"
assert anomP_JJA.time.equals(seas5P_JJA.time), "Time mismatch: anomP_JJA vs seas5P_JJA"
assert anomP_JJA.time.equals(model_seas5P_JJA.time), "Time mismatch: anomP_JJA vs model_seas5P_JJA"

# --------------
# Compute skills
# --------------
print('Computing skills...')

# model CRPS
model_seas5T_DJF_CRPS = crps_vectorized(model_seas5T_DJF, anomT_DJF)
model_seas5T_JJA_CRPS = crps_vectorized(model_seas5T_JJA, anomT_JJA)
model_seas5P_DJF_CRPS = crps_vectorized(model_seas5P_DJF, anomP_DJF) * 100 # m to cm
model_seas5P_JJA_CRPS = crps_vectorized(model_seas5P_JJA, anomP_JJA) * 100 # m to cm

# SEAS5 CRPS
seas5T_DJF_CRPS = crps_vectorized(seas5T_DJF, anomT_DJF)
seas5T_JJA_CRPS = crps_vectorized(seas5T_JJA, anomT_JJA)
seas5P_DJF_CRPS = crps_vectorized(seas5P_DJF, anomP_DJF) * 100 # m to cm
seas5P_JJA_CRPS = crps_vectorized(seas5P_JJA, anomP_JJA) * 100 # m to cm

# model spread skill ratio
model_seas5T_DJF_SSR = (model_seas5T_DJF.std(dim='number', ddof=1).mean(dim='time') / np.sqrt(((anomT_DJF - model_seas5T_DJF.mean(dim=['number']))**2).mean(dim='time')))
model_seas5T_JJA_SSR = (model_seas5T_JJA.std(dim='number', ddof=1).mean(dim='time') / np.sqrt(((anomT_JJA - model_seas5T_JJA.mean(dim=['number']))**2).mean(dim='time')))
model_seas5P_DJF_SSR = (model_seas5P_DJF.std(dim='number', ddof=1).mean(dim='time') / np.sqrt(((anomP_DJF - model_seas5P_DJF.mean(dim=['number']))**2).mean(dim='time')))
model_seas5P_JJA_SSR = (model_seas5P_JJA.std(dim='number', ddof=1).mean(dim='time') / np.sqrt(((anomP_JJA - model_seas5P_JJA.mean(dim=['number']))**2).mean(dim='time')))

# SEAS5 spread skill ratio
seas5T_DJF_SSR = (seas5T_DJF.std(dim='number', ddof=1).mean(dim='time') / np.sqrt(((anomT_DJF - seas5T_DJF.mean(dim='number'))**2).mean(dim='time')))
seas5T_JJA_SSR = (seas5T_JJA.std(dim='number', ddof=1).mean(dim='time') / np.sqrt(((anomT_JJA - seas5T_JJA.mean(dim='number'))**2).mean(dim='time')))
seas5P_DJF_SSR = (seas5P_DJF.std(dim='number', ddof=1).mean(dim='time') / np.sqrt(((anomP_DJF - seas5P_DJF.mean(dim='number'))**2).mean(dim='time')))
seas5P_JJA_SSR = (seas5P_JJA.std(dim='number', ddof=1).mean(dim='time') / np.sqrt(((anomP_JJA - seas5P_JJA.mean(dim='number'))**2).mean(dim='time')))

# model mean absolute error
model_seas5T_DJF_MAE = abs(anomT_DJF - model_seas5T_DJF.mean(dim=['number'])).mean(dim='time')
model_seas5T_JJA_MAE = abs(anomT_JJA - model_seas5T_JJA.mean(dim=['number'])).mean(dim='time')
model_seas5P_DJF_MAE = abs(anomP_DJF - model_seas5P_DJF.mean(dim=['number'])).mean(dim='time') * 100 # m to cm
model_seas5P_JJA_MAE = abs(anomP_JJA - model_seas5P_JJA.mean(dim=['number'])).mean(dim='time') * 100 # m to cm

# SEAS5 mean absolute error
seas5T_DJF_MAE = abs(anomT_DJF - seas5T_DJF.mean(dim='number')).mean(dim='time')
seas5T_JJA_MAE = abs(anomT_JJA - seas5T_JJA.mean(dim='number')).mean(dim='time')
seas5P_DJF_MAE = abs(anomP_DJF - seas5P_DJF.mean(dim='number')).mean(dim='time') * 100 # m to cm
seas5P_JJA_MAE = abs(anomP_JJA - seas5P_JJA.mean(dim='number')).mean(dim='time') * 100 # m to cm

# model anomaly correlation coefficient
model_seas5T_DJF_ACC = xr.corr(anomT_DJF, model_seas5T_DJF.mean(dim=['number']), dim='time')
model_seas5T_JJA_ACC = xr.corr(anomT_JJA, model_seas5T_JJA.mean(dim=['number']), dim='time')
model_seas5P_DJF_ACC = xr.corr(anomP_DJF, model_seas5P_DJF.mean(dim=['number']), dim='time')
model_seas5P_JJA_ACC = xr.corr(anomP_JJA, model_seas5P_JJA.mean(dim=['number']), dim='time')

# SEAS5 anomaly correlation coefficient
seas5T_DJF_ACC = xr.corr(anomT_DJF, seas5T_DJF.mean(dim='number'), dim='time')
seas5T_JJA_ACC = xr.corr(anomT_JJA, seas5T_JJA.mean(dim='number'), dim='time')
seas5P_DJF_ACC = xr.corr(anomP_DJF, seas5P_DJF.mean(dim='number'), dim='time')
seas5P_JJA_ACC = xr.corr(anomP_JJA, seas5P_JJA.mean(dim='number'), dim='time')

# model coefficient of efficacy
model_seas5T_DJF_CE = get_ce(anomT_DJF, model_seas5T_DJF.mean(dim=['number']))
model_seas5T_JJA_CE = get_ce(anomT_JJA, model_seas5T_JJA.mean(dim=['number']))
model_seas5P_DJF_CE = get_ce(anomP_DJF, model_seas5P_DJF.mean(dim=['number']))
model_seas5P_JJA_CE = get_ce(anomP_JJA, model_seas5P_JJA.mean(dim=['number']))

# SEAS coefficient of efficacy
seas5T_DJF_CE = get_ce(anomT_DJF, seas5T_DJF.mean(dim='number'))
seas5T_JJA_CE = get_ce(anomT_JJA, seas5T_JJA.mean(dim='number'))
seas5P_DJF_CE = get_ce(anomP_DJF, seas5P_DJF.mean(dim='number'))
seas5P_JJA_CE = get_ce(anomP_JJA, seas5P_JJA.mean(dim='number'))

# mask outside land
model_seas5T_DJF_CRPS = model_seas5T_DJF_CRPS.where(lsm > .8)
model_seas5T_JJA_CRPS = model_seas5T_JJA_CRPS.where(lsm > .8)
model_seas5P_DJF_CRPS = model_seas5P_DJF_CRPS.where(lsm > .8)
model_seas5P_JJA_CRPS = model_seas5P_JJA_CRPS.where(lsm > .8)
seas5T_DJF_CRPS = seas5T_DJF_CRPS.where(lsm > .8)
seas5T_JJA_CRPS = seas5T_JJA_CRPS.where(lsm > .8)
seas5P_DJF_CRPS = seas5P_DJF_CRPS.where(lsm > .8)
seas5P_JJA_CRPS = seas5P_JJA_CRPS.where(lsm > .8)

model_seas5T_DJF_SSR = model_seas5T_DJF_SSR.where(lsm > .8)
model_seas5T_JJA_SSR = model_seas5T_JJA_SSR.where(lsm > .8)
model_seas5P_DJF_SSR = model_seas5P_DJF_SSR.where(lsm > .8)
model_seas5P_JJA_SSR = model_seas5P_JJA_SSR.where(lsm > .8)
seas5T_DJF_SSR = seas5T_DJF_SSR.where(lsm > .8)
seas5T_JJA_SSR = seas5T_JJA_SSR.where(lsm > .8)
seas5P_DJF_SSR = seas5P_DJF_SSR.where(lsm > .8)
seas5P_JJA_SSR = seas5P_JJA_SSR.where(lsm > .8)

model_seas5T_DJF_MAE = model_seas5T_DJF_MAE.where(lsm > .8)
model_seas5T_JJA_MAE = model_seas5T_JJA_MAE.where(lsm > .8)
model_seas5P_DJF_MAE = model_seas5P_DJF_MAE.where(lsm > .8)
model_seas5P_JJA_MAE = model_seas5P_JJA_MAE.where(lsm > .8)
seas5T_DJF_MAE = seas5T_DJF_MAE.where(lsm > .8)
seas5T_JJA_MAE = seas5T_JJA_MAE.where(lsm > .8)
seas5P_DJF_MAE = seas5P_DJF_MAE.where(lsm > .8)
seas5P_JJA_MAE = seas5P_JJA_MAE.where(lsm > .8)

model_seas5T_DJF_ACC = model_seas5T_DJF_ACC.where(lsm > .8)
model_seas5T_JJA_ACC = model_seas5T_JJA_ACC.where(lsm > .8)
model_seas5P_DJF_ACC = model_seas5P_DJF_ACC.where(lsm > .8)
model_seas5P_JJA_ACC = model_seas5P_JJA_ACC.where(lsm > .8)
seas5T_DJF_ACC = seas5T_DJF_ACC.where(lsm > .8)
seas5T_JJA_ACC = seas5T_JJA_ACC.where(lsm > .8)
seas5P_DJF_ACC = seas5P_DJF_ACC.where(lsm > .8)
seas5P_JJA_ACC = seas5P_JJA_ACC.where(lsm > .8)

model_seas5T_DJF_CE = model_seas5T_DJF_CE.where(lsm > .8)
model_seas5T_JJA_CE = model_seas5T_JJA_CE.where(lsm > .8)
model_seas5P_DJF_CE = model_seas5P_DJF_CE.where(lsm > .8)
model_seas5P_JJA_CE = model_seas5P_JJA_CE.where(lsm > .8)
seas5T_DJF_CE = seas5T_DJF_CE.where(lsm > .8)
seas5T_JJA_CE = seas5T_JJA_CE.where(lsm > .8)
seas5P_DJF_CE = seas5P_DJF_CE.where(lsm > .8)
seas5P_JJA_CE = seas5P_JJA_CE.where(lsm > .8)

# mask CE values below certain threshold (not relevant)
# threshold = -1 # min(model_seas5T_DJF_CE.min(), model_seas5T_JJA_CE.min(), model_seas5P_DJF_CE.min(), model_seas5P_JJA_CE.min())
# model_seas5T_DJF_CE = xr.where(model_seas5T_DJF_CE < threshold, threshold, model_seas5T_DJF_CE)
# model_seas5T_JJA_CE = xr.where(model_seas5T_JJA_CE < threshold, threshold, model_seas5T_JJA_CE)
# model_seas5P_DJF_CE = xr.where(model_seas5P_DJF_CE < threshold, threshold, model_seas5P_DJF_CE)
# model_seas5P_JJA_CE = xr.where(model_seas5P_JJA_CE < threshold, threshold, model_seas5P_JJA_CE)
# seas5T_DJF_CE = xr.where(seas5T_DJF_CE < threshold, threshold, seas5T_DJF_CE)
# seas5T_JJA_CE = xr.where(seas5T_JJA_CE < threshold, threshold, seas5T_JJA_CE)
# seas5P_DJF_CE = xr.where(seas5P_DJF_CE < threshold, threshold, seas5P_DJF_CE)
# seas5P_JJA_CE = xr.where(seas5P_JJA_CE < threshold, threshold, seas5P_JJA_CE)

# -----------
# Plot skills
# -----------
print('Plotting skills...')
columnwidth = 196.1
seas5_color = '#b81b22'
model_color = '#204487'
fig, axs = plt.subplots(
    4, 2, figsize=set_figsize(columnwidth, .5, subplots=(4,2)), layout="constrained", sharex=True, sharey='row'
)
fig.get_layout_engine().set(w_pad=0, h_pad=0, hspace=0, wspace=0)
axs[0,0].text(0.5, 1.15, 'temperature (K)', horizontalalignment='center', transform=axs[0,0].transAxes)
axs[0,0].set_ylabel('CRPS\n(K or cm)', labelpad=9)
axs[1,0].set_ylabel('SSR', labelpad=10)
axs[2,0].set_ylabel('ACC')
axs[3,0].set_ylabel('CE')
axs[0,1].text(0.5, 1.15, 'precipitation (cm)', horizontalalignment='center', transform=axs[0,1].transAxes)

axs[0,0].set_ylim(0, 5)
axs[2,0].set_ylim(-1, 1)
axs[3,0].set_ylim(-1, 1)


# ----------------
# Plot temperature
# ----------------
# SEAS5 winter
#plot_boxplot(axs[0,0], seas5T_DJF_MAE.data.flatten(), [-0.3,], seas5_color, '*')
plot_boxplot(axs[0,0], seas5T_DJF_CRPS.data.flatten(), [-0.3,], seas5_color, '*')
plot_boxplot(axs[1,0], seas5T_DJF_SSR.data.flatten(), [-0.3,], seas5_color, '*')
plot_boxplot(axs[2,0], seas5T_DJF_ACC.data.flatten(), [-0.3,], seas5_color, '*')
plot_boxplot(axs[3,0], seas5T_DJF_CE.data.flatten(), [-0.3,], seas5_color, '*')

# model winter
#plot_boxplot(axs[0,0], model_seas5T_DJF_MAE.data.flatten(), [0.3,], model_color, '*')
plot_boxplot(axs[0,0], model_seas5T_DJF_CRPS.data.flatten(), [0.3,], model_color, '*')
plot_boxplot(axs[1,0], model_seas5T_DJF_SSR.data.flatten(), [0.3,], model_color, '*')
plot_boxplot(axs[2,0], model_seas5T_DJF_ACC.data.flatten(), [0.3,], model_color, '*')
plot_boxplot(axs[3,0], model_seas5T_DJF_CE.data.flatten(), [0.3,], model_color, '*')

# seas summer
#plot_boxplot(axs[0,0], seas5T_JJA_MAE.data.flatten(), [1.7,], seas5_color, '*')
plot_boxplot(axs[0,0], seas5T_JJA_CRPS.data.flatten(), [1.7,], seas5_color, '*')
plot_boxplot(axs[1,0], seas5T_JJA_SSR.data.flatten(), [1.7,], seas5_color, '*')
plot_boxplot(axs[2,0], seas5T_JJA_ACC.data.flatten(), [1.7,], seas5_color, '*')
plot_boxplot(axs[3,0], seas5T_JJA_CE.data.flatten(), [1.7,], seas5_color, '*')

# model summer
#plot_boxplot(axs[0,0], model_seas5T_JJA_MAE.data.flatten(), [2.3,], model_color, '*')
plot_boxplot(axs[0,0], model_seas5T_JJA_CRPS.data.flatten(), [2.3,], model_color, '*')
plot_boxplot(axs[1,0], model_seas5T_JJA_SSR.data.flatten(), [2.3,], model_color, '*')
plot_boxplot(axs[2,0], model_seas5T_JJA_ACC.data.flatten(), [2.3,], model_color, '*')
plot_boxplot(axs[3,0], model_seas5T_JJA_CE.data.flatten(), [2.3,], model_color, '*')

# ------------------
# Plot precipitation
# ------------------
# SEAS5 winter
#plot_boxplot(axs[0,1], seas5P_DJF_MAE.data.flatten(), [-0.3,], seas5_color, '*')
plot_boxplot(axs[0,1], seas5P_DJF_CRPS.data.flatten(), [-0.3,], seas5_color, '*')
plot_boxplot(axs[1,1], seas5P_DJF_SSR.data.flatten(), [-0.3,], seas5_color, '*')
plot_boxplot(axs[2,1], seas5P_DJF_ACC.data.flatten(), [-0.3,], seas5_color, '*')
plot_boxplot(axs[3,1], seas5P_DJF_CE.data.flatten(), [-0.3,], seas5_color, '*')

# model winter
#plot_boxplot(axs[0,1], model_seas5P_DJF_MAE.data.flatten(), [0.3,], model_color, '*')
plot_boxplot(axs[0,1], model_seas5P_DJF_CRPS.data.flatten(), [0.3,], model_color, '*')
plot_boxplot(axs[1,1], model_seas5P_DJF_SSR.data.flatten(), [0.3,], model_color, '*')
plot_boxplot(axs[2,1], model_seas5P_DJF_ACC.data.flatten(), [0.3,], model_color, '*')
plot_boxplot(axs[3,1], model_seas5P_DJF_CE.data.flatten(), [0.3,], model_color, '*')

# seas summer
#plot_boxplot(axs[0,1], seas5P_JJA_MAE.data.flatten(), [1.7,], seas5_color, '*')
plot_boxplot(axs[0,1], seas5P_JJA_CRPS.data.flatten(), [1.7,], seas5_color, '*')
plot_boxplot(axs[1,1], seas5P_JJA_SSR.data.flatten(), [1.7,], seas5_color, '*')
plot_boxplot(axs[2,1], seas5P_JJA_ACC.data.flatten(), [1.7,], seas5_color, '*')
plot_boxplot(axs[3,1], seas5P_JJA_CE.data.flatten(), [1.7,], seas5_color, '*')

# model summer
#plot_boxplot(axs[0,1], model_seas5P_JJA_MAE.data.flatten(), [2.3,], model_color, '*')
plot_boxplot(axs[0,1], model_seas5P_JJA_CRPS.data.flatten(), [2.3,], model_color, '*')
plot_boxplot(axs[1,1], model_seas5P_JJA_SSR.data.flatten(), [2.3,], model_color, '*')
plot_boxplot(axs[2,1], model_seas5P_JJA_ACC.data.flatten(), [2.3,], model_color, '*')
plot_boxplot(axs[3,1], model_seas5P_JJA_CE.data.flatten(), [2.3,], model_color, '*')

# legend
legend_elements = [
    Patch(color=seas5_color, label='SEAS5'),
    Patch(color=model_color, label=r'AI-model with $I_{\rm wr}^{\rm SEAS5}$'),
]
fig.legend(handles=legend_elements, ncol=2, loc='upper left',  bbox_to_anchor=(-.2, 1.8), bbox_transform=axs[0,0].transAxes)

# set some ticks options
[axs[i,0].set_xticks([0.1, 1.9]) for i in range(3)]
axs[2,0].set_xticklabels(['DJF', 'JJA'])
[axs[i,j].tick_params(axis='x', which='both', top=False) for i in range(3) for j in range(2)]
[axs[i,j].tick_params(axis='x', which='both', bottom=False) for i in range(3) for j in range(2)]
axs[0,0].set_yticks([1, 3])
axs[1,0].set_yticks([0.5, 1])
axs[2,0].set_yticks([-0.5, 0, 0.5])
axs[3,0].set_yticks([-0.5, 0, 0.5])

# lines separating seasons
[axs[i,j].axvline(x=1, color='k', linestyle='--', linewidth=.5) for i in range(4) for j in range(2)]

# save
plt.savefig(f'plots/IwrSEAS5_skills_{start}-{end}.pdf', bbox_inches='tight')
