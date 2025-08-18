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
parser.add_argument("-start", type=str, default='2011', help="start date of the analysis (default 2011)")
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

# reduce all variables to common time-range
start, end   = args.start, '2024'
anom_temp_winter  = anom_temp_winter.sel(time=slice(start, end))
anom_temp_summer  = anom_temp_summer.sel(time=slice(start, end))
model_temp_winter = model_temp_winter.sel(time=slice(start, end))
model_temp_summer = model_temp_summer.sel(time=slice(start, end))
anom_prec_winter  = anom_prec_winter.sel(time=slice(start, end))
anom_prec_summer  = anom_prec_summer.sel(time=slice(start, end))
model_prec_winter = model_prec_winter.sel(time=slice(start, end))
model_prec_summer = model_prec_summer.sel(time=slice(start, end))

# mean absolute error 
model_temp_winter_mae = abs(anom_temp_winter - model_temp_winter.mean(dim='number')).mean(dim='time')
model_temp_summer_mae = abs(anom_temp_summer - model_temp_summer.mean(dim='number')).mean(dim='time')
model_prec_winter_mae = abs(anom_prec_winter - model_prec_winter.mean(dim='number')).mean(dim='time')
model_prec_summer_mae = abs(anom_prec_summer - model_prec_summer.mean(dim='number')).mean(dim='time')

# anomaly correlation coefficient
model_temp_winter_acc = pearsonr(anom_temp_winter, model_temp_winter.mean(dim='number'), axis=0)
model_temp_summer_acc = pearsonr(anom_temp_summer, model_temp_summer.mean(dim='number'), axis=0)
model_prec_winter_acc = pearsonr(anom_prec_winter, model_prec_winter.mean(dim='number'), axis=0)
model_prec_summer_acc = pearsonr(anom_prec_summer, model_prec_summer.mean(dim='number'), axis=0)

# coefficient of efficacy
model_temp_winter_ce = get_ce(anom_temp_winter, model_temp_winter.mean(dim='number'))
model_temp_summer_ce = get_ce(anom_temp_summer, model_temp_summer.mean(dim='number'))
model_prec_winter_ce = get_ce(anom_prec_winter, model_prec_winter.mean(dim='number'))
model_prec_summer_ce = get_ce(anom_prec_summer, model_prec_summer.mean(dim='number'))

print('Temp   Median MAE    Median ACC    Median CE')
print('Winter    %.2f        %.2f         %.2f' % (model_temp_winter_mae.median(), np.median(model_temp_winter_acc.statistic), model_temp_winter_ce.median()))
print('Summer    %.2f        %.2f         %.2f' % (model_temp_summer_mae.median(), np.median(model_temp_summer_acc.statistic), model_temp_summer_ce.median()))

print('Prec   Median MAE    Median ACC    Mean CE')
print('Winter    %.2f        %.2f         %.2f' % (model_prec_winter_mae.median(), np.median(model_prec_winter_acc.statistic), model_prec_winter_ce.median()))
print('Summer    %.2f        %.2f         %.2f' % (model_prec_summer_mae.median(), np.median(model_prec_summer_acc.statistic), model_prec_summer_ce.median()))


# ----------------
# plots
# ----------------
textwidth = 585 #509  # QJRMS
textwidth = 460 #405  # IJC
rasterized=True
palette = ['#ffffff', '#fef8de', '#fceda3', '#fdce67', '#fdaa31', '#f8812c', '#ed5729', '#da2f28', '#b81b22', '#921519']
cmap = LinearSegmentedColormap.from_list("", palette)
vmin, vmax = 0, 1.5
levels = np.linspace(0, 1.5, 16)
fig, axs = plt.subplots(
    3, 5, figsize=set_figsize(textwidth, .6, subplots=(3, 3)), layout="tight",
    sharex=True, sharey=True, gridspec_kw={'wspace':0, 'hspace':0.05, 'width_ratios' : [1,1,.05,1,1]},
    subplot_kw={'projection': ccrs.PlateCarree()}
)
axs[0,0].set_xlim(-20,40)
axs[0,0].set_ylim(35,70)
[axs[i,2].axis('off') for i in range(3)]
[axs[i,j].coastlines() for j in range(2) for i in range(3)]
[axs[i,j].coastlines() for j in range(3,5) for i in range(3)]
[axs[i,j].add_feature(cfeature.OCEAN, facecolor=(1,1,1), zorder=999) for i in range(3) for j in range(5)]

axs[0,0].text(0.35, 1.3, 'two-meter temperature', fontsize=9.5, transform=axs[0,0].transAxes)
axs[0,3].text(0.6, 1.3, 'precipitation', fontsize=9.5, transform=axs[0,3].transAxes)
axs[0,0].set_title('winter (DJF)', fontsize=8)
axs[0,1].set_title('summer (JJA)', fontsize=8)
axs[0,3].set_title('winter (DJF)', fontsize=8)
axs[0,4].set_title('summer (JJA)', fontsize=8)

# mean absolute error
axs[0,0].contourf(anom_temp.lon, anom_temp.lat, model_temp_winter_mae, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, extend='max', rasterized=rasterized)
pcm = axs[0,1].contourf(anom_temp.lon, anom_temp.lat, model_temp_summer_mae, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, rasterized=rasterized)
axs[0,3].contourf(anom_prec.lon, anom_prec.lat, model_prec_winter_mae, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, rasterized=rasterized)
axs[0,4].contourf(anom_prec.lon, anom_prec.lat, model_prec_summer_mae, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, rasterized=rasterized)

cax = inset_axes(axs[0,-1], width="5%", height="100%", loc="upper right", bbox_to_anchor=(0.15, .04, 1, 1), bbox_transform=axs[0,-1].transAxes)
cb = fig.colorbar(pcm, cax=cax, ticks=np.arange(0.2, 1.5, .3), orientation='vertical', extend='max')
cb.set_label('MAE', labelpad=11)

# anomaly correlation coefficient
palette = ['#1b2c62', '#204487', '#2d66af', '#2d66af', '#6bb4e1', '#94d3f3', '#b8e4f8', '#d9f0f9', '#f0f9fd', '#ffffff', '#ffffff', '#fef8de', '#fceda3', '#fdce67', '#fdaa31', '#f8812c', '#ed5729', '#da2f28', '#b81b22', '#921519']
cmap = LinearSegmentedColormap.from_list("", palette)
vmin, vmax = -1, 1
ptresh = .05
levels = np.linspace(-1, 1, 21)

pcm = axs[1,0].contourf(anom_temp.lon, anom_temp.lat, model_temp_winter_acc.statistic, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, rasterized=rasterized)
axs[1,1].contourf(anom_temp.lon, anom_temp.lat, model_temp_summer_acc.statistic, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, rasterized=rasterized)
axs[1,3].contourf(anom_prec.lon, anom_prec.lat, model_prec_winter_acc.statistic, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, rasterized=rasterized)
axs[1,4].contourf(anom_prec.lon, anom_prec.lat, model_prec_summer_acc.statistic, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, rasterized=rasterized)

h = axs[1,0].contourf(anom_temp.lon, anom_temp.lat, model_temp_winter_acc.pvalue > ptresh, hatches=['+++', None], colors='none', vmin=vmin, vmax=vmax, levels=[1-ptresh,1], rasterized=rasterized)
h._hatch_color = (0,0,0,1)
h = axs[1,1].contourf(anom_temp.lon, anom_temp.lat, model_temp_summer_acc.pvalue > ptresh, hatches=['+++', None], colors='none', vmin=vmin, vmax=vmax, levels=[1-ptresh,1], rasterized=rasterized)
h._hatch_color = (0,0,0,1)
h = axs[1,3].contourf(anom_prec.lon, anom_prec.lat, model_prec_winter_acc.pvalue > ptresh, hatches=['+++', None], colors='none', vmin=vmin, vmax=vmax, levels=[1-ptresh,1], rasterized=rasterized)
h._hatch_color = (0,0,0,1)
h = axs[1,4].contourf(anom_prec.lon, anom_prec.lat, model_prec_summer_acc.pvalue > ptresh, hatches=['+++', None], colors='none', vmin=vmin, vmax=vmax, levels=[1-ptresh,1], rasterized=rasterized)
h._hatch_color = (0,0,0,1)

axs[1,0].contour(anom_temp.lon, anom_temp.lat, model_temp_winter_acc.statistic, colors='k', linewidths=.5, levels=[.5], rasterized=rasterized)
axs[1,1].contour(anom_temp.lon, anom_temp.lat, model_temp_summer_acc.statistic, colors='k', linewidths=.5, levels=[.5], rasterized=rasterized)
axs[1,3].contour(anom_prec.lon, anom_prec.lat, model_prec_winter_acc.statistic, colors='k', linewidths=.5, levels=[.5], rasterized=rasterized)
axs[1,4].contour(anom_prec.lon, anom_prec.lat, model_prec_summer_acc.statistic, colors='k', linewidths=.5, levels=[.5], rasterized=rasterized)

cax = inset_axes(axs[1,-1], width="5%", height="100%", loc="upper right", bbox_to_anchor=(0.15, .04, 1, 1), bbox_transform=axs[1,-1].transAxes)
cb = fig.colorbar(pcm, cax=cax, orientation='vertical', label='ACC')

# coefficient of efficacy
palette = ['#1b2c62', '#204487', '#2d66af', '#2d66af', '#6bb4e1', '#94d3f3', '#b8e4f8', '#d9f0f9', '#f0f9fd', '#ffffff', '#ffffff', '#fef8de', '#fceda3', '#fdce67', '#fdaa31', '#f8812c', '#ed5729', '#da2f28', '#b81b22', '#921519']
cmap = LinearSegmentedColormap.from_list("", palette)
vmin, vmax = -1, 1
levels = np.linspace(-1, 1, 21)

pcm = axs[2,0].contourf(anom_temp.lon, anom_temp.lat, model_temp_winter_ce, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, rasterized=rasterized)
axs[2,1].contourf(anom_temp.lon, anom_temp.lat, model_temp_summer_ce, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, extend='min', rasterized=rasterized)
axs[2,3].contourf(anom_prec.lon, anom_prec.lat, model_prec_winter_ce, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, extend='min', rasterized=rasterized)
axs[2,4].contourf(anom_prec.lon, anom_prec.lat, model_prec_summer_ce, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, extend='min', rasterized=rasterized)

axs[2,0].contour(anom_temp.lon, anom_temp.lat, model_temp_winter_ce, ls='--', colors='k', linewidths=.5, levels=[.3], rasterized=rasterized)
axs[2,1].contour(anom_temp.lon, anom_temp.lat, model_temp_summer_ce, ls='--', colors='k', linewidths=.5, levels=[.3], rasterized=rasterized)
axs[2,3].contour(anom_prec.lon, anom_prec.lat, model_prec_winter_ce, ls='--', colors='k', linewidths=.5, levels=[.3], rasterized=rasterized)
axs[2,4].contour(anom_prec.lon, anom_prec.lat, model_prec_summer_ce, ls='--', colors='k', linewidths=.5, levels=[.3], rasterized=rasterized)

cax = inset_axes(axs[2,-1], width="5%", height="100%", loc="upper right", bbox_to_anchor=(0.15, .04, 1, 1), bbox_transform=axs[2,-1].transAxes)
fig.colorbar(pcm, cax=cax, orientation='vertical', label='CE')

# save
[axs[i,j].set_aspect('auto') for j in range(5) for i in range(3)]
plt.savefig(f'plots/temp_prec_skills_7wr_{start}-{end}.png', bbox_inches='tight', dpi=300)
