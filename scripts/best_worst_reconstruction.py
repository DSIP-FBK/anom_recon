import argparse
import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# custom functions
from functions import get_torch_models_infos, get_models_out, set_figsize

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("-temp_model", type=str, help="path to the folder containing the trained model for temperature")
parser.add_argument("-prec_model", type=str, help="path to the folder containing the trained model for precipitation")
parser.add_argument("-start", type=str, default='2011', help="start date of the analysis (default 2011)")
args = parser.parse_args()

# parameters
lat_min, lat_max = 35, 70
lon_min, lon_max = -20, 30

# models for temperature
torch_model, datamodule, config= get_torch_models_infos(args.temp_model)
anom_temp  = xr.open_dataarray(config['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
idxs_temp  = xr.open_dataarray(config['data']['indexes_paths'][0]).sel(mode=slice(1, config['data']['num_indexes'][0]))
model_temp = get_models_out(torch_model, idxs_temp, anom_temp, datamodule).mean(dim='number')

# models for precipitation
torch_model, datamodule, config = get_torch_models_infos(args.prec_model)
anom_prec  = xr.open_dataarray(config['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'}) * 100 # m to cm
idxs_prec  = xr.open_dataarray(config['data']['indexes_paths'][0]).sel(mode=slice(1, config['data']['num_indexes'][0]))
model_prec = get_models_out(torch_model, idxs_prec, anom_prec, datamodule).mean(dim='number') * 100 # m to cm

# cut to the European region
anom_temp = anom_temp.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
anom_prec = anom_prec.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
model_temp = model_temp.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
model_prec = model_prec.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))

# reduce all variables to common time-range
start, end = args.start, '2024'
anom_temp  = anom_temp.sel(time=slice(start, end))
model_temp = model_temp.sel(time=slice(start, end))
anom_prec  = anom_prec.sel(time=slice(start, end))
model_prec = model_prec.sel(time=slice(start, end))

# absolute error 
model_temp_err = ((anom_temp - model_temp)**2).mean(dim=['lat', 'lon'])
model_prec_err = ((anom_prec - model_prec)**2).mean(dim=['lat', 'lon'])

# best temperature reconstruction
time_best_temp = model_temp_err[model_temp_err.argmin()].time
model_temp_best = model_temp.sel(time=time_best_temp)

# best precipitation reconstruction
time_best_prec = model_prec_err[model_prec_err.argmin()].time
model_prec_best = model_prec.sel(time=time_best_prec)

# worst temperature reconstruction
time_worst_temp = model_temp_err[model_temp_err.argmax()].time
model_temp_worst = model_temp.sel(time=time_worst_temp)

# worst precipitation reconstruction
time_worst_prec = model_prec_err[model_prec_err.argmax()].time
model_prec_worst = model_prec.sel(time=time_worst_prec)

print('                time of best                         time of worst')
print(f'temperature     {time_best_temp.data}        {time_worst_temp.data}')
print(f'precipitation   {time_best_prec.data}        {time_worst_prec.data}')

textwidth = 504
rasterized=True
palette = ['#1b2c62', '#204487', '#2d66af', '#2d66af', '#6bb4e1', '#94d3f3', '#b8e4f8', '#d9f0f9', '#f0f9fd', '#ffffff', '#ffffff', '#fef8de', '#fceda3', '#fdce67', '#fdaa31', '#f8812c', '#ed5729', '#da2f28', '#b81b22', '#921519']
cmap = LinearSegmentedColormap.from_list("", palette)
fig, axs = plt.subplots(
    2, 5, figsize=set_figsize(textwidth, .75, subplots=(2, 3)), layout="tight",
    sharex=True, sharey=True, gridspec_kw={'wspace':0, 'hspace':0.55, 'width_ratios' : [1,1,.1,1,1]},
    subplot_kw={'projection': ccrs.PlateCarree()}
)
axs[0,0].set_xlim(lon_min, lon_max)
axs[0,0].set_ylim(lat_min, lat_max)
[axs[i,2].axis('off') for i in range(2)]
[axs[i,j].coastlines() for j in range(2) for i in range(2)]
[axs[i,j].coastlines() for j in range(3,5) for i in range(2)]
[axs[i,j].add_feature(cfeature.OCEAN, facecolor=(1,1,1), zorder=999) for i in range(2) for j in range(5)]

axs[0,0].set_title('ERA5', fontsize=8)
axs[0,1].set_title('AI-Model', fontsize=8)
axs[0,3].set_title('ERA5', fontsize=8)
axs[0,4].set_title('AI-Model', fontsize=8)

# best temperature reconstruction
absmax = int(np.round(abs(anom_temp.sel(time=time_best_temp)).max().data))
levels = np.linspace(-absmax, absmax, (absmax * 4) + 1)
pcm = axs[0,0].contourf(anom_temp.lon, anom_temp.lat, anom_temp.sel(time=time_best_temp), cmap=cmap, vmin=-absmax, vmax=absmax, levels=levels, rasterized=rasterized)
axs[0,1].contourf(anom_temp.lon, anom_temp.lat, model_temp_best, cmap=cmap, vmin=-absmax, vmax=absmax, levels=levels, rasterized=rasterized)

cax = inset_axes(axs[0,0], width="185%", height="5%", loc="lower left", bbox_to_anchor=(0.02, -.15, 1, 1), bbox_transform=axs[0,0].transAxes)
cb = fig.colorbar(pcm, cax=cax, orientation='horizontal', label='temperature (K)')

# best precipitation reconstruction
absmax = int(np.round(abs(anom_prec.sel(time=time_best_prec)).max().data)) - 4 # adapt range for the plot
levels = np.linspace(-absmax, absmax, (absmax * 4) + 1)
pcm = axs[0,3].contourf(anom_prec.lon, anom_prec.lat, anom_prec.sel(time=time_best_prec), cmap=cmap.reversed(), vmin=-absmax, vmax=absmax, levels=levels, rasterized=rasterized)
axs[0,4].contourf(anom_prec.lon, anom_prec.lat, model_prec_best, cmap=cmap.reversed(), vmin=-absmax, vmax=absmax, levels=levels, rasterized=rasterized)

cax = inset_axes(axs[0,3], width="185%", height="5%", loc="lower left", bbox_to_anchor=(0.02, -.15, 1, 1), bbox_transform=axs[0,3].transAxes)
cb = fig.colorbar(pcm, cax=cax, orientation='horizontal', label='precipitation (cm)')

# worst temperature reconstruction
absmax = int(np.round(abs(anom_temp.sel(time=time_worst_temp)).max().data))
levels = np.linspace(-absmax, absmax, (absmax * 2) + 1)
pcm = axs[1,0].contourf(anom_temp.lon, anom_temp.lat, anom_temp.sel(time=time_worst_temp), cmap=cmap, vmin=-absmax, vmax=absmax, levels=levels, rasterized=rasterized)
axs[1,1].contourf(anom_temp.lon, anom_temp.lat, model_temp_worst, cmap=cmap, vmin=-absmax, vmax=absmax, levels=levels, rasterized=rasterized)

cax = inset_axes(axs[1,0], width="185%", height="5%", loc="lower left", bbox_to_anchor=(0.02, -.15, 1, 1), bbox_transform=axs[1,0].transAxes)
cb = fig.colorbar(pcm, cax=cax, orientation='horizontal', label='temperature (K)')

# worst precipitation reconstruction
absmax = int(np.round(abs(anom_prec.sel(time=time_worst_prec)).max().data)) // 2 # adapt range for the plot
levels = np.linspace(-absmax, absmax, (absmax * 2) + 1)
pcm = axs[1,3].contourf(anom_prec.lon, anom_prec.lat, anom_prec.sel(time=time_worst_prec), cmap=cmap.reversed(), vmin=-absmax, vmax=absmax, levels=levels, rasterized=rasterized, extend='max')
axs[1,4].contourf(anom_prec.lon, anom_prec.lat, model_prec_worst, cmap=cmap.reversed(), vmin=-absmax, vmax=absmax, levels=levels, rasterized=rasterized)

cax = inset_axes(axs[1,3], width="185%", height="5%", loc="lower left", bbox_to_anchor=(0.02, -.15, 1, 1), bbox_transform=axs[1,3].transAxes)
cb = fig.colorbar(pcm, cax=cax, orientation='horizontal', label='precipitation (cm)')

# save
[axs[i,j].set_aspect('auto') for j in range(5) for i in range(2)]
plt.savefig(f'plots/best_worst_reconstruction_{start}-{end}.png', bbox_inches='tight', dpi=300)