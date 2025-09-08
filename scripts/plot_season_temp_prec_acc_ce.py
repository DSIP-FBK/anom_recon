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
parser.add_argument("-season", type=str, help="season or summer")
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
if args.season == 'winter':
    season = (12,1,2)
elif args.season == 'summer':
    season = (6,7,8)
model_temp_season = model_temp[np.isin(model_temp.time.dt.month, season)]
anom_temp_season  = anom_temp[np.isin(anom_temp.time.dt.month, season)]
model_prec_season = model_prec[np.isin(model_prec.time.dt.month, season)]
anom_prec_season  = anom_prec[np.isin(anom_prec.time.dt.month, season)]

# reduce all variables to common time-range
start, end   = '2011', '2024'
anom_temp_season  = anom_temp_season.sel(time=slice(start, end))
model_temp_season = model_temp_season.sel(time=slice(start, end))
anom_prec_season  = anom_prec_season.sel(time=slice(start, end))
model_prec_season = model_prec_season.sel(time=slice(start, end))

# anomaly correlation coefficient
model_temp_season_acc = pearsonr(anom_temp_season, model_temp_season.mean(dim='number'), axis=0)
model_prec_season_acc = pearsonr(anom_prec_season, model_prec_season.mean(dim='number'), axis=0)

# coefficient of efficacy
model_temp_season_ce = get_ce(anom_temp_season, model_temp_season.mean(dim='number'))
model_prec_season_ce = get_ce(anom_prec_season, model_prec_season.mean(dim='number'))


# ----------------
# plots
# ----------------
# columnwidth = 248.9  # QJRMS
columnwidth = 196.1  # IJC
fig, axs = plt.subplots(
    2, 2, figsize=set_figsize(columnwidth, .8, subplots=(2, 2)), layout="tight",
    sharex=True, sharey=True, gridspec_kw={'wspace':0, 'hspace':0.05},
    subplot_kw={'projection': ccrs.PlateCarree()}
)
axs[0,0].set_xlim(-20,40)
axs[0,0].set_ylim(35,70)
[axs[i,j].coastlines() for j in range(2) for i in range(2)]
[axs[i,j].add_feature(cfeature.OCEAN, facecolor=(1,1,1), zorder=999) for i in range(2) for j in range(2)]

axs[0,0].text(0.14, 1.1, 'temperature (K)', fontsize=8, transform=axs[0,0].transAxes)
axs[0,1].text(0.12, 1.1, 'precipitation (cm)', fontsize=8, transform=axs[0,1].transAxes)

# anomaly correlation coefficient
palette = ['#1b2c62', '#204487', '#2d66af', '#2d66af', '#6bb4e1', '#94d3f3', '#b8e4f8', '#d9f0f9', '#f0f9fd', '#ffffff', '#ffffff', '#fef8de', '#fceda3', '#fdce67', '#fdaa31', '#f8812c', '#ed5729', '#da2f28', '#b81b22', '#921519']
cmap = LinearSegmentedColormap.from_list("", palette)
vmin, vmax = -1, 1
ptresh = .05
levels = np.linspace(-1, 1, 21)

pcm = axs[0,0].contourf(anom_temp.lon, anom_temp.lat, model_temp_season_acc.statistic, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels)
axs[0,1].contourf(anom_prec.lon, anom_prec.lat, model_prec_season_acc.statistic, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels)

h = axs[0,0].contourf(anom_temp.lon, anom_temp.lat, model_temp_season_acc.pvalue > ptresh, hatches=['+++', None], colors='none', vmin=vmin, vmax=vmax, levels=[1-ptresh,1])
h._hatch_color = (0,0,0,1)
h = axs[0,1].contourf(anom_prec.lon, anom_prec.lat, model_prec_season_acc.pvalue > ptresh, hatches=['+++', None], colors='none', vmin=vmin, vmax=vmax, levels=[1-ptresh,1])
h._hatch_color = (0,0,0,1)

axs[0,0].contour(anom_temp.lon, anom_temp.lat, model_temp_season_acc.statistic, ls='--', colors='k', linewidths=.5, levels=[.5])
axs[0,1].contour(anom_prec.lon, anom_prec.lat, model_prec_season_acc.statistic, ls='--', colors='k', linewidths=.5, levels=[.5])

cax = inset_axes(axs[0,-1], width="5%", height="100%", loc="upper right", bbox_to_anchor=(0.15, .06, 1, 1), bbox_transform=axs[0,-1].transAxes)
cb = fig.colorbar(pcm, cax=cax, orientation='vertical', label='ACC')

# coefficient of efficacy
palette = ['#1b2c62', '#204487', '#2d66af', '#2d66af', '#6bb4e1', '#94d3f3', '#b8e4f8', '#d9f0f9', '#f0f9fd', '#ffffff', '#ffffff', '#fef8de', '#fceda3', '#fdce67', '#fdaa31', '#f8812c', '#ed5729', '#da2f28', '#b81b22', '#921519']
cmap = LinearSegmentedColormap.from_list("", palette)
vmin, vmax = -1, 1
levels = np.linspace(-1, 1, 21)

pcm = axs[1,0].contourf(anom_temp.lon, anom_temp.lat, model_temp_season_ce, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, extend='min')
axs[1,1].contourf(anom_prec.lon, anom_prec.lat, model_prec_season_ce, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, extend='min')

axs[1,0].contour(anom_temp.lon, anom_temp.lat, model_temp_season_ce, ls='--', colors='k', linewidths=.5, levels=[.3])
axs[1,1].contour(anom_prec.lon, anom_prec.lat, model_prec_season_ce, ls='--', colors='k', linewidths=.5, levels=[.3])

cax = inset_axes(axs[1,-1], width="5%", height="100%", loc="upper right", bbox_to_anchor=(0.15, .06, 1, 1), bbox_transform=axs[1,-1].transAxes)
fig.colorbar(pcm, cax=cax, orientation='vertical', label='CE')

# save
[axs[i,j].set_aspect('auto') for j in range(2) for i in range(2)]
plt.savefig(f'plots/{args.season}_temp_prec_skills_0wr.pdf', bbox_inches='tight')
