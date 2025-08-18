import argparse
import numpy as np
import xarray as xr
from scipy.stats import pearsonr

# plotting
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.1
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cartopy.crs as ccrs

# custom functions
from functions import *

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--wr7", type=str, help="path to the folder containing the trained model with 7 WR")
parser.add_argument("--wr4", type=str, help="path to the folder containing the trained model with 4 WR" )
parser.add_argument("--seas5", type=str, help="path to the bias-corrected seas5 anomalies")
parser.add_argument("--season", type=str, help="(str) winter or summer")
args = parser.parse_args()

# season
if args.season == 'winter':
    months = (12,1,2)
elif args.season == 'summer':
    months = (6,7,8)
else:
    print('--season must be winter or summer')
    exit(0)

# models using 7 wr index as input
torch_models_7wr, datamodule_7wr, config_7wr = get_torch_models_infos(args.wr7)
anom_7wr       = xr.open_dataarray(config_7wr['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
idxs_7wr       = xr.open_dataarray(config_7wr['data']['indexes_paths'][0]).sel(mode=slice(1, config_7wr['data']['num_indexes'][0]))
models_7wr     = get_models_out(torch_models_7wr, idxs_7wr, anom_7wr, datamodule_7wr)
models_7wr_sea = models_7wr[np.isin(models_7wr.time.dt.month, months)]
anom_7wr_sea   = anom_7wr[np.isin(anom_7wr.time.dt.month, months)]

# models using 4 wr index as input
torch_models_4wr, datamodule_4wr, config_4wr = get_torch_models_infos(args.wr4)
anom_4wr       = xr.open_dataarray(config_4wr['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
idxs_4wr       = xr.open_dataarray(config_4wr['data']['indexes_paths'][0]).sel(mode=slice(1, config_4wr['data']['num_indexes'][0]))
models_4wr     = get_models_out(torch_models_4wr, idxs_4wr, anom_4wr, datamodule_4wr)
models_4wr_sea = models_4wr[np.isin(models_4wr.time.dt.month, months)]
anom_4wr_sea   = anom_4wr[np.isin(anom_4wr.time.dt.month, months)]

# SEAS5
anom_seas5      = xr.open_dataarray(args.seas5).rename({'latitude': 'lat', 'longitude': 'lon'}).transpose('time', 'lat', 'lon', 'number', 'forecastMonth')
anom_seas5_sea  = get_SEAS5_season(anom_seas5, args.season)

# anomalies are the same for all models (are just era5)
anom     = anom_7wr
anom_sea = anom_7wr_sea

# reduce all variables to common time-range
start, end     = '2011', '2024'
seas5_start    = start
seas5_end      = end
anom_sea       = anom_sea.sel(time=slice(start, end))
anom_seas5     = anom_seas5.sel(time=slice(start, end))
models_7wr_sea = models_7wr_sea.sel(time=slice(start, end))
models_4wr_sea = models_4wr_sea.sel(time=slice(start, end))

# mean absolute error 
seas5_sea_mae      = abs(anom_sea - anom_seas5_sea.mean(dim='number')).mean(dim='time')
models_7wr_sea_mae = abs(anom_sea - models_7wr_sea.mean(dim='number')).mean(dim='time')
models_4wr_sea_mae = abs(anom_sea - models_4wr_sea.mean(dim='number')).mean(dim='time')

# anomaly correlation coefficient
seas5_sea_acc      = pearsonr(anom_sea.sel(time=slice(seas5_start, seas5_end)), anom_seas5_sea.mean(dim='number'), axis=0)
models_7wr_sea_acc = pearsonr(anom_sea, models_7wr_sea.mean(dim='number'), axis=0)
models_4wr_sea_acc = pearsonr(anom_sea, models_4wr_sea.mean(dim='number'), axis=0)

# coefficient of efficacy
seas5_sea_ce      = get_ce(anom_sea.sel(time=slice(seas5_start, seas5_end)), anom_seas5_sea.mean(dim='number'))
models_7wr_sea_ce = get_ce(anom_sea, models_7wr_sea.mean(dim='number'))
models_4wr_sea_ce = get_ce(anom_sea, models_4wr_sea.mean(dim='number'))

print('         Mean MAE    Mean ACC    Mean CE')
print('SEAS5      %.2f       %.2f        %.2f' % (seas5_sea_mae.mean(), seas5_sea_acc.statistic.mean(), seas5_sea_ce.mean()))
print('Model 7WR  %.2f       %.2f        %.2f' % (models_7wr_sea_mae.mean(), models_7wr_sea_acc.statistic.mean(), models_7wr_sea_ce.mean()))
print('Model 4WR  %.2f       %.2f        %.2f' % (models_4wr_sea_mae.mean(), models_4wr_sea_acc.statistic.mean(), models_4wr_sea_ce.mean()))

# ----------------
# plots
# ----------------
textwidth = 509
fig, axs = plt.subplots(
    3, 3, figsize=set_figsize(textwidth, .7, subplots=(3, 3)), layout="constrained",
    sharex=True, sharey=True, gridspec_kw={'wspace':0, 'hspace':0},
    subplot_kw={'projection': ccrs.PlateCarree()}
)

# mean absolute error
cmap = 'Reds'
vmin, vmax = 0, 1.5
levels = np.linspace(0, 1.5, 16)

axs[0,0].set_title(f'AI-model - 7 WR', size=9.5)
axs[0,1].set_title(f'AI-model - 4 WR', size=9.5)
axs[0,2].set_title('SEAS5', size=9.5)
axs[0,0].set_xlim(-20,40)
axs[0,0].set_ylim(30,70)
[axs[i,j].coastlines() for j in range(3) for i in range(3)]

pcm = axs[0,0].contourf(anom.lon, anom.lat, models_7wr_sea_mae, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels)
axs[0,1].contourf(anom.lon, anom.lat, models_4wr_sea_mae, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels)
axs[0,2].contourf(anom.lon, anom.lat, seas5_sea_mae, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels)

cb = fig.colorbar(pcm, ax=axs[0,2], ticks=np.arange(0.2, 1.5, .2), orientation='vertical', label='MAE')
cb.set_label('MAE', labelpad=12)

# anomaly correlation coefficient
cmap = 'RdBu_r'
vmin, vmax = -1, 1
ptresh = .05
levels = np.linspace(-1, 1, 21)

pcm = axs[1,0].contourf(anom.lon, anom.lat, models_7wr_sea_acc.statistic, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels)
axs[1,1].contourf(anom.lon, anom.lat, models_4wr_sea_acc.statistic, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels)
axs[1,2].contourf(anom.lon, anom.lat, seas5_sea_acc.statistic, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels)

h = axs[1,0].contourf(anom.lon, anom.lat, models_7wr_sea_acc.pvalue < ptresh, hatches=['++', None], colors='none', vmin=vmin, vmax=vmax, levels=[1-ptresh,1])
h._hatch_color = (1,1,1,1)
h = axs[1,1].contourf(anom.lon, anom.lat, models_4wr_sea_acc.pvalue < ptresh, hatches=['++', None], colors='none', vmin=vmin, vmax=vmax, levels=[1-ptresh,1])
h._hatch_color = (1,1,1,1)
h = axs[1,2].contourf(anom.lon, anom.lat, seas5_sea_acc.pvalue < ptresh, hatches=['++', None], colors='none', vmin=vmin, vmax=vmax, levels=[1-ptresh,1])
h._hatch_color = (1,1,1,1)

fig.colorbar(pcm, ax=axs[1,2], orientation='vertical', label='ACC')


# coefficient of efficacy
cmap = 'RdBu_r'
vmin, vmax = -1, 1
levels = np.linspace(-1, 1, 21)

pcm = axs[2,0].contourf(anom.lon, anom.lat, models_7wr_sea_ce, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, extend='min')
axs[2,1].contourf(anom.lon, anom.lat, models_4wr_sea_ce, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, extend='min')
axs[2,2].contourf(anom.lon, anom.lat, seas5_sea_ce, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, extend='min')

fig.colorbar(pcm, ax=axs[2,2], orientation='vertical', label='CE', extend='min')


# save
[axs[i,j].set_aspect('auto') for j in range(3) for i in range(3)]
plt.savefig('plots/numWR_vs_SEAS5_%s.pdf' % args.season, bbox_inches='tight')





