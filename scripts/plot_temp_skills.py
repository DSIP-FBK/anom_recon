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
parser.add_argument("--models", type=str, help="path to the folder containing the trained model")
args = parser.parse_args()


# models using 7 wr index as input
torch_model, datamodule, config = get_torch_models_infos(args.models)
anom         = xr.open_dataarray(config['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
idxs         = xr.open_dataarray(config['data']['indexes_paths'][0]).sel(mode=slice(1, config['data']['num_indexes'][0]))
model        = get_models_out(torch_model, idxs, anom, datamodule)

# seasons
winter = (12,1,2)
summer = (6,7,8)
model_winter = model[np.isin(model.time.dt.month, winter)]
anom_winter  = anom[np.isin(anom.time.dt.month, winter)]
model_summer = model[np.isin(model.time.dt.month, summer)]
anom_summer  = anom[np.isin(anom.time.dt.month, summer)]

# reduce all variables to common time-range
start, end    = '2011', '2024'
anom_winter   = anom_winter.sel(time=slice(start, end))
anom_summer   = anom_summer.sel(time=slice(start, end))
model_winter = model_winter.sel(time=slice(start, end))
model_summer = model_summer.sel(time=slice(start, end))

# mean absolute error 
model_winter_mae = abs(anom_winter - model_winter.mean(dim='number')).mean(dim='time')
model_summer_mae = abs(anom_summer - model_summer.mean(dim='number')).mean(dim='time')

# anomaly correlation coefficient
model_winter_acc = pearsonr(anom_winter, model_winter.mean(dim='number'), axis=0)
model_summer_acc = pearsonr(anom_summer, model_summer.mean(dim='number'), axis=0)

# coefficient of efficacy
model_winter_ce = get_ce(anom_winter, model_winter.mean(dim='number'))
model_summer_ce = get_ce(anom_summer, model_summer.mean(dim='number'))

print('       Mean MAE    Mean ACC    Mean CE')
print('Winter   %.2f       %.2f        %.2f' % (model_winter_mae.mean(), model_winter_acc.statistic.mean(), model_winter_ce.mean()))
print('Summer   %.2f       %.2f        %.2f' % (model_summer_mae.mean(), model_summer_acc.statistic.mean(), model_summer_ce.mean()))

# ----------------
# plots
# ----------------
textwidth = 509
fig, axs = plt.subplots(
    2, 3, figsize=set_figsize(textwidth, .5, subplots=(3, 3)), layout="constrained",
    sharex=True, sharey=True, gridspec_kw={'wspace':0, 'hspace':0},
    subplot_kw={'projection': ccrs.PlateCarree()}
)

# mean absolute error
cmap = 'Reds'
vmin, vmax = 0, 1.5
levels = np.linspace(0, 1.5, 16)

axs[0,0].set_xlim(-20,40)
axs[0,0].set_ylim(30,70)
[axs[i,j].coastlines() for j in range(3) for i in range(2)]

pcm = axs[0,0].contourf(anom.lon, anom.lat, model_winter_mae, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels)
axs[1,0].contourf(anom.lon, anom.lat, model_summer_mae, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels)

cb = fig.colorbar(pcm, ax=axs[0,0], ticks=np.arange(0.2, 1.5, .2), location='top', orientation='horizontal', label='MAE')

# anomaly correlation coefficient
cmap = 'RdBu_r'
vmin, vmax = -1, 1
ptresh = .05
levels = np.linspace(-1, 1, 21)

pcm = axs[0,1].contourf(anom.lon, anom.lat, model_winter_acc.statistic, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels)
axs[1,1].contourf(anom.lon, anom.lat, model_summer_acc.statistic, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels)

h = axs[0,1].contourf(anom.lon, anom.lat, model_winter_acc.pvalue > ptresh, hatches=['+++', None], colors='none', vmin=vmin, vmax=vmax, levels=[1-ptresh,1])
h._hatch_color = (0,0,0,1)#(1,1,1,1)
h = axs[1,1].contourf(anom.lon, anom.lat, model_summer_acc.pvalue > ptresh, hatches=['+++', None], colors='none', vmin=vmin, vmax=vmax, levels=[1-ptresh,1])
h._hatch_color = (0,0,0,1)

fig.colorbar(pcm, ax=axs[0,1], location='top', orientation='horizontal', label='ACC')


# coefficient of efficacy
cmap = 'RdBu_r'
vmin, vmax = -1, 1
levels = np.linspace(-1, 1, 21)

pcm = axs[0,2].contourf(anom.lon, anom.lat, model_winter_ce, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels)
axs[1,2].contourf(anom.lon, anom.lat, model_summer_ce, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels)

fig.colorbar(pcm, ax=axs[0,2], location='top', orientation='horizontal', label='CE')


# save
[axs[i,j].set_aspect('auto') for j in range(3) for i in range(2)]
plt.savefig('plots/temp_skills_stability_test.pdf', bbox_inches='tight')
