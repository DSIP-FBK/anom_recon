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

# reduce all variables to common time-range
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

# uncertanties
mae = abs(anom_temp_winter - model_temp_winter).mean(dim='time')
temp_winter_mae_err = (mae.max(dim='number') - mae.min(dim='number')) / model_temp_winter_mae * 100
mae = abs(anom_temp_summer - model_temp_summer).mean(dim='time')
temp_summer_mae_err = (mae.max(dim='number') - mae.min(dim='number')) / model_temp_summer_mae * 100
mae = abs(anom_prec_winter - model_prec_winter).mean(dim='time')
prec_winter_mae_err = (mae.max(dim='number') - mae.min(dim='number')) / model_prec_winter_mae * 100
mae = abs(anom_prec_summer - model_prec_summer).mean(dim='time')
prec_summer_mae_err = (mae.max(dim='number') - mae.min(dim='number')) / model_prec_summer_mae * 100

acc = xr.corr(anom_temp_winter, model_temp_winter, dim='time')
temp_winter_acc_err = (acc.max(dim='number') - acc.min(dim='number')) / model_temp_winter_acc.statistic
acc = xr.corr(anom_temp_summer, model_temp_summer, dim='time')
temp_summer_acc_err = (acc.max(dim='number') - acc.min(dim='number')) / model_temp_summer_acc.statistic
acc = xr.corr(anom_prec_winter, model_prec_winter, dim='time')
prec_winter_acc_err = (acc.max(dim='number') - acc.min(dim='number')) / model_prec_winter_acc.statistic
acc = xr.corr(anom_prec_summer, model_prec_summer, dim='time')
prec_summer_acc_err = (acc.max(dim='number') - acc.min(dim='number')) / model_prec_summer_acc.statistic

ce = get_ce(anom_temp_winter, model_temp_winter)
temp_winter_ce_err = (ce.max(dim='number') - ce.min(dim='number')) / model_temp_winter_ce
ce = get_ce(anom_temp_summer, model_temp_summer)
temp_summer_ce_err = (ce.max(dim='number') - ce.min(dim='number')) / model_temp_summer_ce
ce = get_ce(anom_prec_winter, model_prec_winter)
prec_winter_ce_err = (ce.max(dim='number') - ce.min(dim='number')) / model_prec_winter_ce
ce = get_ce(anom_prec_summer, model_prec_summer)
prec_summer_ce_err = (ce.max(dim='number') - ce.min(dim='number')) / model_prec_summer_ce


print('Temp   Median MAE    Median ACC    Median CE')
print('Winter    %.2f        %.2f         %.2f' % (model_temp_winter_mae.median(), np.median(model_temp_winter_acc.statistic), model_temp_winter_ce.median()))
print('Summer    %.2f        %.2f         %.2f' % (model_temp_summer_mae.median(), np.median(model_temp_summer_acc.statistic), model_temp_summer_ce.median()))

print('Prec   Median MAE    Median ACC    Mean CE')
print('Winter    %.2f        %.2f         %.2f' % (model_prec_winter_mae.median(), np.median(model_prec_winter_acc.statistic), model_prec_winter_ce.median()))
print('Summer    %.2f        %.2f         %.2f' % (model_prec_summer_mae.median(), np.median(model_prec_summer_acc.statistic), model_prec_summer_ce.median()))


# ----------------
# plots
# ----------------
textwidth = 585 #509
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

axs[0,0].text(0.47, 1.2, 'two-meter temperature', fontsize=9.5, transform=axs[0,0].transAxes)
axs[0,3].text(0.7, 1.2, 'precipitation', fontsize=9.5, transform=axs[0,3].transAxes)
axs[0,0].set_title('winter (DJF)', fontsize=8.5)
axs[0,1].set_title('summer (JJA)', fontsize=8.5)
axs[0,3].set_title('winter (DJF)', fontsize=8.5)
axs[0,4].set_title('summer (JJA)', fontsize=8.5)

# mean absolute error
axs[0,0].contourf(anom_temp.lon, anom_temp.lat, model_temp_winter_mae, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, extend='max')
pcm = axs[0,1].contourf(anom_temp.lon, anom_temp.lat, model_temp_summer_mae, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels)
axs[0,3].contourf(anom_prec.lon, anom_prec.lat, model_prec_winter_mae, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels)
axs[0,4].contourf(anom_prec.lon, anom_prec.lat, model_prec_summer_mae, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels)

# uncertanties
err_tresh = 20
h = axs[0,0].contourf(anom_temp.lon, anom_temp.lat, temp_winter_mae_err , hatches=['ooo', None], colors='none', vmin=vmin, vmax=vmax, levels=[err_tresh,100])
h._hatch_color = (0,0,0,1)
h = axs[0,1].contourf(anom_temp.lon, anom_temp.lat, temp_summer_mae_err , hatches=['ooo', None], colors='none', vmin=vmin, vmax=vmax, levels=[err_tresh,100])
h._hatch_color = (0,0,0,1)
h = axs[0,3].contourf(anom_temp.lon, anom_temp.lat, prec_winter_mae_err , hatches=['ooo', None], colors='none', vmin=vmin, vmax=vmax, levels=[err_tresh,100])
h._hatch_color = (0,0,0,1)
h = axs[0,4].contourf(anom_temp.lon, anom_temp.lat, prec_summer_mae_err , hatches=['ooo', None], colors='none', vmin=vmin, vmax=vmax, levels=[err_tresh,100])
h._hatch_color = (0,0,0,1)

cax = inset_axes(axs[0,-1], width="5%", height="100%", loc="upper right", bbox_to_anchor=(0.15, .04, 1, 1), bbox_transform=axs[0,-1].transAxes)
cb = fig.colorbar(pcm, cax=cax, ticks=np.arange(0.2, 1.5, .3), orientation='vertical', extend='max')
cb.set_label('MAE', labelpad=12)

# anomaly correlation coefficient
palette = ['#1b2c62', '#204487', '#2d66af', '#2d66af', '#6bb4e1', '#94d3f3', '#b8e4f8', '#d9f0f9', '#f0f9fd', '#ffffff', '#ffffff', '#fef8de', '#fceda3', '#fdce67', '#fdaa31', '#f8812c', '#ed5729', '#da2f28', '#b81b22', '#921519']
cmap = LinearSegmentedColormap.from_list("", palette)
vmin, vmax = -1, 1
ptresh = .05
levels = np.linspace(-1, 1, 21)

pcm = axs[1,0].contourf(anom_temp.lon, anom_temp.lat, model_temp_winter_acc.statistic, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels)
axs[1,1].contourf(anom_temp.lon, anom_temp.lat, model_temp_summer_acc.statistic, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels)
axs[1,3].contourf(anom_prec.lon, anom_prec.lat, model_prec_winter_acc.statistic, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels)
axs[1,4].contourf(anom_prec.lon, anom_prec.lat, model_prec_summer_acc.statistic, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels)

h = axs[1,0].contourf(anom_temp.lon, anom_temp.lat, model_temp_winter_acc.pvalue > ptresh, hatches=['+++', None], colors='none', vmin=vmin, vmax=vmax, levels=[1-ptresh,1])
h._hatch_color = (0,0,0,1)
h = axs[1,1].contourf(anom_temp.lon, anom_temp.lat, model_temp_summer_acc.pvalue > ptresh, hatches=['+++', None], colors='none', vmin=vmin, vmax=vmax, levels=[1-ptresh,1])
h._hatch_color = (0,0,0,1)
h = axs[1,3].contourf(anom_prec.lon, anom_prec.lat, model_prec_winter_acc.pvalue > ptresh, hatches=['+++', None], colors='none', vmin=vmin, vmax=vmax, levels=[1-ptresh,1])
h._hatch_color = (0,0,0,1)
h = axs[1,4].contourf(anom_prec.lon, anom_prec.lat, model_prec_summer_acc.pvalue > ptresh, hatches=['+++', None], colors='none', vmin=vmin, vmax=vmax, levels=[1-ptresh,1])
h._hatch_color = (0,0,0,1)

axs[1,0].contour(anom_temp.lon, anom_temp.lat, model_temp_winter_acc.statistic, colors='k', linewidths=.5, levels=[.5])
axs[1,1].contour(anom_temp.lon, anom_temp.lat, model_temp_summer_acc.statistic, colors='k', linewidths=.5, levels=[.5])
axs[1,3].contour(anom_prec.lon, anom_prec.lat, model_prec_winter_acc.statistic, colors='k', linewidths=.5, levels=[.5])
axs[1,4].contour(anom_prec.lon, anom_prec.lat, model_prec_summer_acc.statistic, colors='k', linewidths=.5, levels=[.5])

# uncertanties
err_tresh = 20
h = axs[1,0].contourf(anom_temp.lon, anom_temp.lat, temp_winter_acc_err , hatches=['ooo', None], colors='none', vmin=vmin, vmax=vmax, levels=[err_tresh,100])
h._hatch_color = (0,0,0,1)
h = axs[1,1].contourf(anom_temp.lon, anom_temp.lat, temp_summer_acc_err , hatches=['ooo', None], colors='none', vmin=vmin, vmax=vmax, levels=[err_tresh,100])
h._hatch_color = (0,0,0,1)
h = axs[1,3].contourf(anom_temp.lon, anom_temp.lat, prec_winter_acc_err , hatches=['ooo', None], colors='none', vmin=vmin, vmax=vmax, levels=[err_tresh,100])
h._hatch_color = (0,0,0,1)
h = axs[1,4].contourf(anom_temp.lon, anom_temp.lat, prec_summer_acc_err , hatches=['ooo', None], colors='none', vmin=vmin, vmax=vmax, levels=[err_tresh,100])
h._hatch_color = (0,0,0,1)

cax = inset_axes(axs[1,-1], width="5%", height="100%", loc="upper right", bbox_to_anchor=(0.15, .04, 1, 1), bbox_transform=axs[1,-1].transAxes)
cb = fig.colorbar(pcm, cax=cax, orientation='vertical', label='ACC')

# coefficient of efficacy
palette = ['#1b2c62', '#204487', '#2d66af', '#2d66af', '#6bb4e1', '#94d3f3', '#b8e4f8', '#d9f0f9', '#f0f9fd', '#ffffff', '#ffffff', '#fef8de', '#fceda3', '#fdce67', '#fdaa31', '#f8812c', '#ed5729', '#da2f28', '#b81b22', '#921519']
cmap = LinearSegmentedColormap.from_list("", palette)
vmin, vmax = -1, 1
levels = np.linspace(-1, 1, 21)

pcm = axs[2,0].contourf(anom_temp.lon, anom_temp.lat, model_temp_winter_ce, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels)
axs[2,1].contourf(anom_temp.lon, anom_temp.lat, model_temp_summer_ce, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, extend='min')
axs[2,3].contourf(anom_prec.lon, anom_prec.lat, model_prec_winter_ce, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, extend='min')
axs[2,4].contourf(anom_prec.lon, anom_prec.lat, model_prec_summer_ce, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, extend='min')

axs[2,0].contour(anom_temp.lon, anom_temp.lat, model_temp_winter_ce, ls='--', colors='k', linewidths=.5, levels=[.3])
axs[2,1].contour(anom_temp.lon, anom_temp.lat, model_temp_summer_ce, ls='--', colors='k', linewidths=.5, levels=[.3])
axs[2,3].contour(anom_prec.lon, anom_prec.lat, model_prec_winter_ce, ls='--', colors='k', linewidths=.5, levels=[.3])
axs[2,4].contour(anom_prec.lon, anom_prec.lat, model_prec_summer_ce, ls='--', colors='k', linewidths=.5, levels=[.3])

# uncertanties
err_tresh = 20
h = axs[2,0].contourf(anom_temp.lon, anom_temp.lat, temp_winter_ce_err , hatches=['ooo', None], colors='none', vmin=vmin, vmax=vmax, levels=[-100, -err_tresh, err_tresh,100])
h._hatch_color = (0,0,0,1)
h = axs[2,1].contourf(anom_temp.lon, anom_temp.lat, temp_summer_ce_err , hatches=['ooo', None], colors='none', vmin=vmin, vmax=vmax, levels=[-100, -err_tresh, err_tresh,100])
h._hatch_color = (0,0,0,1)
h = axs[2,3].contourf(anom_temp.lon, anom_temp.lat, prec_winter_ce_err , hatches=['ooo', None], colors='none', vmin=vmin, vmax=vmax, levels=[-100, -err_tresh, err_tresh,100])
h._hatch_color = (0,0,0,1)
h = axs[2,4].contourf(anom_temp.lon, anom_temp.lat, prec_summer_ce_err , hatches=['ooo', None], colors='none', vmin=vmin, vmax=vmax, levels=[-100, -err_tresh, err_tresh,100])
h._hatch_color = (0,0,0,1)

cax = inset_axes(axs[2,-1], width="5%", height="100%", loc="upper right", bbox_to_anchor=(0.15, .04, 1, 1), bbox_transform=axs[2,-1].transAxes)
fig.colorbar(pcm, cax=cax, orientation='vertical', label='CE')

# save
[axs[i,j].set_aspect('auto') for j in range(5) for i in range(3)]
plt.savefig('plots/temp_prec_skills_7wr.pdf', bbox_inches='tight')
