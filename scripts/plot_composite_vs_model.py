import argparse, sys
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

from functions import get_composite_recon, get_ce, get_models_out, get_torch_models_infos, set_figsize

parser = argparse.ArgumentParser()
parser.add_argument("--indices", nargs="+", help="path to the NetCDF containing the monthly WR indices")
parser.add_argument("--temp_anom", help="path to the NetCDF containing the monthly temperature anomalies")
parser.add_argument("--prec_anom", help="path to the NetCDF containing the monthly precipitation anomalies")
parser.add_argument("--temp_model", nargs="+", type=str, help="path to the folder containing the trained model for temperature")
parser.add_argument("--prec_model", nargs="+", type=str, help="path to the folder containing the trained model for precipitation")
parser.add_argument("--clim_start", help="(str) start date of the climatology period (e.g. 1981-01-01)")
parser.add_argument("--clim_end", help="(str) end date of the climatology period (e.g. 2010-12-31)")
parser.add_argument("--start", help="(str) start of the comparison (e.g. 2005-01-01)")
args = parser.parse_args()

# load data
monthly_Iwr = xr.concat([xr.open_dataarray(path) for path in args.indices], dim="time").sortby('time')
monthly_temp = xr.open_dataarray(args.temp_anom).rename({'latitude': 'lat', 'longitude': 'lon'})
monthly_prec = xr.open_dataarray(args.prec_anom).rename({'latitude': 'lat', 'longitude': 'lon'})
n_clusters = len(monthly_Iwr.mode)

# get composite reconstruction
composite_temp_recon_winter = get_composite_recon(monthly_temp, monthly_Iwr, args.clim_start, args.clim_end, months=(1,2,12))
composite_temp_recon_summer = get_composite_recon(monthly_temp, monthly_Iwr, args.clim_start, args.clim_end, months=(6,7,8))
composite_prec_recon_winter = get_composite_recon(monthly_prec, monthly_Iwr, args.clim_start, args.clim_end, months=(1,2,12))
composite_prec_recon_summer = get_composite_recon(monthly_prec, monthly_Iwr, args.clim_start, args.clim_end, months=(6,7,8))

# get ai-model reconstruction
model_temp_recon = []
for model in args.temp_model:
    torch_model, datamodule, config = get_torch_models_infos(model)
    idxs_temp  = xr.open_dataarray(config['data']['indexes_paths'][0]).sel(mode=slice(1, config['data']['num_indexes'][0]))
    model_temp_recon.append(get_models_out(torch_model, idxs_temp, monthly_temp, datamodule))
model_temp_recon = xr.concat(model_temp_recon, dim='time').sortby('time')

model_prec_recon = []
for model in args.prec_model:
    torch_model, datamodule, config = get_torch_models_infos(model)
    idxs_prec  = xr.open_dataarray(config['data']['indexes_paths'][0]).sel(mode=slice(1, config['data']['num_indexes'][0]))
    model_prec_recon.append(get_models_out(torch_model, idxs_prec, monthly_prec, datamodule))
model_prec_recon = xr.concat(model_prec_recon, dim='time').sortby('time')

# reduce all variables to common time-range
start, end           = args.start, '2024'
monthly_temp         = monthly_temp.sel(time=slice(start, end))
monthly_prec         = monthly_prec.sel(time=slice(start, end))
model_temp_recon     = model_temp_recon.sel(time=slice(start, end))
model_prec_recon     = model_prec_recon.sel(time=slice(start, end))
composite_temp_recon_winter = composite_temp_recon_winter.sel(time=slice(start, end))
composite_temp_recon_summer = composite_temp_recon_summer.sel(time=slice(start, end))
composite_prec_recon_winter = composite_prec_recon_winter.sel(time=slice(start, end))
composite_prec_recon_summer = composite_prec_recon_summer.sel(time=slice(start, end))

# select seasons
winter = (12,1,2)
summer = (6,7,8)
monthly_temp_winter = monthly_temp.sel(time=monthly_temp['time.month'].isin(winter))
monthly_temp_summer = monthly_temp.sel(time=monthly_temp['time.month'].isin(summer))
monthly_prec_winter = monthly_prec.sel(time=monthly_prec['time.month'].isin(winter))
monthly_prec_summer = monthly_prec.sel(time=monthly_prec['time.month'].isin(summer))
model_prec_recon_winter = model_prec_recon.sel(time=model_prec_recon['time.month'].isin(winter))
model_prec_recon_summer = model_prec_recon.sel(time=model_prec_recon['time.month'].isin(summer))
model_temp_recon_winter = model_temp_recon.sel(time=model_temp_recon['time.month'].isin(winter))
model_temp_recon_summer = model_temp_recon.sel(time=model_temp_recon['time.month'].isin(summer))


# -----------------------------------------
# Compute Skills
# -----------------------------------------
# mean absolute error 
model_temp_mae_winter = abs(monthly_temp_winter - model_temp_recon_winter.mean(dim='number')).mean(dim='time')
model_temp_mae_summer = abs(monthly_temp_summer - model_temp_recon_summer.mean(dim='number')).mean(dim='time')
model_prec_mae_winter = abs(monthly_prec_winter - model_prec_recon_winter.mean(dim='number')).mean(dim='time')
model_prec_mae_summer = abs(monthly_prec_summer - model_prec_recon_summer.mean(dim='number')).mean(dim='time')

composite_temp_mae_winter = abs(monthly_temp_winter - composite_temp_recon_winter).mean(dim='time')
composite_temp_mae_summer = abs(monthly_temp_summer - composite_temp_recon_summer).mean(dim='time')
composite_prec_mae_winter = abs(monthly_prec_winter - composite_prec_recon_winter).mean(dim='time')
composite_prec_mae_summer = abs(monthly_prec_summer - composite_prec_recon_summer).mean(dim='time')

# anomaly correlation coefficient
model_temp_acc_winter = xr.corr(monthly_temp_winter, model_temp_recon_winter.mean(dim='number'), dim='time')
model_temp_acc_summer = xr.corr(monthly_temp_summer, model_temp_recon_summer.mean(dim='number'), dim='time')
model_prec_acc_winter = xr.corr(monthly_prec_winter, model_prec_recon_winter.mean(dim='number'), dim='time')
model_prec_acc_summer = xr.corr(monthly_prec_summer, model_prec_recon_summer.mean(dim='number'), dim='time')

composite_temp_acc_winter = xr.corr(monthly_temp_winter, composite_temp_recon_winter, dim='time')
composite_temp_acc_summer = xr.corr(monthly_temp_summer, composite_temp_recon_summer, dim='time')
composite_prec_acc_winter = xr.corr(monthly_prec_winter, composite_prec_recon_winter, dim='time')
composite_prec_acc_summer = xr.corr(monthly_prec_summer, composite_prec_recon_summer, dim='time')

# coefficient of efficacy
model_temp_ce_winter = get_ce(monthly_temp_winter, model_temp_recon_winter.mean(dim='number'))
model_temp_ce_summer = get_ce(monthly_temp_summer, model_temp_recon_summer.mean(dim='number'))
model_prec_ce_winter = get_ce(monthly_prec_winter, model_prec_recon_winter.mean(dim='number'))
model_prec_ce_summer = get_ce(monthly_prec_summer, model_prec_recon_summer.mean(dim='number'))

composite_temp_ce_winter = get_ce(monthly_temp_winter, composite_temp_recon_winter)
composite_temp_ce_summer = get_ce(monthly_temp_summer, composite_temp_recon_summer)
composite_prec_ce_winter = get_ce(monthly_prec_winter, composite_prec_recon_winter)
composite_prec_ce_summer = get_ce(monthly_prec_summer, composite_prec_recon_summer)


# -----------------------------------------
# Compute Skill Scores
# -----------------------------------------
temp_mae_winter_score = -(model_temp_mae_winter - composite_temp_mae_winter) / composite_temp_mae_winter * 100
temp_mae_summer_score = -(model_temp_mae_summer - composite_temp_mae_summer) / composite_temp_mae_summer * 100
prec_mae_winter_score = -(model_prec_mae_winter - composite_prec_mae_winter) / composite_prec_mae_winter * 100
prec_mae_summer_score = -(model_prec_mae_summer - composite_prec_mae_summer) / composite_prec_mae_summer * 100

temp_acc_winter_score = (model_temp_acc_winter - composite_temp_acc_winter) / (1 - composite_temp_acc_winter) * 100
temp_acc_summer_score = (model_temp_acc_summer - composite_temp_acc_summer) / (1 - composite_temp_acc_summer) * 100
prec_acc_winter_score = (model_prec_acc_winter - composite_prec_acc_winter) / (1 - composite_prec_acc_winter) * 100
prec_acc_summer_score = (model_prec_acc_summer - composite_prec_acc_summer) / (1 - composite_prec_acc_summer) * 100

temp_ce_winter_score = (model_temp_ce_winter - composite_temp_ce_winter) / (1 - composite_temp_ce_winter) * 100
temp_ce_summer_score = (model_temp_ce_summer - composite_temp_ce_summer) / (1 - composite_temp_ce_summer) * 100
prec_ce_winter_score = (model_prec_ce_winter - composite_prec_ce_winter) / (1 - composite_prec_ce_winter) * 100
prec_ce_summer_score = (model_prec_ce_summer - composite_prec_ce_summer) / (1 - composite_prec_ce_summer) * 100


# -----------------------------------------
# Plot difference
# -----------------------------------------
textwidth = 460  # IJC
rasterized=True
palette = ['#1b2c62', '#204487', '#2d66af', '#2d66af', '#6bb4e1', '#94d3f3', '#b8e4f8', '#d9f0f9', '#f0f9fd', '#ffffff', '#ffffff', '#fef8de', '#fceda3', '#fdce67', '#fdaa31', '#f8812c', '#ed5729', '#da2f28', '#b81b22', '#921519']
cmap = LinearSegmentedColormap.from_list("", palette)
vmin, vmax = -100, 100
levels = np.linspace(-100, 100, 21)
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
axs[0,0].set_title('Winter', fontsize=8)
axs[0,1].set_title('Summer', fontsize=8)
axs[0,3].set_title('Winter', fontsize=8)
axs[0,4].set_title('Summer', fontsize=8)
axs[0,0].text(-.1, .5, 'MAE', va='center', ha='center', rotation=90, fontsize=8, transform=axs[0,0].transAxes)
axs[1,0].text(-.1, .5, 'ACC', va='center', ha='center', rotation=90, fontsize=8, transform=axs[1,0].transAxes)
axs[2,0].text(-.1, .5, 'CE', va='center', ha='center', rotation=90, fontsize=8, transform=axs[2,0].transAxes)

# mean absolute error
pcm = axs[0,0].contourf(monthly_temp.lon, monthly_temp.lat, temp_mae_winter_score, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, rasterized=rasterized)
axs[0,1].contourf(monthly_temp.lon, monthly_temp.lat, temp_mae_summer_score, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, rasterized=rasterized)
axs[0,3].contourf(monthly_prec.lon, monthly_prec.lat, prec_mae_winter_score, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, rasterized=rasterized)
axs[0,4].contourf(monthly_prec.lon, monthly_prec.lat, prec_mae_summer_score, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, rasterized=rasterized)

# anomaly correlation coefficient
axs[1,0].contourf(monthly_temp.lon, monthly_temp.lat, temp_acc_winter_score, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, rasterized=rasterized)
axs[1,1].contourf(monthly_temp.lon, monthly_temp.lat, temp_acc_summer_score, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, rasterized=rasterized)
axs[1,3].contourf(monthly_prec.lon, monthly_prec.lat, prec_acc_winter_score, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, rasterized=rasterized)
axs[1,4].contourf(monthly_prec.lon, monthly_prec.lat, prec_acc_summer_score, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, rasterized=rasterized)

axs[1,0].contour(monthly_temp.lon, monthly_temp.lat, temp_acc_winter_score, colors='k', linewidths=.5, levels=[30], rasterized=rasterized)
axs[1,1].contour(monthly_temp.lon, monthly_temp.lat, temp_acc_summer_score, colors='k', linewidths=.5, levels=[30], rasterized=rasterized)
axs[1,3].contour(monthly_prec.lon, monthly_prec.lat, prec_acc_winter_score, colors='k', linewidths=.5, levels=[30], rasterized=rasterized)
axs[1,4].contour(monthly_prec.lon, monthly_prec.lat, prec_acc_summer_score, colors='k', linewidths=.5, levels=[30], rasterized=rasterized)

axs[1,0].contour(monthly_temp.lon, monthly_temp.lat, temp_acc_winter_score, colors='k', ls='--', linewidths=.5, levels=[-30], rasterized=rasterized)
axs[1,1].contour(monthly_temp.lon, monthly_temp.lat, temp_acc_summer_score, colors='k', ls='--', linewidths=.5, levels=[-30], rasterized=rasterized)
axs[1,3].contour(monthly_prec.lon, monthly_prec.lat, prec_acc_winter_score, colors='k', ls='--', linewidths=.5, levels=[-30], rasterized=rasterized)
axs[1,4].contour(monthly_prec.lon, monthly_prec.lat, prec_acc_summer_score, colors='k', ls='--', linewidths=.5, levels=[-30], rasterized=rasterized)

# coefficient of efficacy
axs[2,0].contourf(monthly_temp.lon, monthly_temp.lat, temp_ce_winter_score, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, rasterized=rasterized)
axs[2,1].contourf(monthly_temp.lon, monthly_temp.lat, temp_ce_summer_score, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, rasterized=rasterized)
axs[2,3].contourf(monthly_prec.lon, monthly_prec.lat, prec_ce_winter_score, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, rasterized=rasterized)
axs[2,4].contourf(monthly_prec.lon, monthly_prec.lat, prec_ce_summer_score, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, rasterized=rasterized)

axs[2,0].contour(monthly_temp.lon, monthly_temp.lat, temp_ce_winter_score, ls='--', colors='k', linewidths=.5, levels=[30], rasterized=rasterized)
axs[2,1].contour(monthly_temp.lon, monthly_temp.lat, temp_ce_summer_score, ls='--', colors='k', linewidths=.5, levels=[30], rasterized=rasterized)
axs[2,3].contour(monthly_prec.lon, monthly_prec.lat, prec_ce_winter_score, ls='--', colors='k', linewidths=.5, levels=[30], rasterized=rasterized)
axs[2,4].contour(monthly_prec.lon, monthly_prec.lat, prec_ce_summer_score, ls='--', colors='k', linewidths=.5, levels=[30], rasterized=rasterized)

axs[2,0].contour(monthly_temp.lon, monthly_temp.lat, temp_ce_winter_score, colors='k', ls='--', linewidths=.5, levels=[-30], rasterized=rasterized)
axs[2,1].contour(monthly_temp.lon, monthly_temp.lat, temp_ce_summer_score, colors='k', ls='--', linewidths=.5, levels=[-30], rasterized=rasterized)
axs[2,3].contour(monthly_prec.lon, monthly_prec.lat, prec_ce_winter_score, colors='k', ls='--', linewidths=.5, levels=[-30], rasterized=rasterized)
axs[2,4].contour(monthly_prec.lon, monthly_prec.lat, prec_ce_summer_score, colors='k', ls='--', linewidths=.5, levels=[-30], rasterized=rasterized)

cax = inset_axes(axs[-1,0], width="400%", height="7%", loc="lower left", bbox_to_anchor=(0, -.2, 1, 1), bbox_transform=axs[-1,0].transAxes)
cb = fig.colorbar(pcm, cax=cax, orientation='horizontal')
cb.set_label('Relative Improvement (%)', labelpad=5)

# save
[axs[i,j].set_aspect('auto') for j in range(5) for i in range(3)]
plt.savefig(f'plots/composite_vs_model_{n_clusters}wr_{start}-{end}.png', bbox_inches='tight', dpi=300)