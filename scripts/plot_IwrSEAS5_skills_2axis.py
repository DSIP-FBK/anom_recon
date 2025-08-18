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
args = parser.parse_args()

# seasons
winter_months = (12,1,2)
summer_months = (6,7,8)


# ---------------------
# Load models and SEAS5
# ---------------------
print('Loading models and SEAS5...')

# ERA5 anomalies
anomT      = xr.open_dataarray(args.anom_temp).rename({'latitude': 'lat', 'longitude': 'lon'})
anomP      = xr.open_dataarray(args.anom_prec).rename({'latitude': 'lat', 'longitude': 'lon'})

# SEAS5 winter and summer for temperature and precipitation
seas5T      = xr.open_dataarray(args.seas5_temp).rename({'latitude': 'lat', 'longitude': 'lon'})#.transpose('time', 'lat', 'lon', 'number', 'forecastMonth')
seas5T_DJF  = get_SEAS5_season(seas5T, 'winter')
seas5T_JJA  = get_SEAS5_season(seas5T, 'summer')

seas5P      = xr.open_dataarray(args.seas5_prec).rename({'latitude': 'lat', 'longitude': 'lon'})#.transpose('time', 'lat', 'lon', 'number', 'forecastMonth')
seas5P_DJF  = get_SEAS5_season(seas5P, 'winter')
seas5P_JJA  = get_SEAS5_season(seas5P, 'summer')

# models.rename({'latitude': 'lat', 'longitude': 'lon'})
model_seas5T = xr.open_dataarray(args.model_seas5_temp)
model_seas5P = xr.open_dataarray(args.model_seas5_prec)

# seasons
anomT_DJF            = anomT[np.isin(anomT.time.dt.month, winter_months)]
anomT_JJA            = anomT[np.isin(anomT.time.dt.month, summer_months)]
model_seas5_7wrT_DJF = get_SEAS5_season(model_seas5T, 'winter')
model_seas5_7wrT_JJA = get_SEAS5_season(model_seas5T, 'summer')

anomP_DJF            = anomP[np.isin(anomP.time.dt.month, winter_months)]
anomP_JJA            = anomP[np.isin(anomP.time.dt.month, summer_months)]
model_seas5_7wrP_DJF = get_SEAS5_season(model_seas5P, 'winter')
model_seas5_7wrP_JJA = get_SEAS5_season(model_seas5P, 'summer')

# reduce all variables to common time-range
start, end           = '2011', '2024'
anomT_DJF            = anomT_DJF.sel(time=slice(start, end))
anomT_JJA            = anomT_JJA.sel(time=slice(start, end))
seas5T_DJF           = seas5T_DJF.sel(time=slice(start, end))
seas5T_JJA           = seas5T_JJA.sel(time=slice(start, end))
model_seas5_7wrT_DJF = model_seas5_7wrT_DJF.sel(time=slice(start, end))
model_seas5_7wrT_JJA = model_seas5_7wrT_JJA.sel(time=slice(start, end))
anomP_DJF            = anomP_DJF.sel(time=slice(start, end))
anomP_JJA            = anomP_JJA.sel(time=slice(start, end))
seas5P_DJF           = seas5P_DJF.sel(time=slice(start, end))
seas5P_JJA           = seas5P_JJA.sel(time=slice(start, end))
model_seas5_7wrP_DJF = model_seas5_7wrP_DJF.sel(time=slice(start, end))
model_seas5_7wrP_JJA = model_seas5_7wrP_JJA.sel(time=slice(start, end))

# ensemble and models mean
seas5T_DJF = seas5T_DJF.mean(dim='number')
seas5T_JJA = seas5T_JJA.mean(dim='number')
seas5P_DJF = seas5P_DJF.mean(dim='number')
seas5P_JJA = seas5P_JJA.mean(dim='number')
model_seas5_7wrT_DJF = model_seas5_7wrT_DJF.mean(dim=['number', 'ensemble_member'])
model_seas5_7wrT_JJA = model_seas5_7wrT_JJA.mean(dim=['number', 'ensemble_member'])
model_seas5_7wrP_DJF = model_seas5_7wrP_DJF.mean(dim=['number', 'ensemble_member'])
model_seas5_7wrP_JJA = model_seas5_7wrP_JJA.mean(dim=['number', 'ensemble_member'])

# --------------
# Compute skills
# --------------
print('Computing skills...')

# model mean absolute error
model_seas5_7wrT_DJF_MAE = abs(anomT_DJF - model_seas5_7wrT_DJF).mean(dim='time')
model_seas5_7wrT_JJA_MAE = abs(anomT_JJA - model_seas5_7wrT_JJA).mean(dim='time')
model_seas5_7wrP_DJF_MAE = abs(anomP_DJF - model_seas5_7wrP_DJF).mean(dim='time')
model_seas5_7wrP_JJA_MAE = abs(anomP_JJA - model_seas5_7wrP_JJA).mean(dim='time')

# SEAS5 mean absolute error
seas5T_DJF_MAE = abs(anomT_DJF - seas5T_DJF).mean(dim='time')
seas5T_JJA_MAE = abs(anomT_JJA - seas5T_JJA).mean(dim='time')
seas5P_DJF_MAE = abs(anomP_DJF - seas5P_DJF).mean(dim='time')
seas5P_JJA_MAE = abs(anomP_JJA - seas5P_JJA).mean(dim='time')

# model anomaly correlation coefficient
model_seas5_7wrT_DJF_ACC = xr.corr(anomT_DJF, model_seas5_7wrT_DJF, dim='time')
model_seas5_7wrT_JJA_ACC = xr.corr(anomT_JJA, model_seas5_7wrT_JJA, dim='time')
model_seas5_7wrP_DJF_ACC = xr.corr(anomP_DJF, model_seas5_7wrP_DJF, dim='time')
model_seas5_7wrP_JJA_ACC = xr.corr(anomP_JJA, model_seas5_7wrP_JJA, dim='time')

# SEAS5 anomaly correlation coefficient
seas5T_DJF_ACC = xr.corr(anomT_DJF, seas5T_DJF, dim='time')
seas5T_JJA_ACC = xr.corr(anomT_JJA, seas5T_JJA, dim='time')
seas5P_DJF_ACC = xr.corr(anomP_DJF, seas5P_DJF, dim='time')
seas5P_JJA_ACC = xr.corr(anomP_JJA, seas5P_JJA, dim='time')

# model coefficient of efficacy
model_seas5_7wrT_DJF_CE = get_ce(anomT_DJF, model_seas5_7wrT_DJF)
model_seas5_7wrT_JJA_CE = get_ce(anomT_JJA, model_seas5_7wrT_JJA)
model_seas5_7wrP_DJF_CE = get_ce(anomP_DJF, model_seas5_7wrP_DJF)
model_seas5_7wrP_JJA_CE = get_ce(anomP_JJA, model_seas5_7wrP_JJA)

# SEAS coefficient of efficacy
seas5T_DJF_CE = get_ce(anomT_DJF, seas5T_DJF)
seas5T_JJA_CE = get_ce(anomT_JJA, seas5T_JJA)
seas5P_DJF_CE = get_ce(anomP_DJF, seas5P_DJF)
seas5P_JJA_CE = get_ce(anomP_JJA, seas5P_JJA)

# mask outside land
lsm = xr.open_dataarray('../data/lsm_regrid_shift_europe.nc').rename({'latitude': 'lat', 'longitude': 'lon'})
model_seas5_7wrT_DJF_MAE = model_seas5_7wrT_DJF_MAE.where(lsm > .8)
model_seas5_7wrT_JJA_MAE = model_seas5_7wrT_JJA_MAE.where(lsm > .8)
model_seas5_7wrP_DJF_MAE = model_seas5_7wrP_DJF_MAE.where(lsm > .8)
model_seas5_7wrP_JJA_MAE = model_seas5_7wrP_JJA_MAE.where(lsm > .8)
seas5T_DJF_MAE = seas5T_DJF_MAE.where(lsm > .8)
seas5T_JJA_MAE = seas5T_JJA_MAE.where(lsm > .8)
seas5P_DJF_MAE = seas5P_DJF_MAE.where(lsm > .8)
seas5P_JJA_MAE = seas5P_JJA_MAE.where(lsm > .8)
model_seas5_7wrT_DJF_ACC = model_seas5_7wrT_DJF_ACC.where(lsm > .8)
model_seas5_7wrT_JJA_ACC = model_seas5_7wrT_JJA_ACC.where(lsm > .8)
model_seas5_7wrP_DJF_ACC = model_seas5_7wrP_DJF_ACC.where(lsm > .8)
model_seas5_7wrP_JJA_ACC = model_seas5_7wrP_JJA_ACC.where(lsm > .8)
seas5T_DJF_ACC = seas5T_DJF_ACC.where(lsm > .8)
seas5T_JJA_ACC = seas5T_JJA_ACC.where(lsm > .8)
seas5P_DJF_ACC = seas5P_DJF_ACC.where(lsm > .8)
seas5P_JJA_ACC = seas5P_JJA_ACC.where(lsm > .8)
model_seas5_7wrT_DJF_CE = model_seas5_7wrT_DJF_CE.where(lsm > .8)
model_seas5_7wrT_JJA_CE = model_seas5_7wrT_JJA_CE.where(lsm > .8)
model_seas5_7wrP_DJF_CE = model_seas5_7wrP_DJF_CE.where(lsm > .8)
model_seas5_7wrP_JJA_CE = model_seas5_7wrP_JJA_CE.where(lsm > .8)
seas5T_DJF_CE = seas5T_DJF_CE.where(lsm > .8)
seas5T_JJA_CE = seas5T_JJA_CE.where(lsm > .8)
seas5P_DJF_CE = seas5P_DJF_CE.where(lsm > .8)
seas5P_JJA_CE = seas5P_JJA_CE.where(lsm > .8)

# mask CE values below certain threshold (not relevant)
# threshold = -1 # min(model_seas5_7wrT_DJF_CE.min(), model_seas5_7wrT_JJA_CE.min(), model_seas5_7wrP_DJF_CE.min(), model_seas5_7wrP_JJA_CE.min())
# model_seas5_7wrT_DJF_CE = xr.where(model_seas5_7wrT_DJF_CE < threshold, threshold, model_seas5_7wrT_DJF_CE)
# model_seas5_7wrT_JJA_CE = xr.where(model_seas5_7wrT_JJA_CE < threshold, threshold, model_seas5_7wrT_JJA_CE)
# model_seas5_7wrP_DJF_CE = xr.where(model_seas5_7wrP_DJF_CE < threshold, threshold, model_seas5_7wrP_DJF_CE)
# model_seas5_7wrP_JJA_CE = xr.where(model_seas5_7wrP_JJA_CE < threshold, threshold, model_seas5_7wrP_JJA_CE)
# seas5T_DJF_CE = xr.where(seas5T_DJF_CE < threshold, threshold, seas5T_DJF_CE)
# seas5T_JJA_CE = xr.where(seas5T_JJA_CE < threshold, threshold, seas5T_JJA_CE)
# seas5P_DJF_CE = xr.where(seas5P_DJF_CE < threshold, threshold, seas5P_DJF_CE)
# seas5P_JJA_CE = xr.where(seas5P_JJA_CE < threshold, threshold, seas5P_JJA_CE)


# -----------
# Plot skills
# -----------
print('Plotting skills...')

columnwidth = 236 #248.
seas5_color = '#b81b22' #'indianred'
model_color = '#204487' #'cornflowerblue'
fig, axs_sx = plt.subplots(
    2, 2, figsize=set_figsize(columnwidth, 1, subplots=(2,2)), layout="tight",
    sharex='row', sharey=True, gridspec_kw={'wspace': 0, 'hspace': 0.7}
)
axs_dx = np.array([[axs_sx[0,0].twinx(), axs_sx[0,1].twinx()], [axs_sx[1,0].twinx(), axs_sx[1,1].twinx()]])
axs_sx[0,0].set_ylabel('MAE', color='seagreen')
axs_sx[1,0].set_ylabel('MAE', color='seagreen')
axs_dx[0,1].set_ylabel('ACC and CE', color='rebeccapurple')
axs_dx[1,1].set_ylabel('ACC and CE', color='rebeccapurple')
axs_sx[0,0].text(0.45, 1.3, 'two-meter temperature', fontsize=9.5, transform=axs_sx[0,0].transAxes)
axs_sx[0,0].set_title('winter (DJF)', fontsize=8.5)
axs_sx[0,1].set_title('summer (JJA)', fontsize=8.5)
axs_sx[1,0].text(0.7, 1.3, 'precipitation', fontsize=9.5, transform=axs_sx[1,0].transAxes)
axs_sx[1,0].set_title('winter (DJF)', fontsize=8.5)
axs_sx[1,1].set_title('summer (JJA)', fontsize=8.5)
axs_dx[0,0].set_xlim(-1.3, 5.3)
axs_dx[1,0].set_xlim(-1.3, 5.3)
[axs_sx[i,j].set_ylim(0, 1.5) for i in range(2) for j in range(2)]
[axs_dx[i,j].set_ylim(-1, 1) for i in range(2) for j in range(2)]

# ----------------
# Plot temperature
# ----------------
# SEAS5 winter
plot_boxplot(axs_sx[0,0], seas5T_DJF_MAE.data.flatten(), [-0.3,], seas5_color, '*')
plot_boxplot(axs_dx[0,0], seas5T_DJF_ACC.data.flatten(), [1.7,], seas5_color, '*')
plot_boxplot(axs_dx[0,0], seas5T_DJF_CE.data.flatten(), [3.7,], seas5_color, '*')

# model winter
plot_boxplot(axs_sx[0,0], model_seas5_7wrT_DJF_MAE.data.flatten(), [0.3,], model_color, '*')
plot_boxplot(axs_dx[0,0], model_seas5_7wrT_DJF_ACC.data.flatten(), [2.3,], model_color, '*')
plot_boxplot(axs_dx[0,0], model_seas5_7wrT_DJF_CE.data.flatten(), [4.3,], model_color, '*')

# seas summer
plot_boxplot(axs_sx[0,1], seas5T_JJA_MAE.data.flatten(), [-0.3,], seas5_color, '*')
plot_boxplot(axs_dx[0,1], seas5T_JJA_ACC.data.flatten(), [1.7,], seas5_color, '*')
plot_boxplot(axs_dx[0,1], seas5T_JJA_CE.data.flatten(), [3.7,], seas5_color, '*')

# model summer
plot_boxplot(axs_sx[0,1], model_seas5_7wrT_JJA_MAE.data.flatten(), [0.3,], model_color, '*')
plot_boxplot(axs_dx[0,1], model_seas5_7wrT_JJA_ACC.data.flatten(), [2.3,], model_color, '*')
plot_boxplot(axs_dx[0,1], model_seas5_7wrT_JJA_CE.data.flatten(), [4.3,], model_color, '*')

# ------------------
# Plot precipitation
# ------------------
# SEAS5 winter
plot_boxplot(axs_sx[1,0], seas5P_DJF_MAE.data.flatten(), [-0.3,], seas5_color, '*')
plot_boxplot(axs_dx[1,0], seas5P_DJF_ACC.data.flatten(), [1.7,], seas5_color, '*')
plot_boxplot(axs_dx[1,0], seas5P_DJF_CE.data.flatten(), [3.7,], seas5_color, '*')

# model winter
plot_boxplot(axs_sx[1,0], model_seas5_7wrP_DJF_MAE.data.flatten(), [0.3,], model_color, '*')
plot_boxplot(axs_dx[1,0], model_seas5_7wrP_DJF_ACC.data.flatten(), [2.3,], model_color, '*')
plot_boxplot(axs_dx[1,0], model_seas5_7wrP_DJF_CE.data.flatten(), [4.3,], model_color, '*')

# seas summer
plot_boxplot(axs_sx[1,1], seas5P_JJA_MAE.data.flatten(), [-0.3,], seas5_color, '*')
plot_boxplot(axs_dx[1,1], seas5P_JJA_ACC.data.flatten(), [1.7,], seas5_color, '*')
plot_boxplot(axs_dx[1,1], seas5P_JJA_CE.data.flatten(), [3.7,], seas5_color, '*')

# model summer
plot_boxplot(axs_sx[1,1], model_seas5_7wrP_JJA_MAE.data.flatten(), [0.3,], model_color, '*')
plot_boxplot(axs_dx[1,1], model_seas5_7wrP_JJA_ACC.data.flatten(), [2.3,], model_color, '*')
plot_boxplot(axs_dx[1,1], model_seas5_7wrP_JJA_CE.data.flatten(), [4.3,], model_color, '*')

# legend
legend_elements = [
    Patch(color=seas5_color, label='SEAS5'),
    Patch(color=model_color, label=r'AI-model with $I_{\rm wr}^{\rm SEAS5}$'),
]
fig.legend(handles=legend_elements, ncol=2, loc='upper left',  bbox_to_anchor=(0.1, 1.75), bbox_transform=axs_sx[0,0].transAxes)

# set some ticks options
axs_sx[0,0].set_xticks([0, 2, 4])
axs_sx[0,1].set_xticks([0, 2, 4])
axs_sx[1,0].set_xticks([0, 2, 4])
axs_sx[1,1].set_xticks([0, 2, 4])
axs_sx[0,0].set_xticklabels(['MAE', 'ACC', 'CE'])
axs_sx[0,1].set_xticklabels(['MAE', 'ACC', 'CE'])
axs_sx[1,0].set_xticklabels(['MAE', 'ACC', 'CE'])
axs_sx[1,1].set_xticklabels(['MAE', 'ACC', 'CE'])
[axs_sx[i,j].tick_params(axis='x', which='both', top=False) for i in range(2) for j in range(2)]
[axs_sx[i,j].tick_params(axis='x', which='minor', bottom=False) for i in range(2) for j in range(2)]
axs_dx[0,0].set_yticklabels([])
axs_dx[1,0].set_yticklabels([])


# vertical spans
[axs_sx[i,j].axvspan(-1.3, 1.1, color='seagreen', alpha=.1, lw=0) for i in range(2) for j in range(2)]
[axs_sx[i,j].axvspan(1.1, 5.3, color='rebeccapurple', alpha=.1, lw=0) for i in range(2) for j in range(2)]

# save
plt.savefig('plots/IwrSEAS5_skills.pdf', bbox_inches='tight')
