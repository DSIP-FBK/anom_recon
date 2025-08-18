import argparse
import numpy as np
import xarray as xr
from scipy.stats import pearsonr

# plotting
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

# custom functions
from functions import *

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_temp", type=str, help="path to the folder containing the model trained on temperature")
parser.add_argument("--model_prec", type=str, help="path to the folder containing the model trained on precipitation" )
parser.add_argument("--seas5_temp", type=str, help="path to the bias-corrected seas5 temperature anomalies")
parser.add_argument("--seas5_prec", type=str, help="path to the bias-corrected seas5 precipitation anomalies")
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

# models on temprature
torch_modelsT, datamoduleT, configT = get_torch_models_infos(args.model_temp)
model_anomT     = xr.open_dataarray(configT['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
idxsT           = xr.open_dataarray(configT['data']['indexes_paths'][0]).sel(mode=slice(1, configT['data']['num_indexes'][0]))
modelT          = get_models_out(torch_modelsT, idxsT, model_anomT, datamoduleT)
modelT_sea      = modelT[np.isin(modelT.time.dt.month, months)]
model_anomT_sea = model_anomT[np.isin(model_anomT.time.dt.month, months)]

# models on precipitation
# ToDo

# SEAS5
anom_seas5T      = xr.open_dataarray(args.seas5_temp).rename({'latitude': 'lat', 'longitude': 'lon'}).transpose('time', 'lat', 'lon', 'number', 'forecastMonth')
anom_seas5T_sea  = get_SEAS5_season(anom_seas5T, args.season)

# reduce all variables to common time-range
start, end      = '2011', '2024'
model_anomT     = model_anomT.sel(time=slice(start, end))
model_anomT_sea = model_anomT_sea.sel(time=slice(start, end))
anom_seas5T     = anom_seas5T.sel(time=slice(start, end))
idxsT           = idxsT.sel(time=slice(start, end))
modelT          = modelT.sel(time=slice(start, end))

# SEAS5 skills. Note: model_anomX_sea is ERA5 anomaly of X
seas5T_sea_mae     = abs(model_anomT_sea - anom_seas5T_sea.mean(dim='number')).mean(dim='time')
seas5T_sea_acc     = xr.corr(model_anomT_sea, anom_seas5T_sea.mean(dim='number'), dim='time')
seas5T_sea_ce      = get_ce(model_anomT_sea, anom_seas5T_sea.mean(dim='number'))
seas5T_sea_avg_mae = seas5T_sea_mae.mean().data
seas5T_sea_avg_acc = seas5T_sea_acc.mean().data
seas5T_sea_avg_ce  = seas5T_sea_ce.mean().data


# ------------------------
# Loop for the temperature
# ------------------------
mare     = 0.
eps      = .1
mean_err = 0
N        = 50
max_mare = 1.5

print('\n-------------------------------------------------')
print('TEMPERATURE')
print('-------------------------------------------------')
print(f'realizations = {N}, inital MARE = {mare}, eps = {eps}')
print('               MAE      ACC      CE')
print(f'SEAS5         {seas5T_sea_avg_mae:.3}    {seas5T_sea_avg_acc:.3}     {seas5T_sea_avg_ce:.3}')
print('-------------------------------------------------')
print('MARE')

pert_modelsT_sea_avg_mae = []
pert_modelsT_sea_avg_acc = []
pert_modelsT_sea_avg_ce  = []
while mare <= max_mare:
    # compute temperature and precipitation from perturbed indexes
    pert_idxsT       = generate_perturbed_indexes(idxsT, idx_mare=mare, idx_mre=mean_err, N=N)
    pert_modelsT     = get_perturbed_models_out(torch_modelsT, pert_idxsT, model_anomT, datamoduleT).sel(time=slice(start, end))
    pert_modelsT_sea =  pert_modelsT[:, np.isin(pert_modelsT.time.dt.month, months)]

    # skills
    pert_modelsT_sea_avg_mae.append(abs(model_anomT_sea - pert_modelsT_sea.mean(dim='number')).mean().data)
    pert_modelsT_sea_avg_acc.append(np.mean(
        [pearsonr(model_anomT_sea, pert_modelsT_sea.isel(realization=i).mean(dim='number'), axis=0).statistic for i in range(N)]
    ))
    pert_modelsT_sea_avg_ce.append(
        get_ce(model_anomT_sea, pert_modelsT_sea.mean(dim='number')).mean().data
    )

    print(f'{mare:.3}           {pert_modelsT_sea_avg_mae[-1]:.3}    {pert_modelsT_sea_avg_acc[-1]:.3}     {pert_modelsT_sea_avg_ce[-1]:.3}')
    mare += eps


# --------------------------
# Loop for the precipitation
# --------------------------
"""
mare     = 0.
eps      = .1
mean_err = 0
N        = 25
max_mare = 1.5

print('\n-------------------------------------------------')
print('PRECIPITATION')
print('-------------------------------------------------')
print(f'realizations = {N}, inital MARE = {mare}, eps = {eps}')
print('               MAE      ACC')
print(f'SEAS5         {seas5P_sea_avg_mae:.3}     {seas5P_sea_avg_acc:.3}')
print('-------------------------------------------------')
print('MARE')

pert_modelsP_sea_avg_mae = []
pert_modelsP_sea_avg_acc = []
while mare <= max_mare:
    # compute temperature and precipitation from perturbed indexes
    pert_idxsP       = generate_perturbed_indexes(idxsP, idx_mare=mare, idx_mre=0, N=25)
    pert_modelsP     = get_perturbed_models_out(torch_modelsP, pert_idxsP, model_anomP, datamoduleP).sel(time=slice(start, end))
    pert_modelsP_sea =  pert_modelsP[:, np.isin(pert_modelsP.time.dt.month, months)]

    # skills
    pert_modelsP_sea_avg_mae.append(abs(model_anomP - pert_modelsP_sea.mean(dim='number')).mean().data)
    pert_modelsP_sea_avg_acc.append(np.mean(
        [pearsonr(model_anomP, pert_modelsP_sea.isel(realization=i).mean(dim='number'), axis=0).statistic for i in range(N)]
    ))

    print(f'{mare:.3}           {pert_modelsP_sea_avg_mae[-1]:.3}     {pert_modelsP_sea_avg_acc[-1]:.3}')
    mare += eps
"""

# ------------
# Plot
# ------------
columnwidth = 248.
prec_color = 'cornflowerblue'
temp_color = 'indianred'
fig, ax1 = plt.subplots(1, 1, figsize=set_figsize(columnwidth, .7), layout="constrained")
ax2 = ax1.twinx()
ax1.set_xlabel(r'WR index MARE')
ax1.set_ylabel('MAE')
ax2.set_ylabel('ACC and CE')
ax1.set_xlim(-10, 160)
ax1.set_ylim(0.36, 0.57)
ax2.set_ylim(0.1, 0.65)

# temperature
x = np.linspace(0, mare, len(pert_modelsT_sea_avg_mae)) * 100
ax1.scatter(x, pert_modelsT_sea_avg_mae, marker='^', color=temp_color)
ax2.scatter(x, pert_modelsT_sea_avg_acc, marker='v', color=temp_color)
#ax2.scatter(x, pert_modelsT_sea_avg_acc, marker='s', color=temp_color)
ax1.hlines(seas5T_sea_avg_mae, -10, 200, ls='-', color=temp_color)
ax2.hlines(seas5T_sea_avg_acc, -10, 200, ls='--', color=temp_color)
#ax2.hlines(seas5T_sea_avg_ce, -10, 200, ls=':', color=temp_color)

# precipitation
# use v marker

# legend
legend_elements = [
    Line2D([0], [0], marker='^', color='gray', ls='', label='AI-model MAE'),
    Line2D([0], [0], marker='v', color='gray', ls='', label='AI-model ACC'),
    Line2D([0], [0], marker='s', color='gray', ls='', label='AI-model CE'),
    Line2D([0], [0], marker='', color='darkgray', ls='-', label='SEAS5 MAE'),
    Line2D([0], [0], marker='', color='darkgray', ls='--', label='SEAS5 ACC'),
    Line2D([0], [0], marker='', color='darkgray', ls=':', label='SEAS5 CE'),
    Patch(color=temp_color, label='temperature'),
    Patch(color='cornflowerblue', label='precipitation'),
]
fig.legend(handles=legend_elements, ncol=2, loc='upper center',  bbox_to_anchor=(0.5, 1.4), bbox_transform=ax1.transAxes)

# save
plt.savefig(f'plots/MARE_MAE_ACC_CE_{args.season}.pdf', bbox_inches='tight')