import argparse
import numpy as np
import xarray as xr

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
parser.add_argument("--start", type=str, default='2011', help="start date of the analysis (default 2011)")
args = parser.parse_args()


# parameters 
lat_min, lat_max = 35, 70
lon_min, lon_max = -20, 30

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
model_anomT_sea = model_anomT[np.isin(model_anomT.time.dt.month, months)]

# models on precipitation
torch_modelsP, datamoduleP, configP = get_torch_models_infos(args.model_prec)
model_anomP     = xr.open_dataarray(configP['data']['anomalies_path']).rename({'latitude': 'lat', 'longitude': 'lon'})
idxsP           = xr.open_dataarray(configP['data']['indexes_paths'][0]).sel(mode=slice(1, configP['data']['num_indexes'][0]))
model_anomP_sea = model_anomP[np.isin(model_anomP.time.dt.month, months)]

# SEAS5 temperature
anom_seas5T      = xr.open_dataarray(args.seas5_temp)\
    .rename({'latitude': 'lat', 'longitude': 'lon'}).transpose('time', 'lat', 'lon', 'number', 'forecastMonth').mean(dim='number')
anom_seas5T_sea  = get_SEAS5_season(anom_seas5T, args.season)

# SEAS5 precipitation
anom_seas5P      = xr.open_dataarray(args.seas5_prec)\
    .rename({'latitude': 'lat', 'longitude': 'lon'}).transpose('time', 'lat', 'lon', 'number', 'forecastMonth').mean(dim='number')
anom_seas5P_sea  = get_SEAS5_season(anom_seas5P, args.season)

# cut to the European region
model_anomT_sea = model_anomT_sea.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
model_anomP_sea = model_anomP_sea.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
anom_seas5T_sea = anom_seas5T_sea.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
anom_seas5P_sea = anom_seas5P_sea.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))

# reduce all variables to common time-range
end = '2024'
common_time = np.intersect1d(
    model_anomT_sea.sel(time=slice(args.start, end)).time.values, 
    anom_seas5P_sea.sel(time=slice(args.start, end)).time.values
    )
model_anomT_sea = model_anomT_sea.sel(time=common_time)
model_anomP_sea = model_anomP_sea.sel(time=common_time)
anom_seas5T_sea = anom_seas5T_sea.sel(time=common_time)
anom_seas5P_sea = anom_seas5P_sea.sel(time=common_time)

# mask outside land
lsm = xr.open_dataarray('../data/lsm_regrid_shift_europe.nc').rename({'latitude': 'lat', 'longitude': 'lon'})
model_anomT_sea = model_anomT_sea.where(lsm > .8)
model_anomP_sea = model_anomP_sea.where(lsm > .8)
anom_seas5T_sea = anom_seas5T_sea.where(lsm > .8)
anom_seas5P_sea = anom_seas5P_sea.where(lsm > .8)

# SEAS5 skills temperature. Note: model_anomX_sea is ERA5 anomaly of X
seas5T_sea_mae     = abs(model_anomT_sea - anom_seas5T_sea).median(dim='time')
seas5T_sea_acc     = xr.corr(model_anomT_sea, anom_seas5T_sea, dim='time')
seas5T_sea_ce      = get_ce(model_anomT_sea, anom_seas5T_sea)
seas5T_sea_med_mae = seas5T_sea_mae.median().data
seas5T_sea_med_acc = seas5T_sea_acc.median().data
seas5T_sea_med_ce  = seas5T_sea_ce.median().data

# SEAS5 skills precipitation
seas5P_sea_mae     = abs(model_anomP_sea - anom_seas5P_sea).median(dim='time')
seas5P_sea_acc     = xr.corr(model_anomP_sea, anom_seas5P_sea, dim='time')
seas5P_sea_ce      = get_ce(model_anomP_sea, anom_seas5P_sea)
seas5P_sea_med_mae = seas5P_sea_mae.median().data
seas5P_sea_med_acc = seas5P_sea_acc.median().data
seas5P_sea_med_ce  = seas5P_sea_ce.median().data


# ------------------------
# Loop for the temperature
# ------------------------
"""
mare     = 0.
eps      = .2
mean_err = 0
N        = 50
max_mare = 2.0

print('\n-------------------------------------------------')
print('TEMPERATURE')
print('-------------------------------------------------')
print(f'realizations = {N}, inital MARE = {mare}, eps = {eps}')
print('               MAE      ACC      CE')
print(f'SEAS5         {seas5T_sea_med_mae:.3}    {seas5T_sea_med_acc:.3}     {seas5T_sea_med_ce:.3}')
print('-------------------------------------------------')
print('MARE')

pert_modelsT_sea_med_mae = []
pert_modelsT_sea_med_acc = []
pert_modelsT_sea_med_ce  = []
while mare <= max_mare:
    # compute temperature from perturbed indexes
    pert_idxsT       = generate_perturbed_indexes(idxsT, idx_mare=mare, idx_mre=mean_err, N=N)
    pert_modelsT     = get_perturbed_models_out(torch_modelsT, pert_idxsT, model_anomT, datamoduleT)\
                        .sel(time=common_time).mean(dim='number')    
    pert_modelsT     = pert_modelsT.where(lsm > .8)
    pert_modelsT_sea = pert_modelsT[:, np.isin(pert_modelsT.time.dt.month, months)]

    # skills
    pert_modelsT_sea_med_mae.append(abs(model_anomT_sea - pert_modelsT_sea).median().data)
    pert_modelsT_sea_med_acc.append(xr.corr(model_anomT_sea, pert_modelsT_sea, dim='time').median())
    pert_modelsT_sea_med_ce.append(
        get_ce(model_anomT_sea, pert_modelsT_sea).median().data
    )

    print(f'{mare:.3}           {pert_modelsT_sea_med_mae[-1]:.3}    {pert_modelsT_sea_med_acc[-1]:.3}     {pert_modelsT_sea_med_ce[-1]:.3}')
    mare += eps

# Save temperature
np.save(f'data/seas5T_{args.season}_med_mae_acc_ce_{N}_{args.start}-{end}.npy', [seas5T_sea_med_mae, seas5T_sea_med_acc, seas5T_sea_med_ce])
np.save(f'data/pert_modelsT_{args.season}_med_mae_{N}_{args.start}-{end}.npy', pert_modelsT_sea_med_mae)
np.save(f'data/pert_modelsT_{args.season}_med_acc_{N}_{args.start}-{end}.npy', pert_modelsT_sea_med_acc)
np.save(f'data/pert_modelsT_{args.season}_med_ce_{N}_{args.start}-{end}.npy', pert_modelsT_sea_med_ce)
"""
# --------------------------
# Loop for the precipitation
# --------------------------
mare     = 0.
eps      = .2
mean_err = 0
N        = 50
max_mare = 2.0

print('\n-------------------------------------------------')
print('PRECIPITATION')
print('-------------------------------------------------')
print(f'realizations = {N}, inital MARE = {mare}, eps = {eps}')
print('               MAE      ACC      CE')
print(f'SEAS5         {seas5P_sea_med_mae:.3}    {seas5P_sea_med_acc:.3}     {seas5P_sea_med_ce:.3}')
print('-------------------------------------------------')
print('MARE')

pert_modelsP_sea_med_mae = []
pert_modelsP_sea_med_acc = []
pert_modelsP_sea_med_ce  = []
while mare <= max_mare:
    # compute precipitation from perturbed indexes
    pert_idxsP       = generate_perturbed_indexes(idxsP, idx_mare=mare, idx_mre=mean_err, N=N)
    pert_modelsP     = get_perturbed_models_out(torch_modelsP, pert_idxsP, model_anomP, datamoduleP)\
                        .sel(time=common_time).mean(dim='number')
    pert_modelsP     = pert_modelsP.where(lsm > .8)
    pert_modelsP_sea = pert_modelsP[:, np.isin(pert_modelsP.time.dt.month, months)]

    # skills
    pert_modelsP_sea_med_mae.append(abs(model_anomP_sea - pert_modelsP_sea).median().data)
    pert_modelsP_sea_med_acc.append(xr.corr(model_anomP_sea, pert_modelsP_sea, dim='time').median())
    pert_modelsP_sea_med_ce.append(
        get_ce(model_anomP_sea, pert_modelsP_sea).median().data
    )


    print(f'{mare:.3}           {pert_modelsP_sea_med_mae[-1]:.3}     {pert_modelsP_sea_med_acc[-1]:.3}     {pert_modelsP_sea_med_ce[-1]:.3}')
    mare += eps

# Save precipitation
np.save(f'data/seas5P_{args.season}_med_mae_acc_ce_{N}_{args.start}-{end}.npy', [seas5P_sea_med_mae, seas5P_sea_med_acc, seas5P_sea_med_ce])
np.save(f'data/pert_modelsP_{args.season}_med_mae_{N}_{args.start}-{end}.npy', pert_modelsP_sea_med_mae)
np.save(f'data/pert_modelsP_{args.season}_med_acc_{N}_{args.start}-{end}.npy', pert_modelsP_sea_med_acc)
np.save(f'data/pert_modelsP_{args.season}_med_ce_{N}_{args.start}-{end}.npy', pert_modelsP_sea_med_ce)
