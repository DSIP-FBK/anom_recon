import argparse
import numpy as np
import pandas as pd
import xarray as xr

# custom functions
from functions import low_pass_weights

parser = argparse.ArgumentParser()
parser.add_argument("--era5", help="path to the NetCDF input file containing the ERA5 data")
parser.add_argument("--seas5", help="path to the NetCDF input file containing the SEAS5 data")
parser.add_argument("--out", help="path and filename of the output NetCDF file" )
parser.add_argument("--clim_start", help="(str) start date of the climatology period (e.g. 1981-01-01)")
parser.add_argument("--clim_end", help="(str) end date of the climatology period (e.g. 2010-12-31)")
args = parser.parse_args()

print('Reading NetCDF...')
era5 = xr.open_dataarray(args.era5).sel(time=slice(args.clim_start, args.clim_end))
seas5 = xr.open_dataarray(args.seas5).sel(latitude=slice(None, 30), longitude=slice(-80, 40))#.sel(pressure_level=500).drop_vars({'pressure_level'})

print('Computing climatology...')
# calendar day climatology
cal_clim = era5.groupby('time.dayofyear').mean().pad(dayofyear=7, mode='wrap').rolling(center=True, dayofyear=15).mean().dropna(dim='dayofyear')
cal_clim_monthly = cal_clim.coarsen(dayofyear=30, boundary='trim').mean()
cal_clim_monthly = cal_clim_monthly.rename({'dayofyear': 'month'})
cal_clim_monthly['month'] = np.arange(1, 13)

print('Computing normalization...')
# anomalies 
era5_anom = (era5.groupby('time.dayofyear') - cal_clim)

# 10 days low-pass filtered
window = 31
wgts = xr.DataArray(
    low_pass_weights(window, 1. / 10.),
    dims=['window']
    )
era5_anom_filtered = era5_anom.pad(time=15, mode='wrap').rolling(time=window, center=True).construct('window').dot(wgts).dropna(dim='time')

# calendar day standard deviation
cal_std = era5_anom_filtered.groupby('time.dayofyear').std().pad(dayofyear=7, mode='wrap').rolling(center=True, dayofyear=15).mean().dropna(dim='dayofyear')
cal_std_monthly = cal_std.coarsen(dayofyear=30, boundary='trim').mean()
cal_std_monthly = cal_std_monthly.rename({'dayofyear': 'month'})
cal_std_monthly['month'] = np.arange(1, 13)

# --------------------------------
# Compute SEAS5 biased anomalies
#---------------------------------
print('computing SEAS5 anomalies...')

seas5_anom_norm = xr.DataArray(
    dims=['time', 'forecastMonth', 'latitude', 'longitude', 'number'],
    coords=dict(
        time=seas5.time,
        forecastMonth=seas5.forecastMonth,
        longitude=seas5.longitude,
        latitude=seas5.latitude,
        number=seas5.number
    )
)

for forecastMonth in range(1,7):
    print(f'  leadtime {forecastMonth}  ')
    prediction = seas5.sel(forecastMonth=forecastMonth)
    forecast_reference_time = pd.to_datetime(prediction['time'])
    prediction_times = forecast_reference_time + pd.DateOffset(months=forecastMonth-1)
    
    # shift time for comparison with ground truth
    prediction['time'] = prediction_times

    # compute the anomalies
    anom = prediction.groupby('time.month')  - cal_clim_monthly

    # normalize
    anom_norm = anom.groupby('time.month') / cal_std_monthly.mean(dim=['latitude', 'longitude'])
    
    # shift back to valid_time
    anom_norm['time'] = forecast_reference_time
    seas5_anom_norm.loc[dict(forecastMonth=forecastMonth)] = anom_norm
    
# save file
print('Saving...')
seas5_anom_norm.to_netcdf(args.out)
