import argparse, time
import pandas as pd
import xarray as xr
import xsdba

start_sec = time.time()

# --------------------------------
# Parse Arguments
#---------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-era5", type=str, help="path to ERA5 NetCDF (ground truth)")
parser.add_argument("-seas5", type=str, help="path to biased SEAS5 NetCDF")
parser.add_argument("-bias_start", type=str, help="start year of the bias correction period")
parser.add_argument("-bias_end", type=str, help="start year of the bias correction period")
parser.add_argument("-out", type=str, help="path and name of the output NetCDF")
args = parser.parse_args()

# parameters
min_lon, max_lon = -80, 40
min_lat, max_lat = 30, 90
start = '1981-01-01'
end   = '2024-12-31'

# --------------------------------
# Read Data
#---------------------------------
print('reading data...', end='\r')
era5 = xr.open_dataarray(args.era5)\
    .sel(time=slice(start, end), longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat))# / 1000  # useful for pressure

# SEAS biased. Note: only the first 25 members are for hindcast
seas5_biased = xr.open_dataarray(args.seas5)\
    .sel(number=slice(0,24))\
    .sel(forecast_reference_time=slice(start, end), longitude=slice(min_lon, max_lon), latitude=slice(max_lat, min_lat))# / 1000  # useful for pressure
seas5_biased = seas5_biased.rename({'forecast_reference_time': 'time'})#.sel(pressure_level=500).drop_vars(['pressure_level'])

# setting names
var_name  = 'var'
era5.name = var_name
seas5_biased.name = var_name

# assign units (need by xclim)
era5 = era5.assign_attrs({'units': ''})
seas5_biased = seas5_biased.assign_attrs({'units': ''})

# --------------------------------
# SEAS5 bias correction
#---------------------------------
print('bias correction...       ', end='\r')

start = pd.to_datetime(args.bias_end) + pd.DateOffset(days=1)
seas5_nobias = xr.DataArray(
    name=var_name,
    dims=['time', 'latitude', 'longitude', 'number', 'forecastMonth'],
    coords=dict(
        time=seas5_biased.sel(time=slice(start, None)).time.data,
        longitude=seas5_biased.longitude,
        latitude=seas5_biased.latitude,
        number=seas5_biased.number,
        forecastMonth=seas5_biased.forecastMonth
    ),
    attrs={"units": ""}
)

# observations
observation = era5.sel(time=slice(args.bias_start, args.bias_end))

# loop over the forecastMonths
for forecastMonth in seas5_biased.forecastMonth.data:
    print(f'bias correction  forecastMonth = {forecastMonth}  ', end='\r')
    offset = pd.DateOffset(months=forecastMonth-1)
    
    # select hindcast and forecast
    hindcast = seas5_biased.sel(time=slice(args.bias_start, args.bias_end), forecastMonth=forecastMonth)
    forecast = seas5_biased.sel(time=slice(start, None), forecastMonth=forecastMonth)

    # shift hindcast for comparison with ground truth
    hindcast_reference_time  = pd.to_datetime(hindcast['time'])
    hindcast_prediction_time = hindcast_reference_time + offset
    hindcast['time'] = hindcast_prediction_time

    # crop hindcast and observation to the same time window
    # (bias_start + offset : bias_end)
    observation = observation.sel(time=slice(hindcast.time.min(), args.bias_end))
    hindcast    = hindcast.sel(time=slice(None, args.bias_end))

    # bias correction of each ensemble member
    QM = xsdba.EmpiricalQuantileMapping.train(
            observation, hindcast, nquantiles=10, group="time.month", kind="+"
    )
    adjusted = QM.adjust(forecast, extrapolation="constant", interp="linear").reindex(latitude=list(hindcast.latitude))
    
    seas5_nobias.loc[dict(forecastMonth=forecastMonth)] = adjusted # * 1000 # useful for pressure


print('saving...                                       ', end='\r')
seas5_nobias.to_netcdf(args.out, mode='w')

print('done in %d sec                                  ' % (time.time() - start_sec))