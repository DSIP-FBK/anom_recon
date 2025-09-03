import argparse
import xarray as xr

# custom functions
from functions import low_pass_weights

parser = argparse.ArgumentParser()
parser.add_argument("--file", help="path to the NetCDF input file")
parser.add_argument("--out", help="path and filename of the output NetCDF file" )
parser.add_argument("--clim_start", help="(str) start date of the climatology period (e.g. 1981-01-01)")
parser.add_argument("--clim_end", help="(str) end date of the climatology period (e.g. 2010-12-31)")
args = parser.parse_args()

# parameters
g = 9.80665  # m s^-2

print('Reading NetCDF...', end='\r')
da = xr.open_dataarray(args.file)

if any([s in args.file for s in ('z500', 'geopotential')]):
    da = da / g  # despite z500 in the name, the NetCDF files contain the geopotential

print('Computing Anomalies...', end='\r')

# calendar day climatology
cal_clim = da.sel(time=slice(args.clim_start, args.clim_end))\
    .groupby('time.dayofyear').mean().pad(dayofyear=7, mode='wrap')\
        .rolling(center=True, dayofyear=15).mean().dropna(dim='dayofyear')
# anomalies 
anom = da.groupby('time.dayofyear') - cal_clim

# 10 days Lanczos filtered
window = 31
wgts = xr.DataArray(
    low_pass_weights(window, 1. / 10.),
    dims=['window']
    )
anom_filtered = anom.pad(time=15, mode='wrap')\
    .rolling(time=window, center=True)\
        .construct('window').dot(wgts).dropna(dim='time')

# calendar day standard deviation
cal_std = anom_filtered.sel(time=slice(args.clim_start, args.clim_end))\
    .groupby('time.dayofyear').std().pad(dayofyear=7, mode='wrap')\
        .rolling(center=True, dayofyear=15).mean().dropna(dim='dayofyear')

# normalized anomalies
anom_norm = anom_filtered.groupby('time.dayofyear') / cal_std.mean(dim=['latitude', 'longitude'])

# save file
print('Saving...', end='\r')
anom.to_netcdf(args.out)
anom_norm.to_netcdf(args.out.replace('_anom_', '_anom_norm_'))

# save mean and std to recover actual values
if 'z500' in args.file:
    name = 'z500'
elif 't2m' in args.file:
    name = 't2m'
elif 'tp' in args.file:
    name = 'tp'
cal_clim.to_netcdf(f'data/cal_clim_{name}_{args.clim_start}-{args.clim_end}.nc')
cal_std.to_netcdf(f'data/cal_std_{name}_{args.clim_start}-{args.clim_end}.nc')
