import argparse
import numpy as np
import xarray as xr
import xesmf

parser = argparse.ArgumentParser()
parser.add_argument("--files", help="list of NetCDF files (regex allowed)", nargs='*')
parser.add_argument("--out", help="path and filename of the output NetCDF file" )
args = parser.parse_args()

print('Opening files...')
da = xr.open_mfdataset(args.files)

lat_min, lat_max = da.latitude.min(), da.latitude.max()
lon_min, lon_max = da.longitude.min(), da.longitude.max()

# interpolate to 1deg x 1deg for efficiency
new_lats = np.arange(lat_max, lat_min-1, -1)
new_lons = np.arange(lon_min, lon_max+1, 1)
ds_out = xr.Dataset(
    {
        "latitude": (["latitude"], new_lats, {"units": "degrees_north"}),
        "longitude": (["longitude"], new_lons, {"units": "degrees_east"}),
    }
)

print('Regridding...')
regridder = xesmf.Regridder(da, ds_out, method='bilinear')
da_regrid = regridder(da)

print('Saving...')
da_regrid.to_netcdf(args.out)
