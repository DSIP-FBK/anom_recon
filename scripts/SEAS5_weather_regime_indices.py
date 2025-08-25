import argparse
import numpy as np
import xarray as xr
import xeofs as xe
from sklearn.cluster import KMeans

"""
Compute SEAS5 weather regime index by projecting the SEAS5 monthly anomalies onto the ERA5 weather regimes
computing according to Grams et al., i.e. from the ERA5 daily anomalies (during the climatology period).
Note: projecting 
"""

parser = argparse.ArgumentParser()
parser.add_argument("--era5_anom", help="path to NetCDF file containing the ERA5 daily anomalies")
parser.add_argument("--seas5_anom", help="path to NetCDF file containing the SEAS5 monthly anomalies" )
parser.add_argument("--clim_start", help="(str) start date of the climatology period (e.g. 1981-01-01)")
parser.add_argument("--clim_end", help="(str) end date of the climatology period (e.g. 2010-12-31)")
parser.add_argument("--clusters", default=7, type=int, help="(int) number of clusters to use (default = 7)")
parser.add_argument("--out", help="path and name of the output NetCDF file" )
args = parser.parse_args()

# parameters
n_eof             = 7
n_eofs_for_kmeans = 7
n_clusters        = args.clusters
lat_min, lat_max  = 30, 90   # Grams et. al
lon_min, lon_max  = -80, 40  # Grams et. al

print('Reading NetCDF...')
# era5 anomalies (in the climatology period)
era5_anom_clim = xr.open_dataarray(args.era5_anom).sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max), time=slice(args.clim_start, args.clim_end))
# seas5 anomalies
seas5_anom = xr.open_dataarray(args.seas5_anom).sel(number=slice(None, 24), latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
# clustering
cid = xr.open_dataarray(f'data/{n_clusters}_ANN_{args.clim_start}-{args.clim_end}_clusters.nc').sel(time=slice(args.clim_start, args.clim_end))

# era5 mean cluster centers
era5_clim_wr = era5_anom_clim.groupby(cid).mean(dim='time')

# era5 Pwr. Note: 1./sum(cos(lat)) cancelate out in the normalization
era5_Pwr = era5_clim_wr.dot(
    era5_anom_clim * np.cos(np.deg2rad(era5_anom_clim.latitude)),
    dim=['latitude', 'longitude']
    )
era5_Pwr_avg = era5_Pwr.mean(dim='time')
era5_Pwr_std = era5_Pwr.std(dim='time')

# ---------------------
# SEAS5 Weather Regimes
# ---------------------
seas5_Pwr = era5_clim_wr.dot(
    seas5_anom * np.cos(np.deg2rad(seas5_anom.latitude)),
    dim=['latitude', 'longitude']
    )
seas5_Iwr = (seas5_Pwr - era5_Pwr_avg) / era5_Pwr_std

print('Finalizing...')
# shift cid and rename to mode for compatibility with pca
seas5_Iwr['cid'] = seas5_Iwr['cid'] + 1
seas5_Iwr        = seas5_Iwr.rename({'cid': 'mode'})

print('Saving...')
seas5_Iwr.to_netcdf(args.out, mode='w')

print('Done.')