import argparse
import numpy as np
import pandas as pd
import xarray as xr
import xeofs as xe
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument("--anom", type=str, help="path to the NetCDF containing z500 anomalies")
parser.add_argument("--out", type=str, help="path and filename of the output NetCDF file" )
parser.add_argument("--clim_start", type=str, help="(str) start date of the climatology period (e.g. 1981-01-01)")
parser.add_argument("--clim_end", type=str, help="(str) end date of the climatology period (e.g. 2010-12-31)")
parser.add_argument("--season", type=str, help="(str) season: winter (DJF) or summer (JJA)")
args = parser.parse_args()

# parameters
n_eof             = 1
lat_min, lat_max  = 30, 90
lon_min, lon_max  = -80, 40

print('Reading NetCDF...')
anom = xr.open_dataarray(args.anom).sel(
    latitude=slice(lat_max, lat_min),
    longitude=slice(lon_min, lon_max)
    )

# reduce to the season of interest
if args.season == 'DJF':
    anom = anom.sel(time=anom['time.month'].isin([12, 1, 2]))
elif args.season == 'JJA':
    anom = anom.sel(time=anom['time.month'].isin([6, 7, 8]))

anom_clim = anom.sel(time=slice(args.clim_start, args.clim_end))

# EOF decomposition (climatology)
print('EOF decomposition in the climatology...')
eof = xe.single.EOF(n_modes=n_eof, use_coslat=True)
eof.fit(anom_clim, dim="time")
z500_clim_eof  = eof.components(normalized=False)
z500_clim_pca = eof.scores(normalized=False)

var = eof.explained_variance_ratio()
print('  explained variance: %.2f' % (np.cumsum(var)[-1] * 100))

# PCA: project onto climatology
z500_pca = eof.transform(anom, normalized=False)

# normalize the PCA scores
z500_pca = (z500_pca - z500_clim_pca.mean()) / z500_clim_pca.std()

# resample
z500_pca_monthly = z500_pca.resample(time='MS').mean().dropna(dim='time')  # resampling create nan where month not in season

# save
print('Saving...')
z500_pca_monthly.to_netcdf(args.out, mode='w')

