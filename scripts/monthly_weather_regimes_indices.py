import argparse
import numpy as np
import xarray as xr
import xeofs as xe
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument("--anom", type=str, help="path to the NetCDF containing z500 anomalies")
parser.add_argument("--clim_start", type=str, help="(str) start date of the climatology period (e.g. 1981-01-01)")
parser.add_argument("--clim_end", type=str, help="(str) end date of the climatology period (e.g. 2010-12-31)")
parser.add_argument("--clusters", default=7, type=int, help="(int) number of clusters to use (default = 7)")
parser.add_argument("--season", default='ANN', type=str, help="(str) season: winter (DJF), summer (JJA), annual (ANN) (default = ANN)")
parser.add_argument("--out", help="path and name of the output NetCDF file" )
args = parser.parse_args()

# parameters
n_eof             = 7
n_eofs_for_kmeans = 7
n_clusters        = args.clusters
random_state      = 42  # to preserve WR ordering
lat_min, lat_max  = 30, 90   # Grams et. al
lon_min, lon_max  = -80, 40  # Grams et. al

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
print('EOF on the ERA5 climatology period...')
eof = xe.single.EOF(n_modes=n_eof, use_coslat=True, random_state=random_state)
eof.fit(anom_clim, dim="time")
clim_pc = eof.scores(normalized=False)
pc = eof.transform(anom, normalized=False)

var = eof.explained_variance_ratio()
print('  explained variance: %.2f' % (np.cumsum(var)[-1] * 100))

# K-means clustering of climatology PC
print('K-means clustering...')
km = KMeans(n_clusters=n_clusters, n_init=500, verbose=0, tol=1e-6, max_iter=5000, random_state=random_state)
km.fit(clim_pc.sel(mode=slice(1, n_eofs_for_kmeans)).T)

# cluster id
cid = xr.DataArray(
    name='cid',
    dims=['time'],
    coords=dict(
        time=pc.time,
    ),
    data=km.predict(pc.sel(mode=slice(1, n_eofs_for_kmeans)).T)
)

# save the clusterization for future usage
cid.to_netcdf(f'data/{n_clusters}_{args.season}_{args.clim_start}-{args.clim_end}_clusters.nc')

# compute the cluster centroids (i.e. the weather regimes)
print('Computing cluster centroids...')
clim_cid = cid.sel(time=slice(args.clim_start, args.clim_end))
clim_wr  = anom_clim.groupby(clim_cid).mean(dim='time')

# weather regime index
print('WR index computation...')
Pwr = clim_wr.dot(                                  # note: 1./sum(cos(lat))
        anom * np.cos(np.deg2rad(anom.latitude)),   # prefactor cancelate
        dim=['latitude', 'longitude']               # out in Iwr
    )
Pwr_avg = Pwr.sel(time=slice(args.clim_start, args.clim_end)).mean(dim='time')
Pwr_std = Pwr.sel(time=slice(args.clim_start, args.clim_end)).std(dim='time')
Iwr = (Pwr - Pwr_avg) / Pwr_std

# resample
Iwr_monthly = Iwr.resample(time='MS').mean().dropna(dim='time')  # resampling create nan where month not in season

# shift cid and rename to mode for compatibility with pca
Iwr_monthly['cid'] = Iwr_monthly['cid'] + 1
Iwr_monthly = Iwr_monthly.rename({'cid': 'mode'})

# save
print('Saving...')
Iwr_monthly.to_netcdf(args.out, mode='w')

print('Done.')