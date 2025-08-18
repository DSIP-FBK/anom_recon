import argparse
import numpy as np
import xarray as xr
import xeofs as xe
from sklearn.cluster import KMeans

# custom functions
from functions import low_pass_weights

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

"""
Script for plotting the WR (cluster means) for the climatology period.
"""

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--z500", type=str, help="path to the Netcdf containing the ERA5 geopotential at 500 hPa")
parser.add_argument("--clusters", default=7, type=int, help="(int) number of clusters to use (default = 7)")
args = parser.parse_args()

# parameters
n_eof             = 7
n_eofs_for_kmeans = 7
n_clusters        = args.clusters  # number of clusters to use
random_state      = 42  # to preserve WR ordering at every run

clim_start        = '1981-01-01' #'1979-01-11' # Grams et. al
clim_end          = '2010-12-31' #'2015-12-31' # Grams et. al

lat_min, lat_max = 30, 90   # Grams et. al
lon_min, lon_max = -80, 40  # Grams et. al

# graviational acceleration
g = 9.80665  # m s^-2


# -------------------------------------------
# Daily standardized anomalies (Grams et al.)
# -------------------------------------------
print('Computing anomalies...')

# ERA5 daily Z500 (from the start and end climatology)
z500 = xr.open_dataarray(args.z500).sel(time=slice(clim_start, clim_end)) / g

# calendar day climatology
cal_clim = z500.groupby('time.dayofyear').mean().pad(dayofyear=7, mode='wrap').rolling(center=True, dayofyear=15).mean().dropna(dim='dayofyear')

# daily anomalies 
anom = z500.groupby('time.dayofyear') - cal_clim

# 10 days low-pass filtered
window = 31
wgts = xr.DataArray(
    low_pass_weights(window, 1. / 10.),
    dims=['window']
    )
anom_filtered = anom.pad(time=15, mode='wrap').rolling(time=window, center=True).construct('window').dot(wgts).dropna(dim='time')

# normalization with the spatial averaged standard deviation
# note: this is done only in the spatial region of interest
cal_31day_std = anom_filtered.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)).groupby('time.dayofyear').std().pad(dayofyear=7, mode='wrap').rolling(center=True, dayofyear=15).mean().dropna(dim='dayofyear')
anom_norm = anom_filtered.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)).groupby('time.dayofyear') / cal_31day_std.mean(dim=['latitude', 'longitude'])


# ----------------------------------------------------
# Climatology EOF decomposition and K-means clustering
# ----------------------------------------------------
print('EOF decomposition and K-means clustering...')

# EOF decomposition and projection (climatology)
eof = xe.single.EOF(n_modes=n_eof, use_coslat=True, random_state=random_state)
eof.fit(anom_norm, dim="time")
clim_eof = eof.components(normalized=False)
clim_pc  = eof.scores(normalized=False)

var = eof.explained_variance_ratio()
print('  explained variance: %.2f' % (np.cumsum(var)[-1] * 100))

# K-means clustering of climatology PC
km = KMeans(n_clusters=n_clusters, n_init=500, verbose=0, tol=1e-6, max_iter=5000, random_state=random_state)
km.fit(clim_pc.sel(mode=slice(1, n_eofs_for_kmeans)).T)

# climatology cluster id
cid = xr.DataArray(
    name='cid',
    dims=['time'],
    coords=dict(
        time=clim_pc.time,
    ),
    data=km.predict(clim_pc.sel(mode=slice(1, n_eofs_for_kmeans)).T)
)


# -----------------------------
# WR index (Michel and Rivière)
# -----------------------------
print('Computing WR index and masking...')

Pwr = anom_norm.groupby(cid).mean(dim='time').dot(
        anom_norm * np.cos(np.deg2rad(anom_norm.latitude)),
        dim=['latitude', 'longitude']
    )
Pwr_avg = Pwr.mean(dim='time')
Pwr_std = Pwr.std(dim='time')
Iwr = (Pwr - Pwr_avg) / Pwr_std
Iwr = Iwr.T

# select climatology winter season (DJF)
z500_DJF = z500[np.isin(z500.time.dt.month, (12,1,2))]
anom_filtered_DJF = anom_filtered[np.isin(anom_filtered.time.dt.month, (12,1,2))]

# masking: index life-cycle criteria
cond1 = np.zeros((Iwr.T.shape), dtype=bool)
for i in range(7):
    cond1[i] = (Iwr[:,i] > Iwr[:,np.delete(np.arange(7), i)]).all(dim='cid')
cond2 = (
    ((Iwr > Iwr.std(dim='time')) & cond1.T).pad(time=5, method='constant', constant_values=0).rolling(time=11, center=True).sum().dropna(dim='time') > 4
    )

mask = cond1 & cond2.T
mask_DJF = mask[:,np.isin(Iwr.time.dt.month, (12,1,2))]

# ---------------------------------
# Compute climatology clusters mean 
# ---------------------------------
print('Computing climatology clusters mean...')

# z500 GPH clusters mean
cluster_mean_z500_DJF = xr.DataArray(
    dims=['cid', 'latitude', 'longitude'],
    coords=dict(
        cid=range(n_clusters),
        latitude=z500.latitude,
        longitude=z500.longitude
    )
)

# z500 GPH anomalies clusters mean
cluster_mean_anom_DJF = xr.DataArray(
    dims=['cid', 'latitude', 'longitude'],
    coords=dict(
        cid=range(n_clusters),
        latitude=anom.latitude,
        longitude=anom.longitude
    )
)

# apply mask for each cluster
for c in range(n_clusters): 
    cluster_mean_z500_DJF[dict(cid=c)] = z500_DJF[mask_DJF[c,:]].mean(dim='time')
    cluster_mean_anom_DJF[dict(cid=c)] = anom_filtered_DJF[mask_DJF[c,:]].mean(dim='time')

# no regime days
no_regime_z500 = z500_DJF[(~mask_DJF).prod(axis=0).astype(bool)].mean(dim='time')
no_regime_anom = anom_filtered_DJF[(~mask_DJF).prod(axis=0).astype(bool)].mean(dim='time')

# save
cluster_mean_z500_DJF.to_netcdf(f'data/{n_clusters}cluster_mean_z500_DJF.nc')
cluster_mean_anom_DJF.to_netcdf('data/{n_clusters}cluster_mean_anom_DJF.nc')
no_regime_z500.to_netcdf('data/{n_clusters}clusters_no_regime_z500_DJF.nc')
no_regime_anom.to_netcdf('data/{n_clusters}clusters_no_regime_anom_DJF.nc')
mask_DJF.to_netcdf('data/{n_clusters}clusters_mask_DJF.nc')