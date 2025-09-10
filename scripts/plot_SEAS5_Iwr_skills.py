import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from functions import set_figsize
from matplotlib.colors import LinearSegmentedColormap

def crps_1d(ens, obs):
    """
    ens: shape (n_members,)
    obs: scalar
    """
    n = ens.size
    term1 = abs(ens - obs).mean(dim='number')
    term2 = 0
    for i in ens.number:
        term2 += abs(ens.sel(number=i) - ens).mean(dim='number')
    term2 = 0.5 * term2 / n
    return (term1 - term2).mean(dim='time')

def crps_vectorized(forecast, obs):
    """
    forecast: xarray.DataArray, dims ('number','time')
    obs: xarray.DataArray, dims ('time',)
    returns: scalar CRPS averaged over time
    """
    # term1: mean absolute difference between ensemble and obs
    term1 = abs(forecast - obs).mean(dim='number')

    # term2: mean absolute difference between all ensemble member pairs
    f1 = forecast.expand_dims('number2', axis=0)  # shape (1, number, time)
    f2 = forecast.expand_dims('number2', axis=1)  # shape (number,1,time)
    term2 = 0.5 * abs(f1 - f2).mean(dim=['number','number2'])

    # average over time
    return (term1 - term2).mean(dim='time')

# load files
Iseas5 = xr.open_dataarray('../data/SEAS5_monthly_z500_7wr_noa_bias1981-2010_201101-202412.nc')
Iera5  = xr.open_dataarray('../data/monthly_z500_7wr_noa_19400101-20241201.nc')

# ordering follow Figure 1
names    = ['AT', 'ZO', 'ScTr', 'AR', 'EuBL', 'ScBL', 'GL']
order    = [3,4,5,2,1,0,6]
Iseas5_r = Iseas5.isel(mode=order).assign_coords(mode=names)
Iera5_r  = Iera5.isel(mode=order).assign_coords(mode=names)

mae = xr.DataArray(
    dims=['mode', 'forecastMonth'],
    coords=dict(
        mode = names,
        forecastMonth = Iseas5.forecastMonth
    )
)

corr = xr.DataArray(
    dims=['mode', 'forecastMonth'],
    coords=dict(
        mode = names,
        forecastMonth = Iseas5.forecastMonth
    )
)

crps = xr.DataArray(
    dims=['mode', 'forecastMonth'],
    coords=dict(
        mode = names,
        forecastMonth = Iseas5.forecastMonth
    )
)

for forecastMonth in Iseas5.forecastMonth.data:
    forecast = Iseas5_r.sel(forecastMonth=forecastMonth)

    # shift Iseas5 for comparison with Iera5
    offset = pd.DateOffset(months=forecastMonth-1)
    reference_time  = pd.to_datetime(forecast['time'])
    prediction_time = reference_time + offset
    forecast = forecast.assign_coords(time=prediction_time)

    mae.loc[dict(forecastMonth=forecastMonth)] = abs(forecast.mean(dim='number') - Iera5_r).mean(dim='time')
    corr.loc[dict(forecastMonth=forecastMonth)] = xr.corr(forecast.mean(dim='number'), Iera5_r, dim='time')
    crps.loc[dict(forecastMonth=forecastMonth)] = crps_vectorized(forecast, Iera5_r)

# plot
columnwidth = 197
palette = ['#1b2c62', '#204487', '#2d66af', '#2d66af', '#6bb4e1', '#94d3f3', '#b8e4f8', '#d9f0f9', '#f0f9fd', '#ffffff', '#ffffff', '#fef8de', '#fceda3', '#fdce67', '#fdaa31', '#f8812c', '#ed5729', '#da2f28', '#b81b22', '#921519']
cmap1 = LinearSegmentedColormap.from_list("", palette)
palette = ['#ffffff', '#fef8de', '#fceda3', '#fdce67', '#fdaa31', '#f8812c', '#ed5729', '#da2f28', '#b81b22', '#921519']
cmap2 = LinearSegmentedColormap.from_list("", palette)

fig, axs = plt.subplots(2, 1, figsize=set_figsize(columnwidth, .7, (2,1)), layout="constrained")


im = crps.T.plot(cbar_kwargs={'label': 'CRPS'}, xlim=(-0.5, 6.5), cmap=cmap2, vmin=0, vmax=crps.max(), ax=axs[0])
corr.T.plot(cbar_kwargs={'label': 'correlation'}, xlim=(-0.5, 6.5), cmap=cmap1, vmin=-1, vmax=1, ax=axs[1])

axs[0].set_xlabel('')
axs[0].set_ylabel('forecast month')
axs[1].set_xlabel(r'$I_{\rm wr}$ index')
axs[1].set_ylabel('forecast month')
#axs[0].set_xticks([])
[axs[i].tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False) for i in range(2)]
im.colorbar.set_label('CRPS', labelpad=16)

plt.savefig('plots/ERA5_SEAS5_indices_skills.pdf')