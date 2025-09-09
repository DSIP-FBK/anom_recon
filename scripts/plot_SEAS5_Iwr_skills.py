import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from functions import set_figsize
from matplotlib.colors import LinearSegmentedColormap


# load files
Iseas5 = xr.open_dataarray('../data/SEAS5_monthly_z500_7wr_noa_bias1981-2010_201101-202412.nc').mean(dim='number')
Iera5  = xr.open_dataarray('../data/monthly_z500_7wr_noa_19400101-20241201.nc')

# ordering follow Figure 1
names    = ['AT', 'ZO', 'ScTr', 'AR', 'EuBL', 'ScBL', 'GL']
order    = [3,4,5,2,1,0,6]
Iseas5_r = Iseas5.isel(mode=order).assign_coords(mode=names)
Iera5_r  = Iera5.isel(mode=order).assign_coords(mode=names)

# compute correlation
corr = xr.DataArray(
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
    forecast['time'] = prediction_time

    corr.loc[dict(forecastMonth=forecastMonth)] = xr.corr(forecast, Iera5_r, dim='time')

# plot
columnwidth = 248. #236
fig = plt.figure(figsize=set_figsize(columnwidth, .8), layout="constrained")
palette = ['#1b2c62', '#204487', '#2d66af', '#2d66af', '#6bb4e1', '#94d3f3', '#b8e4f8', '#d9f0f9', '#f0f9fd', '#ffffff', '#ffffff', '#fef8de', '#fceda3', '#fdce67', '#fdaa31', '#f8812c', '#ed5729', '#da2f28', '#b81b22', '#921519']
cmap = LinearSegmentedColormap.from_list("", palette)

corr.T.plot(cbar_kwargs={'label': 'correlation'}, xlim=(-0.5, 6.5), cmap=cmap, vmin=-1, vmax=1)

plt.title(r'$I_{\rm wr}$ and $I^{\rm SEAS5}_{\rm wr}$ correlation')
plt.xlabel(r'$I_{\rm wr}$ index')
plt.ylabel('forecast month')
plt.tick_params(
    axis='both',
    which='both',
    bottom=False,
    top=False,
    left=False,
    right=False
)
plt.xticks(np.arange(0,7))
plt.savefig('plots/ERA5_SEAS5_indices_correlation.pdf')