import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from functions import set_figsize


# load files
Iseas5 = xr.open_dataarray('../data/SEAS5_monthly_z500_7wr_noa_bias1981-2010_201101-202412.nc').mean(dim='number')
Iera5  = xr.open_dataarray('../data/monthly_z500_7wr_noa_19400101-20241201.nc')

# compute correlation
corr = xr.DataArray(
    dims=['mode', 'forecastMonth'],
    coords=dict(
        mode = range(1, 8),
        forecastMonth = Iseas5.forecastMonth
    )
)

for forecastMonth in Iseas5.forecastMonth.data:
    forecast = Iseas5.sel(forecastMonth=forecastMonth)

    # shift Iseas5 for comparison with Iera5
    offset = pd.DateOffset(months=forecastMonth-1)
    reference_time  = pd.to_datetime(forecast['time'])
    prediction_time = reference_time + offset
    forecast['time'] = prediction_time

    corr.loc[dict(forecastMonth=forecastMonth)] = xr.corr(forecast, Iera5, dim='time')

# plot
columnwidth = 248. #236
fig = plt.figure(figsize=set_figsize(columnwidth, .8), layout="constrained")

corr.T.plot(cbar_kwargs={'label': 'correlation'})

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
plt.xticks(np.arange(1,8))
plt.savefig('plots/ERA5_SEAS5_indices_correlation.pdf')