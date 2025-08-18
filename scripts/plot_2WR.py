import numpy as np
import xarray as xr

# plotting
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
from cartopy import feature

# custom functions
from functions import set_figsize

# params
n_clusters = 2

# load
cluster_mean_z500_DJF = xr.open_dataarray(f'data/{n_clusters}cluster_mean_z500_DJF.nc')
cluster_mean_anom_DJF = xr.open_dataarray(f'data/{n_clusters}cluster_mean_anom_DJF.nc')
no_regime_z500 = xr.open_dataarray(f'data/{n_clusters}clusters_no_regime_z500_DJF.nc')
no_regime_anom = xr.open_dataarray(f'data/{n_clusters}clusters_no_regime_anom_DJF.nc')


# ------------------
# Plot clusters mean
# ------------------
print('Plotting clusters mean...')
order = (0,1,2,3)  # adapt the order of the WR to match the title
textwidth = 509
columnwidth = 248
fig, axs = plt.subplots(1, 2, figsize=set_figsize(columnwidth, .8, subplots=(1,2)),
                        sharex=True, sharey=True, layout="constrained",
                        gridspec_kw={'wspace': 0, 'hspace': 0},
                        subplot_kw={'projection': ccrs.TransverseMercator(central_longitude=-10, central_latitude=45)}
                        )
                        
palette = ['#1b2c62', '#204487', '#2d66af', '#2d66af', '#6bb4e1', '#94d3f3', '#b8e4f8', '#d9f0f9', '#f0f9fd', '#ffffff', '#ffffff', '#fef8de', '#fceda3', '#fdce67', '#fdaa31', '#f8812c', '#ed5729', '#da2f28', '#b81b22', '#921519']
cmap = LinearSegmentedColormap.from_list("", palette)

#cmap='RdBu_r'
absmax = 240
vmin = -absmax
vmax = absmax
step = 20
levels = np.arange(vmin, vmax+step, step)
klevels = np.arange(4000, 6000, 100)

cax = inset_axes(axs[0], width="200%", height="8%", loc="upper center", bbox_to_anchor=(0.52, .5, 1, 1), bbox_transform=axs[0].transAxes)
[axs[i].coastlines(linewidth=1, color='gray') for i in range(len(axs))]
[axs[i].set_aspect('auto') for i in range(len(axs))]

# plot WR regimes
i,j = 0,0
for k in range(1,n_clusters+1):
    co = axs[k-1].contour(cluster_mean_z500_DJF.longitude, cluster_mean_z500_DJF.latitude, cluster_mean_z500_DJF.sel(cid=order[k-1]).data, colors='k', linewidths=.5, levels=klevels, transform=ccrs.PlateCarree())
    cf = axs[k-1].contourf(cluster_mean_anom_DJF.longitude, cluster_mean_anom_DJF.latitude, cluster_mean_anom_DJF.sel(cid=order[k-1]).data, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, transform=ccrs.PlateCarree(), extend='max')
    axs[k-1].clabel(co, [5100, 5200, 5400, 5600, 5700], inline=True, inline_spacing=-5, fontsize=7)

# colorbar
fs = 8.5
cticks = np.arange(-absmax, absmax+step, step).tolist()
cb = fig.colorbar(cf, cax=cax, orientation='horizontal', extend='neither', extendfrac=0)
cb.set_label(labelpad=-35, label='geopotential height anomaly (gpm)', fontsize=fs)
cax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, labelsize=fs)
cax.set_xticks([-180, -120, -60, 0, 60, 120, 180])

# plot titles (WR names)
axs[0].set_title('Zonal Regime (ZO)', fontsize=fs)
axs[1].set_title('Greenland Blocking (GL)', fontsize=fs)
axs[0].set_extent([-80, 40, 30, 90], crs=ccrs.PlateCarree())

# save
[axs[i].gridlines(color='gray', linestyle=':') for i in range(len(axs))]
plt.savefig('plots/2winter_clusters_mean_WR.pdf', bbox_inches='tight')
