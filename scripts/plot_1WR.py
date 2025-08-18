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
n_clusters = 1

# load
cluster_mean_z500_DJF = xr.open_dataarray(f'data/{n_clusters}cluster_mean_z500_DJF.nc')
cluster_mean_anom_DJF = xr.open_dataarray(f'data/{n_clusters}cluster_mean_anom_DJF.nc')
no_regime_z500 = xr.open_dataarray(f'data/{n_clusters}clusters_no_regime_z500_DJF.nc')
no_regime_anom = xr.open_dataarray(f'data/{n_clusters}clusters_no_regime_anom_DJF.nc')


# ------------------
# Plot clusters mean
# ------------------
print('Plotting clusters mean...')
textwidth = 509
columnwidth = 248
#fig = plt.figure(figsize=set_figsize(columnwidth, .8), layout='constrained')
#ax  = plt.axes(projection=ccrs.TransverseMercator(central_longitude=-10, central_latitude=45))         
fig, ax = plt.subplots(1, 1, figsize=set_figsize(columnwidth, .8, subplots=(1,1)),
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

cax = inset_axes(ax, width="100%", height="8%", loc="upper center", bbox_to_anchor=(0., .3, 1, 1), bbox_transform=ax.transAxes)
ax.coastlines(linewidth=1, color='gray')
ax.set_aspect('auto')

# plot WR regimes
co = ax.contour(cluster_mean_z500_DJF.longitude, cluster_mean_z500_DJF.latitude, cluster_mean_z500_DJF[0].data, colors='k', linewidths=.5, levels=klevels, transform=ccrs.PlateCarree())
cf = ax.contourf(cluster_mean_anom_DJF.longitude, cluster_mean_anom_DJF.latitude, cluster_mean_anom_DJF[0].data, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, transform=ccrs.PlateCarree(), extend='max')
ax.clabel(co, [5100, 5200, 5400, 5600, 5700], inline=True, inline_spacing=-5, fontsize=7)

# colorbar
fs = 8.5
cticks = np.arange(-absmax, absmax+step, step).tolist()
cb = fig.colorbar(cf, cax=cax, orientation='horizontal', extend='neither', extendfrac=0)
cb.set_label(labelpad=-35, label='geopotential height anomaly (gpm)', fontsize=fs)
cax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, labelsize=fs)
cax.set_xticks([-180, -120, -60, 0, 60, 120, 180])

# plot titles (WR names)
ax.set_title('Atlantic Trough (AT)', fontsize=fs)
ax.set_extent([-80, 40, 30, 90], crs=ccrs.PlateCarree())

# save
ax.gridlines(color='gray', linestyle=':')
plt.savefig('plots/1winter_clusters_mean_WR.pdf', bbox_inches='tight')
