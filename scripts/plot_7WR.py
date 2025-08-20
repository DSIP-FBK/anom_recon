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
n_clusters = 7

# load
cluster_mean_z500_DJF = xr.open_dataarray(f'data/{n_clusters}cluster_mean_z500_DJF.nc')
cluster_mean_anom_DJF = xr.open_dataarray(f'data/{n_clusters}cluster_mean_anom_DJF.nc')
no_regime_z500 = xr.open_dataarray(f'data/{n_clusters}clusters_no_regime_z500_DJF.nc')
no_regime_anom = xr.open_dataarray(f'data/{n_clusters}clusters_no_regime_anom_DJF.nc')


# ------------------
# Plot clusters mean
# ------------------
print('Plotting clusters mean...')
order = (3,4,5,2,1,0,6)  # adapt the order of the WR to match the title
textwidth = 405
fig, axs = plt.subplots(2, 4, figsize=set_figsize(textwidth, .85, subplots=(2,4)),
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

cax = inset_axes(axs[0,0], width="400%", height="12%", loc="upper center", bbox_to_anchor=(0.55, .7, 1, 1), bbox_transform=axs[0,1].transAxes)
[axs[i,j].coastlines(linewidth=1, color='gray') for i in range(len(axs[:,0])) for j in range(len(axs[0,:]))]
[axs[i,j].set_aspect('auto') for i in range(len(axs[:,0])) for j in range(len(axs[0,:]))]

# plot WR regimes
i,j = 0,0
for k in range(1,n_clusters+1):
    #axs[i,j].axis('off')
    co = axs[i,j].contour(cluster_mean_z500_DJF.longitude, cluster_mean_z500_DJF.latitude, cluster_mean_z500_DJF.sel(cid=order[k-1]).data, colors='k', linewidths=.5, levels=klevels, transform=ccrs.PlateCarree())
    cf = axs[i,j].contourf(cluster_mean_anom_DJF.longitude, cluster_mean_anom_DJF.latitude, cluster_mean_anom_DJF.sel(cid=order[k-1]).data, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, transform=ccrs.PlateCarree(), extend='max')
    
    #axs[i,j].clabel(co, fontsize=7)
    axs[i,j].clabel(co, [5100, 5200, 5400, 5600, 5700], inline=True, inline_spacing=-5, fontsize=7)
    


    j += 1
    if j == len(axs[0,:]):
        i += 1
        j = 0

# plot no regime
#axs[-1,-1].axis('off')
axs[-1,-1].contour(no_regime_z500.longitude, no_regime_z500.latitude, no_regime_z500, colors='black', levels=klevels, linewidths=.5, transform=ccrs.PlateCarree())
axs[-1,-1].contourf(no_regime_anom.longitude, no_regime_anom.latitude, no_regime_anom.data, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, transform=ccrs.PlateCarree())

# colorbar
fs = 8  # 9.5
cticks = np.arange(-absmax, absmax+step, step).tolist()
cb = fig.colorbar(cf, cax=cax, orientation='horizontal', extend='neither', extendfrac=0)
cb.set_label(labelpad=-30, label='geopotential height anomaly (gpm)', fontsize=fs)
cax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, labelsize=fs)
cax.set_xticks([-180, -120, -60, 0, 60, 120, 180])

# plot titles (WR names)
axs[0,0].set_title('Atlantic Trough \n(AT)', fontsize=fs)
axs[0,1].set_title('Zonal Regime \n(ZO)', fontsize=fs)
axs[0,2].set_title('Scandinavian Trough \n(ScTr)', fontsize=fs)
axs[0,3].set_title('Atlantic Ridge \n(AR)', fontsize=fs)
axs[1,0].set_title('European Blocking \n(EuBL)', fontsize=fs)
axs[1,1].set_title('Scandinavian Blocking \n(ScBL)', fontsize=fs)
axs[1,2].set_title('Greenland Blocking \n(GL)', fontsize=fs)
axs[1,3].set_title('No Regime \n', fontsize=fs)
axs[0,0].set_extent([-80, 40, 30, 90], crs=ccrs.PlateCarree())

# save
[axs[i,j].gridlines(color='gray', linestyle=':') for i in range(len(axs[:,0])) for j in range(len(axs[0,:]))]
plt.savefig('plots/7winter_clusters_mean_WR.pdf', bbox_inches='tight')
