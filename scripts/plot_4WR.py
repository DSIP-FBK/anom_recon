import argparse
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

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--season", type=str, help="(str) season: winter (DJF) or summer (JJA)")
args = parser.parse_args()

# params
n_clusters = 4

# load
cluster_mean_z500_season = xr.open_dataarray(f'data/{n_clusters}cluster_mean_z500_{args.season}.nc')
cluster_mean_anom_season = xr.open_dataarray(f'data/{n_clusters}cluster_mean_anom_{args.season}.nc')
no_regime_z500 = xr.open_dataarray(f'data/{n_clusters}clusters_no_regime_z500_{args.season}.nc')
no_regime_anom = xr.open_dataarray(f'data/{n_clusters}clusters_no_regime_anom_{args.season}.nc')


# ------------------
# Plot clusters mean
# ------------------
print('Plotting clusters mean...')
order = (0,1,2,3)  # adapt the order of the WR to match the title
# textwidth = 509  # QJRMS
# columnwidth = 248.9  # QJRMS
textwidth = 405  # IJC
columnwidth = 196.1  # IJC
fig, axs = plt.subplots(2, 2, figsize=set_figsize(columnwidth, .85, subplots=(2,2)),
                        sharex=True, sharey=True, layout="constrained",
                        gridspec_kw={'wspace': 0, 'hspace': 0},
                        subplot_kw={'projection': ccrs.TransverseMercator(central_longitude=-10, central_latitude=45)}
                        )
                        
palette = ['#1b2c62', '#204487', '#2d66af', '#2d66af', '#6bb4e1', '#94d3f3', '#b8e4f8', '#d9f0f9', '#f0f9fd', '#ffffff', '#ffffff', '#fef8de', '#fceda3', '#fdce67', '#fdaa31', '#f8812c', '#ed5729', '#da2f28', '#b81b22', '#921519']
cmap = LinearSegmentedColormap.from_list("", palette)

#cmap='RdBu_r'
if args.season == 'DJF':
    absmax = 240
    klevels = np.arange(4000, 6000, 100)
    cax_ticks = [-180, -120, -60, 0, 60, 120, 180]
    clabel = [5100, 5200, 5400, 5600, 5700]
elif args.season == 'JJA':
    absmax = 160
    klevels = np.arange(2000, 8000, 100)
    cax_ticks = [-160, -120, -80, -40, 0, 40, 80, 120, 160]
    clabel = [5400, 5600, 5800, 6000]

vmin = -absmax
vmax = absmax
step = 20
levels = np.arange(vmin, vmax+step, step)

cax = inset_axes(axs[0,0], width="180%", height="12%", loc="upper center", bbox_to_anchor=(0.53, .75, 1, 1), bbox_transform=axs[0,0].transAxes)
[axs[i,j].coastlines(linewidth=1, color='gray') for i in range(2) for j in range(2)]
[axs[i,j].set_aspect('auto') for i in range(2) for j in range(2)]

# plot WR regimes
i,j = 0,0
for k in range(1,n_clusters+1):
    co = axs[i,j].contour(cluster_mean_z500_season.longitude, cluster_mean_z500_season.latitude, cluster_mean_z500_season.sel(cid=order[k-1]).data, colors='k', linewidths=.5, levels=klevels, transform=ccrs.PlateCarree())
    cf = axs[i,j].contourf(cluster_mean_anom_season.longitude, cluster_mean_anom_season.latitude, cluster_mean_anom_season.sel(cid=order[k-1]).data, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels, transform=ccrs.PlateCarree(), extend='max')
    axs[i,j].clabel(co, clabel, inline=True, inline_spacing=-5, fontsize=7)

    j += 1
    if j == 2:
        i += 1
        j = 0

# colorbar
fs = 8
cticks = np.arange(-absmax, absmax+step, step).tolist()
cb = fig.colorbar(cf, cax=cax, orientation='horizontal', extend='neither', extendfrac=0)
cb.set_label(labelpad=-30, label='geopotential height anomaly (gpm)', fontsize=fs)
cax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, labelsize=fs)
cax.set_xticks(cax_ticks)

# plot titles (WR names)
if args.season == 'DJF':
    titles = ['Atlantic Ridge \n(AT)', 'Scandinavian Blocking \n(ScBL)', 'Zonal Regime \n(ZO)', 'Greenland Blocking \n(GL)']
elif args.season == 'JJA':
    titles = ['NAO-\n', 'Scandinavian Blocking \n(ScBL)', 'Zonal Regime \n(ZO)', 'Atlantic Ridge \n(AT)']
axs[0,0].set_title(titles[0], fontsize=fs)
axs[0,1].set_title(titles[1], fontsize=fs)
axs[1,0].set_title(titles[2], fontsize=fs)
axs[1,1].set_title(titles[3], fontsize=fs)
axs[0,0].set_extent([-80, 40, 30, 90], crs=ccrs.PlateCarree())

# save
[axs[i,j].gridlines(color='gray', linestyle=':') for i in range(2) for j in range(2)]
plt.savefig(f'plots/4{args.season}_clusters_mean_WR.pdf', bbox_inches='tight')
