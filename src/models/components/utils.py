import torch, random
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from omegaconf import OmegaConf
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def max_norm(parameters,  min_val=-4, max_val=4):
     with torch.no_grad():
        for param in parameters:
            norm = param.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, min_val, max_val)
            param *= desired / norm

def log_param_hist(module):
    active_params = np.array([])

    for name, param in module.net.named_parameters():
        module.logger.experiment.add_histogram(name, param, module.global_step)
        if param.requires_grad:
            active_params = np.append(active_params, param.cpu().numpy())
    
    module.logger.experiment.add_histogram('all.param', active_params, module.global_step)

def plot_MSE_val(module, X, Y, out):

    # latitude and longitude
    static = X[3][0]
    lat = static[0][:,0].cpu() * 69
    lon = static[1][0,:].cpu() * 39

    # define figure
    start_year = 1940 #1806
    nrows = 4 #np.min([6, len(Y)])
    ncols = 2
    cmap='RdBu_r'
    if len(Y) < nrows:
        return
    
    fig, axs = plt.subplots(
        nrows, ncols, figsize=(6,11), sharex=True, sharey=True,
        subplot_kw={'projection': ccrs.PlateCarree()}
        )
    axs[0,0].text(.25, 1.2, 'Ground Truth', fontsize=14, transform=axs[0,0].transAxes)
    axs[0,1].text(.25, 1.2, 'Model Output', fontsize=14, transform=axs[0,1].transAxes)

    # chose nrows unique random integers
    # idx = random.sample(range(len(Y)), nrows)
    idx = np.arange(nrows)

    # plot
    min, max = -abs(Y[idx]).max(), abs(Y[idx]).max()
    for i in range(nrows):
        axs[i,0].set_ylabel('latitudine')

        pcm = axs[i,0].pcolormesh(lon, lat, Y[idx[i]].cpu(), vmin=min, vmax=max, cmap=cmap)
        axs[i,1].pcolormesh(lon, lat, out[idx[i]].cpu(), vmin=min, vmax=max, cmap=cmap)

        axs[i,0].coastlines()
        axs[i,1].coastlines()

        axs[i,0].set_title('%d-%d + lead time' % (X[0][idx[i]] + start_year, X[1][idx[i]]))
        axs[i,1].set_title('%d-%d + lead time' % (X[0][idx[i]] + start_year, X[1][idx[i]]))

        # Adjust the location of the subplots on the page to make room for the colorbar
        fig.subplots_adjust(bottom=0., top=.88, left=0.01, right=.99, wspace=0., hspace=0.)

    # colorbar on top
    cax = inset_axes(
    axs[0,0],
    width="200%", 
    height="10%",
    loc="upper left",
    bbox_to_anchor=(-.03, .5, 1, 1),
    bbox_transform=axs[0,0].transAxes,
    )
    fig.colorbar(pcm, cax=cax, orientation='horizontal', label='2mT anomaly (K)', ticklocation='top', location='top')

    module.logger.experiment.add_figure('MSE_val', fig, module.global_step)


def size_out(size_in, stride, padding):
    return (size_in + 2*padding) / stride

