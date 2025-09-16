import sys, os, glob, joblib
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.special import erf


# torch
sys.path.append("../")
import torch
from src.models.anomrecon_module import AnomReconModule
from src.data.anomrecon_datamodule import AnomReconDataModule

from omegaconf import OmegaConf


def low_pass_weights(window, cutoff):
    """Calculate weights for a low pass Lanczos filter.

    Args:

    window: int
        The length of the filter window.

    cutoff: float
        The cutoff frequency in inverse time steps.

    """
    order = ((window - 1) // 2 ) + 1
    nwts = 2 * order + 1
    w = np.zeros([nwts])
    n = nwts // 2
    w[n] = 2 * cutoff
    k = np.arange(1., n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
    firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
    w[n-1:0:-1] = firstfactor * sigma
    w[n+1:-1] = firstfactor * sigma
    return w[1:-1]


# ------------------------------------------------
# Trained models routines
# ------------------------------------------------

def get_torch_models_infos(folder_path):
    PROJECT_ROOT = '..'
    folder_path  = os.path.expanduser(folder_path)
    ckpt         = glob.glob(f'{folder_path}/?/checkpoints/epoch_???.ckpt')
    n_models     = len(ckpt)

    models = []
    for i in range(0,n_models):
        os.environ['PROJECT_ROOT'] = PROJECT_ROOT
        config             = OmegaConf.load('%s/%s/.hydra/config.yaml' % (folder_path, i))
        indexes_paths      = config['data']['indexes_paths']
        anomalies_path     = config['data']['anomalies_path']
        land_sea_mask_path = config['data']['land_sea_mask_path']
        orography_path     = config['data']['orography_path']
        num_indexes        = config['data']['num_indexes']
        months             = config['data']['months']
        train_last_date    = config['data']['train_last_date']
        val_last_date      = config['data']['val_last_date']
        
        if i == 0:
            datamodule = AnomReconDataModule(
                indexes_paths=indexes_paths,
                anomalies_path=anomalies_path,
                land_sea_mask_path=land_sea_mask_path,
                orography_path=orography_path,
                months=months,
                num_indexes=num_indexes,
                train_last_date=train_last_date,
                val_last_date=val_last_date,
                scaler_path=f"{folder_path}/scaler.pkl"
                )
            datamodule.setup(stage='test')
        
        models.append(
            AnomReconModule.load_from_checkpoint(
                ckpt[i]
                ).to('cpu')
            )
        
        models[i].eval()

    return models, datamodule, config

def get_models_inputs(datamodule, start_year):
    years  = [datamodule.data_test[i][0][0] for i in range(len(datamodule.data_test))]
    months = [datamodule.data_test[i][0][1] for i in range(len(datamodule.data_test))]
    #days   = [pd.Period('%d-%d' % (z[0] + start_year, z[1])).days_in_month for z in zip(years, months)]
    days   = 1
    static_data = datamodule.data_test[0][0][3]

    times = pd.to_datetime(
        pd.DataFrame(
            data={'years': np.array(years) + start_year, 'months': months, 'days': days},
            dtype=int
            ),
        ).values

    return years, months, times, static_data

def get_model_out(model, years, months, times, indexes, static_data, scaler):
    
    outputs = []
    for i in range(len(years)):
        time = times[i]
        index = torch.tensor(
            indexes.sel(time=time).data,
            dtype=torch.float32
            )
        
        x = years[i], months[i], index, static_data
        x = [el.unsqueeze(0) for el in x]
        outputs.append(model(x).squeeze(0).detach())
    
    # rescale
    shape = np.shape(outputs)
    outputs = scaler.inverse_transform(np.reshape(outputs, (shape[0], -1))).reshape(shape)
    
    return outputs

def get_models_out(models, indexes, anom, datamodule, start_year=1940):
    model_out = xr.DataArray(
        dims=['time', 'lat', 'lon', 'number'],
        coords=dict(
            time=anom.time,
            lat=anom.lat,
            lon=anom.lon,
            number=np.arange(1,len(models)+1)
        )
    )
    
    years, months, times, static_data = get_models_inputs(datamodule, start_year)
    years = torch.tensor(indexes.time.dt.year.data - start_year)
    months = torch.tensor(indexes.time.dt.month.data)
    times = indexes.time
    scaler = datamodule.scaler

    for number in range(1, len(models)+1):
        model_out.loc[dict(time=times, number=number)] = \
            get_model_out(models[number-1], years, months, times, indexes, static_data, scaler)
        
    return model_out.dropna(dim='time')

def get_perturbed_models_out(models, pert_idxs, anom, datamodule, start_year=1940):    
    model_out = xr.DataArray(
        dims=['realization', 'time', 'lat', 'lon', 'number'],
        coords=dict(
            realization=pert_idxs.realization,
            time=pert_idxs.time,
            lat=anom.lat,
            lon=anom.lon,
            number=np.arange(1,len(models)+1)
        )
    )

    years, months, times, static_data = get_models_inputs(datamodule, start_year)
    scaler = datamodule.scaler


    for realization in pert_idxs.realization:
        rel_idxs = pert_idxs.sel(realization=realization)
        for number in range(1, len(models)+1):
            model_out.loc[dict(time=times, realization=realization, number=number)] = get_model_out(models[number-1], years, months, times, rel_idxs, static_data, scaler)
        
    return model_out.dropna(dim='time')

def models_with_SEAS5_indexes(models, seas5_idxs, anom, datamodule):

    models_seas5_index = xr.DataArray(
        dims=['time', 'lat', 'lon', 'forecastMonth', 'ensemble_member', 'number'],
        coords=dict(
            time=anom.time,
            lat=anom.lat,
            lon=anom.lon,
            forecastMonth=seas5_idxs.forecastMonth,
            ensemble_member=seas5_idxs.number.data,
            number=range(1, len(models)+1)
        )
    )

    for leadtime in seas5_idxs.forecastMonth.data:
        for ensemble_member in seas5_idxs.number.data:
            idxs = seas5_idxs.sel(forecastMonth=leadtime, number=ensemble_member)
            models_out = get_models_out(models, idxs, anom, datamodule)
            models_seas5_index.loc[
                dict(
                    time=models_out.time,
                    forecastMonth=leadtime,
                    ensemble_member=ensemble_member
                )] = models_out

    return models_seas5_index.dropna(dim='time')

def get_SEAS5_mae(seas5_anom, anom):
    # seas5 error
    seas5_mae = xr.DataArray(
        dims=['time', 'forecastMonth', 'lat', 'lon'],
        coords=dict(
            time=seas5_anom.time,
            forecastMonth=seas5_anom.forecastMonth,
            lon=seas5_anom.lon,
            lat=seas5_anom.lat,
        )
    )

    for leadtime in range(1, 7):
        for i in range(leadtime - 1, len(seas5_anom.time.data)):
            time = seas5_anom.time.data[i]
            forecast_reference_time = seas5_anom.time.data[i - leadtime + 1]
            ground_truth = anom.sel(time=time)
            prediction = seas5_anom.sel(time=forecast_reference_time, forecastMonth=leadtime)
            d = abs(ground_truth - prediction.mean(dim='number'))
            seas5_mae.loc[dict(forecastMonth=leadtime, time=forecast_reference_time)] = d
    return seas5_mae

def get_SEAS5_DJF(seas5_anom):
    seas5_decs =  seas5_anom[seas5_anom.time.dt.month == 11].sel(forecastMonth=2).drop_vars({'forecastMonth'})  # prediction of Dec from 1st of Nov
    seas5_jans =  seas5_anom[seas5_anom.time.dt.month == 11].sel(forecastMonth=3).drop_vars({'forecastMonth'})  # prediction of Jan from 1st of Nov
    seas5_febs =  seas5_anom[seas5_anom.time.dt.month == 11].sel(forecastMonth=4).drop_vars({'forecastMonth'})  # prediction of Feb from 1st of Nov

    # shift the time to the predicted time
    seas5_decs['time'] = pd.to_datetime(seas5_decs['time']) + pd.DateOffset(months=1)
    seas5_jans['time'] = pd.to_datetime(seas5_jans['time']) + pd.DateOffset(months=2)
    seas5_febs['time'] = pd.to_datetime(seas5_febs['time']) + pd.DateOffset(months=3)

    return xr.concat((seas5_decs, seas5_jans, seas5_febs), dim='time').sortby('time')

def get_SEAS5_season(seas5_anom, season):
    if season == 'winter':
        init = 11
    elif season == 'summer':
        init = 5
    else:
        return None

    seas5_month1 =  seas5_anom[seas5_anom.time.dt.month == init].sel(forecastMonth=2).drop_vars({'forecastMonth'})
    seas5_month2 =  seas5_anom[seas5_anom.time.dt.month == init].sel(forecastMonth=3).drop_vars({'forecastMonth'})
    seas5_month3 =  seas5_anom[seas5_anom.time.dt.month == init].sel(forecastMonth=4).drop_vars({'forecastMonth'})

    # shift the time to the predicted time
    seas5_month1['time'] = pd.to_datetime(seas5_month1['time']) + pd.DateOffset(months=1)
    seas5_month2['time'] = pd.to_datetime(seas5_month2['time']) + pd.DateOffset(months=2)
    seas5_month3['time'] = pd.to_datetime(seas5_month3['time']) + pd.DateOffset(months=3)

    return xr.concat((seas5_month1, seas5_month2, seas5_month3), dim='time').sortby('time')



# ------------------------------------------------
# Composite Model
# ------------------------------------------------
def life_cicle_monthly_mask(Iwr, clim_start, clim_end):
    n_clusters = len(Iwr.mode)
    std = Iwr.sel(time=slice(clim_start, clim_end)).std(dim='time')  # reference std
    cond1 = np.zeros((Iwr.shape), dtype=bool)
    for i in range(n_clusters):
        cond1[i] = (Iwr[i] > Iwr[np.delete(np.arange(n_clusters), i)]).all(dim='mode')
    cond2 = (
        ((Iwr > std) & cond1)
        )

    mask = cond1 & cond2
    return mask

def get_composite(var, mask, clim_start, clim_end):
    n_clusters = len(mask.mode)
    var_clim   = var.sel(time=slice(clim_start, clim_end))
    mask_clim  = mask.sel(time=slice(clim_start, clim_end))

    cluster_composite = xr.DataArray(
        dims=['mode', 'lat', 'lon'],
        coords=dict(
            mode=mask.mode,
            lat=var.lat,
            lon=var.lon
        )
    )

    # apply mask for each cluster
    for c in range(n_clusters): 
        cluster_composite[dict(mode=c)] = var_clim[mask_clim[c,:]].mean(dim='time')

    # no regime
    no_regime_anom = var_clim[(~mask_clim).all(dim='mode')].mean(dim='time')

    return cluster_composite, no_regime_anom


def get_composite_recon(var, Iwr, clim_start, clim_end, months=None):
    # select season
    if months:
        var = var[np.isin(var.time.dt.month, months)]
        Iwr = Iwr[:, np.isin(Iwr.time.dt.month, months)]

    # life-cicle monthly mask for WR attribution
    life_cicle_mask = life_cicle_monthly_mask(Iwr, clim_start, clim_end)

    # compute temperature and precipitation composite for each WR and no WR class
    composite, no_regime = get_composite(var, life_cicle_mask, clim_start, clim_end)

    # use the composite to provide reconstruction
    composite_recon = xr.where(
        life_cicle_mask.any(dim='mode'),
        (composite * life_cicle_mask).sum(dim='mode'),
        no_regime
    )

    return composite_recon


# ------------------------------------------------
# Anomalies related functions
# ------------------------------------------------
def monthly_anom_from_clim(monthly, daily_cal_clim, method, norm=False):
    month_anom_list = []

    # compute start DOY of each month and number of days in month
    for t in monthly['time']:
        year = t.dt.year.values
        month = t.dt.month.values
        days_in_month = t.dt.days_in_month.values
        # start day of month as DOY
        start_doy = pd.Timestamp(f'{year}-{month}-01').dayofyear - 1
        clim_slice = daily_cal_clim[start_doy:start_doy + days_in_month]

        # clim_slice is at the denominator
        if norm:
            clim_slice = 1. / clim_slice
        
        if method == 'mean':
            month_clim = clim_slice.mean(dim='dayofyear')
        elif method == 'sum':
            month_clim = clim_slice.sum(dim='dayofyear')
            
        if norm:
            month_anom_list.append(monthly.sel(time=t) * month_clim.mean(dim=['latitude', 'longitude']))
        else:    
            month_anom_list.append(monthly.sel(time=t) - month_clim)

    return xr.concat(month_anom_list, dim='time')


# ------------------------------------------------
# Skills
# ------------------------------------------------
def get_ce(ground_truth, reconstruction):
    numerator   = ((ground_truth - reconstruction)**2).sum(dim='time')
    denominator = ((ground_truth - ground_truth.mean(dim='time'))**2).sum(dim='time')
    return 1 - numerator / denominator

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

# ------------------------------------------------
# plotting functions
# ------------------------------------------------
def boxplot(q, color, label):
    bplot = plt.boxplot(
        [q[0].data.flatten(),
        q[1].data.flatten(),
        q[2].data.flatten(),
        np.nan ,np.nan ,np.nan ,np.nan ,np.nan ,np.nan ,np.nan ,
        q[3].data.flatten(),
        q[4].data.flatten()],
        tick_labels=['', '', '', '', '', '', '', '', '', '', '', ''],
        patch_artist=True,
        showmeans=True, boxprops={'color': color},
        medianprops={'linewidth': 0, 'color': color}, 
        meanprops={'marker': 's', 'markeredgecolor': color, 'markerfacecolor': color}, 
        showfliers=False,
        whiskerprops={'color': color}, capprops = {'color': color},
        label=label
    )
    [patch.set(facecolor=color, alpha=.2) for patch in bplot['boxes']]

def boxplot_NDJFM(q, color, label, pos):
    bplot = plt.boxplot(
        [q[-2].data.flatten(),
        q[-1].data.flatten(),
        q[0].data.flatten(),
        q[1].data.flatten(),
        q[2].data.flatten(),
        ],
        positions=pos,
        tick_labels=['', '', '', '', ''],
        patch_artist=True,
        showmeans=True, boxprops={'color': color},
        medianprops={'linewidth': 0, 'color': color}, 
        meanprops={'marker': 's', 'markeredgecolor': color, 'markerfacecolor': color}, 
        showfliers=False,
        whiskerprops={'color': color}, capprops = {'color': color},
        widths=.8,
        label=label
    )
    [patch.set(facecolor=color, alpha=.2, hatch='///') for patch in bplot['boxes']]


def boxplot_year(q, color, label, pos):
    bplot = plt.boxplot(
        [q[0].data.flatten(),
        q[1].data.flatten(),
        q[2].data.flatten(),
        q[3].data.flatten(),
        q[4].data.flatten(),
        q[5].data.flatten(),
        q[6].data.flatten(),
        q[7].data.flatten(),
        q[8].data.flatten(),
        q[9].data.flatten(),
        q[10].data.flatten(),
        q[11].data.flatten()],
        positions=pos,
        tick_labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov','Dec'],
        patch_artist=True,
        showmeans=True, boxprops={'color': color},
        medianprops={'linewidth': 0, 'color': color}, 
        meanprops={'marker': 's', 'markeredgecolor': color, 'markerfacecolor': color}, 
        showfliers=False,
        whiskerprops={'color': color}, capprops = {'color': color},
        label=label
    )
    [patch.set(facecolor=color, alpha=.2) for patch in bplot['boxes']]

def set_figsize(width_pt, height_to_width=1, subplots=(1, 1)):
    """
    Set figure dimensions to avoid scaling in LaTeX.
    :param width_pt: (float) document / column width in points
    :param height_to_width: (float) fraction of the width which you wish the figure to occupy
    :param subplots: (array-like) number of rows and columns of subplots
    :return: (tuple) dimensions of figure in inches
    """
    
    # Convert from pt to inches
    inches_per_pt = 1. / 72.27
    
    # Figure width in inches
    fig_width_in = width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * height_to_width * (float(subplots[0]) / subplots[1])

    return (fig_width_in, fig_height_in)


# -----------------
# perturbed indexes
# -----------------

# compute mean absolute value of a normally distributed variable
def mean_absolute_value(mu, sigma):
    phi = .5 * (1 + erf(-mu / (2**.5 * sigma)))
    return sigma * (2/np.pi)**.5 * np.exp(-mu**2/ (2*sigma**2)) + mu * (1 - 2*phi)

def sigma_from_mu_and_mare(mu, mare, min=1e-3, max=10, eps=1e-5):
    if mare == 0:
        return 0
    for sigma in np.linspace(min, max, int(1e6)):
        if abs(mean_absolute_value(mu, sigma) - mare) <= eps:
            return sigma
    raise Exception('sigma not found in the range %.2e -- %.2e' % (min, max))

def generate_perturbed_indexes(idxs, idx_mare, idx_mre=0, N=25):
    """
    generate N realization of gaussian perturbation such that the 
    mean relative error and the mean absolute relative error are given: 
    the first will fix the mean, the sencod the standard deviation

    idxs: (xarray) containing the indexes to be perturbed
    idx_mare: (float) the mean relavite absolute error of the perturbed indexes
    idx_mre: (float) the mean relative error of the perturbed indexes (default=0)
    N: (int) number of realizations
    """
    pert_idxs = np.zeros((N, *idxs.shape))

    # compute the factor f for the perturbation:
    mu    = idx_mre
    sigma = sigma_from_mu_and_mare(mu, idx_mare)
    f     = np.random.normal(loc=mu, scale=sigma, size=(N, *idxs.shape))

    # compute the perturbations
    for i in range(N):
        pert_idxs[i] = idxs.data * (1 + f[i])


    # transform to xarray for convenience
    pert_idxs = xr.DataArray(
        data=pert_idxs,
        dims=['realization', 'mode', 'time'],
        coords=dict(
            time=idxs.time,
            mode=idxs.mode,
            realization=range(1, N+1)
        )
    )

    return pert_idxs