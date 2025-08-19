#!/usr/bin/bash
set -e

# set directories
repo_dir=$(realpath "$(pwd)/..")
data_dir=$(realpath "$(pwd)/../data")
log_dir=$(realpath "$(pwd)/../logs")

# create new dirs
mkdir -p plots data

# load the environment
source "$repo_dir/.venv/bin/activate"

start="2019"

model_7wr_temp="$log_dir/t2m_ERA5/7wr"  # path to the 7WR temperature model                                                                                          
model_7wr_prec="$log_dir/tp_ERA5/7wr"   # path to the 7WR precipitation model

model_4wr_temp_winter="$log_dir/t2m_ERA5/4wr_winter"  # path to the 4WR temperature model (winter)
model_4wr_temp_summer="$log_dir/t2m_ERA5/4wr_summer"  # path to the 4WR temperature model (summer)
model_4wr_prec_winter="$log_dir/tp_ERA5/4wr_winter"   # path to the 4WR precipitation model (winter)
model_4wr_prec_summer="$log_dir/tp_ERA5/4wr_summer"   # path to the 4WR precipitation model (summer)

model_NAO_temp_winter="$log_dir/t2m_ERA5/NAO_winter"  # path to the NAO temperature model (winter)
model_NAO_temp_summer="$log_dir/t2m_ERA5/NAO_summer"  # path to the NAO temperature model (summer)
model_NAO_prec_winter="$log_dir/tp_ERA5/NAO_winter"   # path to the NAO precipitation model (winter)
model_NAO_prec_summer="$log_dir/t2m_ERA5/NAO_summer"  # path to the NAO precipitation model (summer)

model_0wr_temp="$log_dir/t2m_ERA5/0wr"  # path to the 0WR temperature model
model_0wr_prec="$log_dir/tp_ERA5/0wr"   # path to the 0WR precipitation model

ERA5_z500="$data_dir/daily_z500_noa_19400101-20241231_regrid.nc"          # path to the ERA5 daily z500
ERA5_t2m="$data_dir/monthly_t2m_anom_europe_19400101-20241231_regrid.nc"  # path to the ERA5 monthly t2m anomalies
ERA5_tp="$data_dir/monthly_tp_anom_europe_19400101-20241231_merged.nc"    # path to the ERA5 monthly tp anomalies

seas5_temp="$data_dir/SEAS5_t2m_anom_europe_bias1981-2010_201101-202412.nc"      # path to the SEAS5 monthly t2m anomalies
seas5_prec="$data_dir/SEAS5_tp_anom_europe_bias1981-2010_201101-202412.nc"       # path to the SEAS5 monthly tp anomalies
Iwr_SEAS5="$data_dir/SEAS5_monthly_z500_7wr_noa_bias1981-2010_201101-202412.nc"  # path to the SEAS5 monthly 7 WR indices


# plot 7WR cluster means (Fig. 1)
python compute_nWR.py --z500 $ERA5_z500
python plot_7WR.py  # note: the order of the cluster could be different from the one in the paper, but the clusters are the same

# plot spatial MSE, ACC and CE with 7 WR
python plot_temp_prec_skills.py \
        -temp_model $model_7wr_temp \
        -prec_model $model_7wr_prec \
        -start $start

# plot skills with 7, 4, NAO and 0 indices
python plot_numWR_skills.py \
        -wr7_temp $model_7wr_temp \
        -wr7_prec $model_7wr_prec \
        -wr4_temp_winter $model_4wr_temp_winter \
        -wr4_temp_summer $model_4wr_temp_summer \
        -wr4_prec_winter $model_4wr_prec_winter \
        -wr4_prec_summer $model_4wr_prec_summer \
        -NAO_temp_winter $model_NAO_temp_winter \
        -NAO_temp_summer $model_NAO_temp_summer \
        -NAO_prec_winter $model_NAO_prec_winter \
        -NAO_prec_summer $model_NAO_prec_summer \
        -wr0_temp $model_0wr_temp \
        -wr0_prec $model_0wr_prec  \
        -start $start

# compute MARE vs MAE, ACC and CE in winter
python compute_mare_mae_acc_ce.py \
        --model_temp $model_7wr_temp \
        --model_prec $model_7wr_prec \
        --seas5_temp $seas5_temp \
        --seas5_prec $seas5_prec \
        --season winter \
        --start $start

# compute MARE vs MAE, ACC and CE in summer
python compute_mare_mae_acc_ce.py \
        --model_temp $model_7wr_temp \
        --model_prec $model_7wr_prec \
        --seas5_temp $seas5_temp \
        --seas5_prec $seas5_prec \
        --season summer \
        --start $start

# plot MARE vs MAE, ACC and CE
python plot_mare_acc_ce.py \
        --winter_temp_acc data/pert_modelsT_winter_med_acc_50_2019-12-2024-12.npy \
        --winter_temp_ce data/pert_modelsT_winter_med_ce_50_2019-12-2024-12.npy \
        --winter_prec_acc data/pert_modelsP_winter_med_acc_50_2019-12-2024-12.npy \
        --winter_prec_ce data/pert_modelsP_winter_med_ce_50_2019-12-2024-12.npy \
        --winter_seas5_temp_skills data/seas5T_winter_med_mae_acc_ce_50_2019-12-2024-12.npy \
        --winter_seas5_prec_skills data/seas5P_winter_med_mae_acc_ce_50_2019-12-2024-12.npy \
        --summer_temp_acc data/pert_modelsT_summer_med_acc_50_2019-06-2024-08.npy \
        --summer_temp_ce data/pert_modelsT_summer_med_ce_50_2019-06-2024-08.npy \
        --summer_prec_acc data/pert_modelsP_summer_med_acc_50_2019-06-2024-08.npy \
        --summer_prec_ce data/pert_modelsP_summer_med_ce_50_2019-06-2024-08.npy \
        --summer_seas5_temp_skills data/seas5T_summer_med_mae_acc_ce_50_2019-06-2024-08.npy \
        --summer_seas5_prec_skills data/seas5P_summer_med_mae_acc_ce_50_2019-06-2024-08.npy \
        --start $start

# plot model with SEAS5 WR index vs SEAS5
python compute_IwrSEAS5_models.py \
        --Iwr_SEAS5 $Iwr_SEAS5 \
        --seas5_temp $seas5_temp \
        --seas5_prec $seas5_prec \
        --model_temp $model_7wr_temp \
        --model_prec $seas5_prec

python plot_IwrSEAS5_skills.py \
        --anom_temp $ERA5_t2m \
        --anom_prec $ERA5_tp \
        --seas5_temp $seas5_temp \
        --seas5_prec $seas5_prec \
        --model_seas5_temp data/model_seas5_7wrT.nc \
        --model_seas5_prec data/model_seas5_7wrP.nc \
        --start $start
