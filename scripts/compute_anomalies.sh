#!/usr/bin/bash
set -e

# set directories
repo_dir=$(realpath "$(pwd)/..")
data_dir=$(realpath "$(pwd)/../data")

# load the environment
source "$repo_dir/.venv/bin/activate"

# define the paths to the NetCDF files
era5_z500_noa="$data_dir/daily_z500_noa_19400101-20241231_regrid.nc"
era5_t2m_eu="$data_dir/daily_t2m_europe_19400101-20241231_regrid.nc"
era5_tp_eu="$data_dir/daily_tp_europe_19400101-20241231_regrid.nc"

seas5_z500_noa="$data_dir/SEAS5_z500_noa_19810101-20241231_biased.nc"
seas5_t2m_eu="$data_dir/SEAS5_t2m_europe_19810101-20241231_biased.nc"
seas5_tp_eu="$data_dir/SEAS5_tp_europe_1981-2025.nc"

# climatology
clim_start="1981"
clim_end="2010"

# output paths
era5_daily_z500_anom_noa="${era5_z500_noa/z500_/z500_anom_}"
era5_daily_t2m_anom_eu="${era5_t2m_eu/t2m_/t2m_anom_}"
era5_daily_tp_anom_eu="${era5_tp_eu/tp_/tp_anom_}"

era5_monthly_z500_noa="${era5_z500_noa/daily/monthly}"
era5_monthly_t2m_eu="${era5_t2m_eu/daily/monthly}"
era5_monthly_tp_eu="${era5_tp_eu/daily/monthly}"

seas5_z500_noa_bias="${seas5_z500_noa/_19810101-20241231_biased/_bias1981-2010_201101-202412}"
seas5_t2m_eu_bias="${seas5_t2m_eu/_19810101-20241231_biased/_bias1981-2010_201101-202412}"
seas5_tp_eu_bias="${seas5_tp_eu/_1981-2025/_bias1981-2010_201101-202412}"

seas5_z500_anom_noa_bias="${seas5_z500_noa_bias/z500_/z500_anom_}"
seas5_t2m_anom_eu_bias="${seas5_t2m_eu_bias/t2m_/t2m_anom_}"
seas5_tp_anom_eu_bias="${seas5_tp_eu_bias/tp_/tp_anom_}"


# ------------------------------------------------
# ERA5 ANOMALIES
# ------------------------------------------------
# compute ERA5 daily geopotential heigh anomalies
python daily_anomalies.py \
    --file $era5_z500_noa \
    --out $era5_daily_z500_anom_noa \
    --clim_start $clim_start \
    --clim_end $clim_end 

# compute ERA5 daily two-meter temperature anomalies
python daily_anomalies.py \
    --file $era5_t2m_eu \
    --out $era5_daily_t2m_anom_eu \
    --clim_start $clim_start \
    --clim_end $clim_end 

# compute ERA5 daily precipitation anomalies
python daily_anomalies.py \
    --file $era5_tp_eu \
    --out $era5_daily_tp_anom_eu \
    --clim_start $clim_start \
    --clim_end $clim_end 

# resample to monthly anomalies
python daily_to_monthly.py \
    --z500 $era5_daily_z500_anom_noa \
    --t2m $era5_daily_t2m_anom_eu \
    --tp $era5_daily_tp_anom_eu 

# ------------------------------------------------
# SEAS5 BIAS-CORRECTION
# ------------------------------------------------
# resample to monthly data
python daily_to_monthly.py \
    --z500 $era5_z500_noa \
    --t2m $era5_t2m_eu \
    --tp $era5_tp_eu \

# geopotential height
python SEAS5_bias_correction.py \
    -era5 $era5_monthly_z500_noa \
    -seas5 $seas5_z500_noa \
    -out $seas5_z500_noa_bias \
    -bias_start $clim_start \
    -bias_end $clim_end 

# two-meter temperature
python SEAS5_bias_correction.py \
    -era5 $era5_monthly_t2m_eu \
    -seas5 $seas5_t2m_eu \
    -out $seas5_t2m_eu_bias \
    -bias_start $clim_start \
    -bias_end $clim_end 

# precipitation
python SEAS5_bias_correction.py \
    -era5 $era5_monthly_tp_eu \
    -seas5 $seas5_tp_eu \
    -out $seas5_tp_eu_bias \
    -bias_start $clim_start \
    -bias_end $clim_end 

# ------------------------------------------------
# SEAS5 ANOMALIES
# ------------------------------------------------
# compute SEAS55 monthly geopotential height anomalies
python SEAS5_monthly_anomalies.py \
    --seas5 $seas5_z500_noa_bias \
    --out $seas5_z500_anom_noa_bias \
    --clim_start $clim_start \
    --clim_end $clim_end \

# compute SEAS5 monthly two-meter temperature anomalies
python SEAS5_monthly_anomalies.py \
    --seas5 $seas5_t2m_eu_bias \
    --out $seas5_t2m_anom_eu_bias \
    --clim_start $clim_start \
    --clim_end $clim_end 

# compute SEAS5 monthly precipitation anomalies
python SEAS5_monthly_anomalies.py \
    --seas5 $seas5_tp_eu_bias \
    --out $seas5_tp_anom_eu_bias \
    --clim_start $clim_start \
    --clim_end $clim_end 