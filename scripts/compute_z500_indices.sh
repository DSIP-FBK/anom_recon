#!/usr/bin/bash
set -e

# set directories
repo_dir=$(realpath "$(pwd)/..")
data_dir=$(realpath "$(pwd)/../data")

# load the environment
source "$repo_dir/.venv/bin/activate"

# define the paths to the NetCDF files
era5_daily_z500_anom="$data_dir/daily_z500_anom_noa_19400101-20241231_regrid.nc"
seas5_monthly_z500_anom="$data_dir/SEAS5_z500_anom_noa_bias1981-2010_201101-202412.nc"

# climatology
clim_start="1981-01-01"
clim_end="2010-12-31"

# output paths
era5_monthly_7wr=$HOME/anom_recon/seasonal/data/monthly_z500_7wr_noa_19400101-20241201.nc
era5_monthly_4wr_DJF="${era5_monthly_7wr/_7wr_/_4wr_DJF_}"
era5_monthly_4wr_JJA="${era5_monthly_7wr/_7wr_/_4wr_JJA_}"
era5_monthly_NAO_DJF="${era5_monthly_7wr/_7wr_/_NAO_DJF_}"
era5_monthly_NAO_JJA="${era5_monthly_7wr/_7wr_/_NAO_JJA_}"
seas5_monthly_7wr="$HOME/anom_recon/seasonal/data/SEAS5_monthly_z500_7wr_noa_bias1981-2010_201101-202412.nc"

# seven monthly WR indices from ERA5 daily mean geopotential height anomalies
python monthly_weather_regimes_indices.py \
    --anom $era5_daily_z500_anom \
    --out $era5_monthly_7wr \
    --clim_start $clim_start \
    --clim_end $clim_end \
    --clusters 7

# four monthly WR indices from ERA5 daily mean geopotential height anomalies
python monthly_weather_regimes_indices.py \
    --anom $era5_daily_z500_anom \
    --out $era5_monthly_4wr_DJF \
    --clim_start $clim_start \
    --clim_end $clim_end \
    --clusters 4 \
    --season "DJF"

python monthly_weather_regimes_indices.py \
    --anom $era5_daily_z500_anom \
    --out $era5_monthly_4wr_JJA \
    --clim_start $clim_start \
    --clim_end $clim_end \
    --clusters 4 \
    --season "JJA"

# First monthly principal component from ERA5 daily mean geopotential height anomalies (NAO)
python monthly_NAO_index.py \
    --anom $era5_daily_z500_anom \
    --out $era5_monthly_NAO_DJF \
    --clim_start $clim_start \
    --clim_end $clim_end \
    --season "DJF"

python monthly_NAO_index.py \
    --anom $era5_daily_z500_anom \
    --out $era5_monthly_NAO_JJA \
    --clim_start $clim_start \
    --clim_end $clim_end \
    --season "JJA"

# seven monthly WR indices from SEAS5 forecast of monthly mean geopotential height anomalies
python SEAS5_weather_regime_indices.py \
    --era5_anom $era5_daily_z500_anom \
    --seas5_anom $seas5_monthly_z500_anom \
    --out $seas5_monthly_7wr \
    --clim_start $clim_start \
    --clim_end $clim_end \
