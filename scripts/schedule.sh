#!/usr/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

source ".venv/bin/activate"

# -----------------
# Temperature
# -----------------

# train with 7 wr
python src/train.py -m experiment=europe_t2m_ERA5_7wr \
    hydra.sweep.dir=${paths.log_dir}/${task_name}/7wr \
    seed=0,3,67,72,12,111 \
    data.num_workers=0

# train with 4 wr in winter
python src/train.py -m experiment=europe_t2m_ERA5_4wr_winter \
    hydra.sweep.dir=${paths.log_dir}/${task_name}/4wr_winter \
    seed=0,3,67,72,12,111 \
    data.num_workers=0

# train with 4 wr in summer
python src/train.py -m experiment=europe_t2m_ERA5_4wr_summer \
    hydra.sweep.dir=${paths.log_dir}/${task_name}/4wr_summer \
    seed=0,3,67,72,12,111 \
    data.num_workers=0

# train with NAO index in winter
python src/train.py -m experiment=europe_t2m_ERA5_NAO_winter \
    hydra.sweep.dir=${paths.log_dir}/${task_name}/NAO_winter \
    seed=0,3,67,72,12,111 \
    data.num_workers=0

# train with NAO index in summer
python src/train.py -m experiment=europe_t2m_ERA5_NAO_summer \
    hydra.sweep.dir=${paths.log_dir}/${task_name}/NAO_summer \
    seed=0,3,67,72,12,111 \
    data.num_workers=0

# train with 0 wr
python src/train.py -m experiment=europe_t2m_ERA5_0wr \
    hydra.sweep.dir=${paths.log_dir}/${task_name}/0wr \
    seed=0,3,67,72,12,111 \
    data.num_workers=0 


# -----------------
# Precipitation
# -----------------

# train with 7 wr
python src/train.py -m experiment=europe_tp_ERA5_7wr \
    hydra.sweep.dir=${paths.log_dir}/${task_name}/7wr \
    seed=0,3,67,72,12,111 \
    data.num_workers=0

# train with 4 wr in winter
python src/train.py -m experiment=europe_tp_ERA5_4wr_winter \
    hydra.sweep.dir=${paths.log_dir}/${task_name}/4wr_winter \
    seed=0,3,67,72,12,111 \
    data.num_workers=0 

# train with 4 wr in summer
python src/train.py -m experiment=europe_tp_ERA5_4wr_summer \
    hydra.sweep.dir=${paths.log_dir}/${task_name}/4wr_summer \
    seed=0,3,67,72,12,111 \
    data.num_workers=0

# train with NAO index in winter
python src/train.py -m experiment=europe_tp_ERA5_NAO_winter \
    hydra.sweep.dir=${paths.log_dir}/${task_name}/NAO_winter \
    seed=0,3,67,72,12,111 \
    data.num_workers=0

# train with NAO index in summer
python src/train.py -m experiment=europe_tp_ERA5_NAO_summer \
    hydra.sweep.dir=${paths.log_dir}/${task_name}/NAO_summer \
    seed=0,3,67,72,12,111 \
    data.num_workers=0 

# train with 0 wr
python src/train.py -m experiment=europe_tp_ERA5_0wr \
    hydra.sweep.dir=${paths.log_dir}/${task_name}/0wr \
    seed=0,3,67,72,12,111 \
    data.num_workers=0 
