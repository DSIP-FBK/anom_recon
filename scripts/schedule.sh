#!/usr/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

source ".venv/bin/activate"

# -----------------
# Temperature
# -----------------

# train with 7 wr
python src/train.py -m experiment=europe_t2m_ERA5_7wr data.num_workers=0 seed=0,3,67,72,12,111

# train with 4 wr in winter
python src/train.py -m experiment=europe_t2m_ERA5_4wr_winter data.num_workers=0 seed=0,3,67,72,12,111

# train with 4 wr in summer
python src/train.py -m experiment=europe_t2m_ERA5_4wr_summer data.num_workers=0 seed=0,3,67,72,12,111

# train with NAO index in winter
python src/train.py -m experiment=europe_t2m_ERA5_NAO_winter data.num_workers=0 seed=0,3,67,72,12,111

# train with NAO index in summer
python src/train.py -m experiment=europe_t2m_ERA5_NAO_summer data.num_workers=0 seed=0,3,67,72,12,111

# train with 0 wr
python src/train.py -m experiment=europe_t2m_ERA5_0wr data.num_workers=0 seed=0,3,67,72,12,111


# -----------------
# Precipitation
# -----------------

# train with 7 wr
python src/train.py -m experiment=europe_tp_ERA5_7wr data.num_workers=0 seed=0,3,67,72,12,111

# train with 4 wr in winter
python src/train.py -m experiment=europe_tp_ERA5_4wr_winter data.num_workers=0 seed=0,3,67,72,12,111

# train with 4 wr in summer
python src/train.py -m experiment=europe_tp_ERA5_4wr_summer data.num_workers=0 seed=0,3,67,72,12,111

# train with NAO index in winter
python src/train.py -m experiment=europe_tp_ERA5_NAO_winter data.num_workers=0 seed=0,3,67,72,12,111

# train with NAO index in summer
python src/train.py -m experiment=europe_tp_ERA5_NAO_summer data.num_workers=0 seed=0,3,67,72,12,111

# train with 0 wr
python src/train.py -m experiment=europe_tp_ERA5_0wr data.num_workers=0 seed=0,3,67,72,12,111
