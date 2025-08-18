import argparse
import numpy as np
import xarray as xr
from scipy.stats import pearsonr

# plotting
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

# custom functions
from functions import *

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--winter_temp_acc", type=str, help="path to the numpy file containing the ACC for the winter temperature reconstruction")
parser.add_argument("--winter_temp_ce", type=str, help="path to the numpy file containing the CE for the winter temperature reconstruction")
parser.add_argument("--winter_prec_acc", type=str, help="path to the numpy file containing the ACC for the winter precipitation reconstruction")
parser.add_argument("--winter_prec_ce", type=str, help="path to the numpy file containing the CE for the winter precipitation reconstruction")
parser.add_argument("--winter_seas5_temp_skills", type=str, help="path to the numpy file containing the MAE, ACC and CE of SEAS5 for winter temperature")
parser.add_argument("--winter_seas5_prec_skills", type=str, help="path to the numpy file containing the MAE, ACC and CE of SEAS5 for winter precipitation")

parser.add_argument("--summer_temp_acc", type=str, help="path to the numpy file containing the ACC for the summer temperature reconstruction")
parser.add_argument("--summer_temp_ce", type=str, help="path to the numpy file containing the CE for the summer temperature reconstruction")
parser.add_argument("--summer_prec_acc", type=str, help="path to the numpy file containing the ACC for the summer precipitation reconstruction")
parser.add_argument("--summer_prec_ce", type=str, help="path to the numpy file containing the CE for the summer precipitation reconstruction")
parser.add_argument("--summer_seas5_temp_skills", type=str, help="path to the numpy file containing the MAE, ACC and CE of SEAS5 for summer temperature")
parser.add_argument("--summer_seas5_prec_skills", type=str, help="path to the numpy file containing the MAE, ACC and CE of SEAS5 for summer precipitation")

parser.add_argument("--start", type=str, default='2011', help="start date of the analysis (default 2011)")
args = parser.parse_args()

# ------------
# Load Data
# ------------
# winter
pert_modelsT_winter_med_acc = np.load(args.winter_temp_acc)
pert_modelsT_winter_med_ce  = np.load(args.winter_temp_ce)
pert_modelsP_winter_med_acc = np.load(args.winter_prec_acc)
pert_modelsP_winter_med_ce  = np.load(args.winter_prec_ce)
seas5T_winter_med_mae, seas5T_winter_med_acc, seas5T_winter_med_ce = np.load(args.winter_seas5_temp_skills)
seas5P_winter_med_mae, seas5P_winter_med_acc, seas5P_winter_med_ce = np.load(args.winter_seas5_prec_skills)

# summer
pert_modelsT_summer_med_acc = np.load(args.summer_temp_acc)
pert_modelsT_summer_med_ce  = np.load(args.summer_temp_ce)
pert_modelsP_summer_med_acc = np.load(args.summer_prec_acc)
pert_modelsP_summer_med_ce  = np.load(args.summer_prec_ce)
seas5T_summer_med_mae, seas5T_summer_med_acc, seas5T_summer_med_ce = np.load(args.summer_seas5_temp_skills)
seas5P_summer_med_mae, seas5P_summer_med_acc, seas5P_summer_med_ce = np.load(args.summer_seas5_prec_skills)


# ------------
# Plot
# ------------
# columnwidth = 248.9  # QJRMS
columnwidth = 205 # 197.5  # IJC
prec_color  = '#204487'
temp_color  = '#b81b22'
fig, [row1, row2] = plt.subplots(
    2, 1, figsize=set_figsize(columnwidth, .6, subplots=(2,1)), 
    sharex=True, gridspec_kw={'hspace':0}, layout="constrained")

row2.set_xlabel('WR index MARE (%)')
row1.set_ylabel('ACC and CE')
row2.set_ylabel('ACC and CE')
row2.set_xlim(-10, 210)
row1.set_ylim(-1, .8)
row2.set_ylim(-.5, 0.65)

# winter temperature
x = np.arange(0, 220, 20)
row1.hlines(seas5T_winter_med_acc, -10, 210, ls='-', color=temp_color, zorder=-1, alpha=.7)
row1.hlines(seas5T_winter_med_ce, -10, 210, ls='--', color=temp_color, zorder=-1, alpha=.7)
row1.scatter(x, pert_modelsT_winter_med_acc, marker='v', s=20, color=temp_color)
row1.scatter(x, pert_modelsT_winter_med_ce, marker='d', s=20, color=temp_color)

# winter precipitation
row1.hlines(seas5P_winter_med_acc, -10, 210, ls='-', color=prec_color, zorder=-1, alpha=.7)
row1.hlines(seas5P_winter_med_ce, -10, 210, ls='--', color=prec_color, zorder=-1, alpha=.7)
row1.scatter(x, pert_modelsP_winter_med_acc, marker='v', s=20, color=prec_color)
row1.scatter(x, pert_modelsP_winter_med_ce, marker='d', s=20, color=prec_color)

# summer temperature
row2.hlines(seas5T_summer_med_acc, -10, 210, ls='-', color=temp_color, zorder=-1, alpha=.7)
row2.hlines(seas5T_summer_med_ce, -10, 210, ls='--', color=temp_color, zorder=-1, alpha=.7)
row2.scatter(x, pert_modelsT_summer_med_acc, marker='v', s=20, color=temp_color)
row2.scatter(x, pert_modelsT_summer_med_ce, marker='d', s=20, color=temp_color)

# summer precipitation
row2.hlines(seas5P_summer_med_acc, -10, 210, ls='-', color=prec_color, zorder=-1, alpha=.7)
row2.hlines(seas5P_summer_med_ce, -10, 210, ls='--', color=prec_color, zorder=-1, alpha=.7)
row2.scatter(x, pert_modelsP_summer_med_acc, marker='v', s=20, color=prec_color)
row2.scatter(x, pert_modelsP_summer_med_ce, marker='d', s=20, color=prec_color)

# legend
legend_elements = [
    Line2D([0], [0], marker='v', color='gray', ls='', label='AI-model ACC'),
    Line2D([0], [0], marker='d', color='gray', ls='', label='AI-model CE'),
    Line2D([0], [0], marker='', color='darkgray', ls='-', label='SEAS5 ACC'),
    Line2D([0], [0], marker='', color='darkgray', ls='--', label='SEAS5 CE'),
    Patch(color=temp_color, label='temperature'),
    Patch(color=prec_color, label='precipitation'),
]
fig.legend(
    handles=legend_elements, ncol=2, loc='upper center',
    bbox_to_anchor=(0.35, 1.5), bbox_transform=row1.transAxes
    )

row1.set_yticks([-0.5, 0, 0.5])

# save
fig.tight_layout()
fig.savefig(f'plots/MARE_ACC_CE_{args.start}-2024.pdf', bbox_inches='tight')
