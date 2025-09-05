import argparse
import xarray as xr

parser = argparse.ArgumentParser()
parser.add_argument("--z500", type=str, help="path to the Netcdf containing the daily ERA5 Z500 data")
parser.add_argument("--t2m", type=str, help="path to the Netcdf containing the daily ERA5 t2m data")
parser.add_argument("--tp", type=str, help="path to the Netcdf containing the daily ERA5 tp data")
args = parser.parse_args()

# load data
z500 = xr.open_dataarray(args.z500)
t2m  = xr.open_dataarray(args.t2m)
tp   = xr.open_dataarray(args.tp)

# monthly mean
z500_monthly = z500.resample(time='1MS').mean()
t2m_monthly  = t2m.resample(time='1MS').mean()
tp_monthly = tp.resample(time='1MS').sum()

# output paths
z500_out = args.z500.replace('daily', 'monthly')
t2m_out  = args.t2m.replace('daily', 'monthly')
tp_out   = args.tp.replace('daily', 'monthly')

# save
z500_monthly.to_netcdf(z500_out)
t2m_monthly.to_netcdf(t2m_out)
tp_monthly.to_netcdf(tp_out)