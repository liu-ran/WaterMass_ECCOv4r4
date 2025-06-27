#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xarray as xr
import xgcm
import matplotlib.pyplot as plt
import numpy as np
import gsw
import glob
import sys
import calendar
from jmd95 import jmd95
from plot_tiles import plot_tiles
from extract_boundarys import *
from llc_uv_shift import llc_uv_shift
from cal_wmb_llc import *
import datetime
import os

# 从命令行获取年月，格式如199201
if len(sys.argv) < 2:
    raise ValueError("请提供 YYYYMM 参数")
yyyymm = sys.argv[1]
year = int(yyyymm[:4])
month = int(yyyymm[4:6])
if not (1992 <= year <= 2017 and 1 <= month <= 12):
    raise ValueError("年份应在1992-2017，月份应在1-12")

output_file = f'/pub/ranl24/datasets/g_trans_so_{yyyymm}.nc'
if os.path.exists(output_file):
    #print(f"{output_file} 已存在，正在删除旧文件...")
    #os.remove(output_file)
    print(f"{output_file} 已存在，跳过处理。")
    sys.exit(0)

# 计算该月的天数
days_in_month = calendar.monthrange(year, month)[1]
print(f"年份: {year}, 月份: {month}, 天数: {days_in_month}")

max_year, max_month = 2017, 12
current_date = datetime.date(year, month, 1)

# 判断是否有下个月数据
if year == max_year and month == max_month:
    has_next_month = False
    print("已经是最大年月，没有下个月数据")
else:
    has_next_month = True
    if month == 12:
        next_month_date = datetime.date(year + 1, 1, 1)
    else:
        next_month_date = datetime.date(year, month + 1, 1)

# 构造theta文件路径列表，包括本月所有天
dates_this_and_next = [f"{year}-{month:02d}-{day:02d}" for day in range(1, days_in_month + 1)]
# 如果有下个月1号数据，则加入
if has_next_month:
    dates_this_and_next.append(next_month_date.strftime("%Y-%m-%d"))

tsfiles = []
for date_str in dates_this_and_next:
    files = sorted(glob.glob(
        f'/dfs9/hfdrake_hpc/datasets/ECCOv4r4/ECCO_L4_TEMP_SALINITY_LLC0090GRID_DAILY_V4R4/OCEAN_TEMPERATURE_SALINITY_day_mean_{date_str}_ECCO_V4r4_native_llc0090.nc'))
    tsfiles.extend(files)

expected_file_count = days_in_month + (1 if has_next_month else 0)
if len(tsfiles) != expected_file_count:
    raise FileNotFoundError(f"应有 {expected_file_count} 个theta文件，找到 {len(tsfiles)} 个")

# g_trans 文件加载：本月和（如存在）下个月
g_trans_files = []
months_to_search = [yyyymm]
if has_next_month:
    months_to_search.append(next_month_date.strftime('%Y%m'))

for dt in months_to_search:
    files = sorted(glob.glob(f'/dfs9/hfdrake_hpc/datasets/ECCO_watermassbudget/g_transformation_{dt}*.nc'))
    g_trans_files.extend(files)

if len(g_trans_files) < 1:
    raise FileNotFoundError(f"未找到任何 g_trans 文件")

# 加载数据
theta = xr.open_mfdataset(tsfiles, chunks={'time': 1, 'k': 50, 'tile': 13}).rename({'tile': 'face'}).reset_coords(drop=True).THETA
g_transall = xr.open_mfdataset(g_trans_files)
#g_advneeds_all = xr.open_mfdataset(g_advneeds_files)


# 加载网格数据
ECCOgrid = xr.open_dataset('GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc').rename({'tile': 'face'})
# 区域掩码
region_mask = xr.open_dataarray('so_mask_llc90.nc')
maskC = ECCOgrid.maskC.copy()
region_mask = region_mask & maskC

# 计算 g_trans
more_or_less = '>'
tcenters = np.arange(-4, 28, 1)
#tcenters = np.arange(10, 36, 1)
dt = 0.1

print('start...')
g_advneeds_all = xr.Dataset({'utrans':g_transall.utrans,'utrans_right':g_transall.utrans_right, 
                                 'vtrans':g_transall.vtrans, 'vtrans_up':g_transall.vtrans_up})
if has_next_month == True:
    
    g_trans = cal_g_budgets_all(
        g_transall.Mass.isel(time=slice(0,days_in_month+1)).chunk(chunks={'time':1,'k':-1,'face':-1,'j':-1,'i':-1}),
        g_transall.g_fresh.isel(time=slice(0,days_in_month)).chunk(chunks={'time':1,'face':-1,'j':-1,'i':-1}),
        g_advneeds_all.isel(time=slice(0,days_in_month)).chunk(chunks={'time':1,'k':-1,'face':-1,'j':-1,'i':-1}),
        g_transall.g_mix.isel(time=slice(0,days_in_month)).chunk(chunks={'time':1,'k':-1,'face':-1,'j':-1,'i':-1}),
        g_transall.g_heat.isel(time=slice(0,days_in_month)).chunk(chunks={'time':1,'k':-1,'face':-1,'j':-1,'i':-1}),
        tcenters,
        dt,
        theta.isel(time=slice(0,days_in_month+1)),
        more_or_less,
        maskregion=region_mask,
        g_gm=g_transall.g_gm.isel(time=slice(0,days_in_month)), )
else:
    g_trans = cal_g_budgets_all(
        g_transall.Mass.isel(time=slice(0,days_in_month)).chunk(chunks={'time':1,'k':-1,'face':-1,'j':-1,'i':-1}),
        g_transall.g_fresh.isel(time=slice(0,days_in_month-1)).chunk(chunks={'time':1,'face':-1,'j':-1,'i':-1}),
        g_advneeds_all.isel(time=slice(0,days_in_month-1)).chunk(chunks={'time':1,'k':-1,'face':-1,'j':-1,'i':-1}),
        g_transall.g_mix.isel(time=slice(0,days_in_month-1)).chunk(chunks={'time':1,'k':-1,'face':-1,'j':-1,'i':-1}),
        g_transall.g_heat.isel(time=slice(0,days_in_month-1)).chunk(chunks={'time':1,'k':-1,'face':-1,'j':-1,'i':-1}),
        tcenters,
        dt,
        theta.isel(time=slice(0,days_in_month)),
        more_or_less,
        maskregion=region_mask,
        g_gm=g_transall.g_gm.isel(time=slice(0,days_in_month-1)), )  
# 保存结果
output_file = f'/pub/ranl24/datasets/g_trans_so_{yyyymm}.nc'

g_trans.to_netcdf(output_file)
print(f"结果已保存至 {output_file}")
