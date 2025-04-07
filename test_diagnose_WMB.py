import xarray as xr
import xgcm
import matplotlib.pyplot as plt
import numpy as np
import gsw
import glob

####  prepare data
ECCOgrid = xr.open_dataset('GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc').rename({'tile': 'face'})
rA    = ECCOgrid['rA']      # 水平面积 (m²)
drF   = ECCOgrid['drF']    # 垂直厚度 (m)
hFacC = ECCOgrid['hFacC'] # 有效厚度因子
maskC = ECCOgrid['maskC'] # 海洋掩码

tsfiles   = sorted(glob.glob('/dfs9/hfdrake_hpc/datasets/ECCOv4r4/ECCO_L4_TEMP_SALINITY_LLC0090GRID_DAILY_V4R4/OCEAN_TEMPERATURE_SALINITY_day_mean_*.nc'))
tadvfiles = sorted(glob.glob('/dfs9/hfdrake_hpc/datasets/ECCOv4r4/ECCO_L4_OCEAN_3D_TEMPERATURE_FLUX_LLC0090GRID_DAILY_V4R4/OCEAN_3D_TEMPERATURE_FLUX_day_mean_*.nc'))
sadvfiles = sorted(glob.glob('/dfs9/hfdrake_hpc/datasets/ECCOv4r4/ECCO_L4_OCEAN_3D_SALINITY_FLUX_LLC0090GRID_DAILY_V4R4/OCEAN_3D_SALINITY_FLUX_day_mean_*.nc'))
volfiles  = sorted(glob.glob('/dfs9/hfdrake_hpc/datasets/ECCOv4r4/ECCO_L4_OCEAN_3D_VOLUME_FLUX_LLC0090GRID_DAILY_V4R4/OCEAN_3D_VOLUME_FLUX_day_mean_*.nc'))
velfiles  = sorted(glob.glob('/dfs9/hfdrake_hpc/datasets/ECCOv4r4/ECCO_L4_OCEAN_VEL_LLC0090GRID_DAILY_V4R4/OCEAN_VELOCITY_day_mean_*.nc'))
sshfiles  = sorted(glob.glob('/dfs9/hfdrake_hpc/datasets/ECCOv4r4/ECCO_L4_SSH_LLC0090GRID_DAILY_V4R4/SEA_SURFACE_HEIGHT_day_mean_*.nc'))

tscache   = xr.open_mfdataset(tsfiles[0:10],   chunks={'time': 1, 'k': 50, 'tile': 13}  ).rename({'tile': 'face'})
velcache = xr.open_mfdataset(velfiles[0:10], chunks={'time': 1, 'k': 50, 'tile': 13}  ).rename({'tile': 'face'}).reset_coords(drop=True)
tbudgetcache = xr.open_mfdataset(tadvfiles[0:10], chunks={'time': 1, 'k': 50, 'tile': 13}  ).rename({'tile': 'face'}).reset_coords(drop=True)
sshcache = xr.open_mfdataset(sshfiles[0:10], chunks={'time': 1, 'k': 50, 'tile': 13}  ).rename({'tile': 'face'}).reset_coords(drop=True)

# a trick to make things work a bit faster
coords = tscache.coords.to_dataset().reset_coords()
tscache = tscache.reset_coords(drop=True)

theta = tscache['THETA']  # 温度，形状 (50, 13, 90, 90)
salt  = tscache['SALT' ]   # 盐度

#  potential density
p_ref = xr.zeros_like(theta)  # 参考压力固定为 0 dbar
rho_pot = jmd95(salt, theta, p_ref)  # 位势密度，单位 kg/m³

# define the connectivity between faces
face_connections = {'face':
                    {0: {'X':  ((12, 'Y', False), (3, 'X', False)),
                         'Y':  (None,             (1, 'Y', False))},
                     1: {'X':  ((11, 'Y', False), (4, 'X', False)),
                         'Y':  ((0, 'Y', False),  (2, 'Y', False))},
                     2: {'X':  ((10, 'Y', False), (5, 'X', False)),
                         'Y':  ((1, 'Y', False),  (6, 'X', False))},
                     3: {'X':  ((0, 'X', False),  (9, 'Y', False)),
                         'Y':  (None,             (4, 'Y', False))},
                     4: {'X':  ((1, 'X', False),  (8, 'Y', False)),
                         'Y':  ((3, 'Y', False),  (5, 'Y', False))},
                     5: {'X':  ((2, 'X', False),  (7, 'Y', False)),
                         'Y':  ((4, 'Y', False),  (6, 'Y', False))},
                     6: {'X':  ((2, 'Y', False),  (7, 'X', False)),
                         'Y':  ((5, 'Y', False),  (10, 'X', False))},
                     7: {'X':  ((6, 'X', False),  (8, 'X', False)),
                         'Y':  ((5, 'X', False),  (10, 'Y', False))},
                     8: {'X':  ((7, 'X', False),  (9, 'X', False)),
                         'Y':  ((4, 'X', False),  (11, 'Y', False))},
                     9: {'X':  ((8, 'X', False),  None),
                         'Y':  ((3, 'X', False),  (12, 'Y', False))},
                     10: {'X': ((6, 'Y', False),  (11, 'X', False)),
                          'Y': ((7, 'Y', False),  (2, 'X', False))},
                     11: {'X': ((10, 'X', False), (12, 'X', False)),
                          'Y': ((8, 'Y', False),  (1, 'X', False))},
                     12: {'X': ((11, 'X', False), None),
                          'Y': ((9, 'Y', False),  (0, 'X', False))}}}



################################################################################################################
####  diagnose the water mass budget of the water temperature > 29 degree
################################################################################################################

G_mat = cal_g_mat(tbudgetcache)
#### 

t2 = 29.25
t1 = 28.75

####  dt = 0.5 degree, t>=t + dt/2
threshold = t2

boundary, tile_or_face, interior, diff_left_boundary, diff_right_boundary, diff_up_boundary, diff_down_boundary, diff_above_boundary, diff_below_boundary = extract_boundary_gradient_method(tscache.THETA, face_connections, threshold)

G_mat_t2 = (g_mat*interior).compute()

G_mat_method2_t2 = cal_curve_tranport(tbudgetcache, diff_left_boundary,diff_right_boundary,diff_up_boundary,\
                                   diff_down_boundary,diff_above_boundary,diff_below_boundary)

####  t>=t - dt/2
threshold = t1

boundary, tile_or_face, interior, diff_left_boundary, diff_right_boundary, diff_up_boundary, diff_down_boundary, diff_above_boundary, diff_below_boundary = extract_boundary_gradient_method(tscache.THETA, face_connections, threshold)

G_mat_t1 = (g_mat*interior).compute()

G_mat_method2_t1 = cal_curve_tranport(tbudgetcache, diff_left_boundary,diff_right_boundary,diff_up_boundary,\
                                   diff_down_boundary,diff_above_boundary,diff_below_boundary)


#### t>= 29
threshold = (t2+t1)/2

boundary, tile_or_face, interior, diff_left_boundary, diff_right_boundary, diff_up_boundary, diff_down_boundary, diff_above_boundary, diff_below_boundary = extract_boundary_gradient_method(tscache.THETA, face_connections, threshold)


#### transformation term through two methods
g_mat = (G_mat_t2-G_mat_t1)/(t2-t1)
g_mat_method2 = (G_mat_method2_t2-G_mat_method2_t1)/(t2-t1)


#### mass tendency
dMdt = (dMassdt*interior).compute()




