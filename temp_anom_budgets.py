from llc_uv_shift import llc_uv_shift
import numpy as np
import xarray as xr

def cal_anom(dataarray,  climarray, varname):
    # 提取每个时间点所属的月份 (1..12)
    months = dataarray['time'].dt.month  # (time,)
    # 用 month=months 做高级索引，得到与 time 对齐的气候态
    clim_for_each_time = climarray.sel(month=xr.DataArray(months, dims=['time']))
    # 现在 clim_for_each_time 的维度是 (time, k, face, j, i)
    var_anom = (dataarray - clim_for_each_time).rename(varname+'_anom')
    return var_anom

def clim2daily(dataarray, climarray):
    # 提取每个时间点所属的月份 (1..12)
    months = dataarray['time'].dt.month  # (time,)
    # 用 month=months 做高级索引，得到与 time 对齐的气候态
    clim_for_each_time = climarray.sel(month=xr.DataArray(months, dims=['time']))
    return clim_for_each_time


    
def cal_T_advlike(ADVx_TH,ADVy_TH,ADVr_TH, face_connections):
    DFxE_TH = ADVx_TH.astype('float64').fillna(0.).rename({'i_g': 'i'})
    DFyE_TH = ADVy_TH.astype('float64').fillna(0.).rename({'j_g': 'j'})
    DFr_TH  = ADVr_TH.astype('float64').fillna(0.).rename({'k_l': 'k'})
    
    DFxE_TH_right   = llc_uv_shift(DFxE_TH, DFyE_TH, face_connections, shift_x=-1, shift_y=0)
    DFyE_TH_up      = llc_uv_shift(DFyE_TH, DFxE_TH, face_connections, shift_x=0, shift_y=-1)
    

    adv_hConvH2_x = DFxE_TH_right - DFxE_TH
    adv_hConvH2_y = DFyE_TH_up    - DFyE_TH

    # Convergence of horizontal advection (degC m^3/s)
    adv_hConvH2 = -( adv_hConvH2_x  +  adv_hConvH2_y )

    DFr_TH_below = DFr_TH.shift(k=-1).fillna(0.)
    adv_vConvH2 = -(DFr_TH - DFr_TH_below )

    return adv_hConvH2, adv_vConvH2

def get_T_at_u(THETA, face_connections,ECCOgrid):
    from extract_boundarys import llc_shift
    Tleft = llc_shift( THETA, face_connections, shift_x=1, shift_y=0,method='isel' )
    vol_U = (ECCOgrid.rAw*ECCOgrid.drF*ECCOgrid.hFacW).transpose('k','face','j','i_g').rename({'i_g':'i'})
    vol = (ECCOgrid.rA*ECCOgrid.drF*ECCOgrid.hFacC).transpose('k','face','j','i')
    volleft=llc_shift( vol, face_connections, shift_x=1, shift_y=0,method='isel' )

    T_u = (Tleft*volleft/2 + THETA*vol/2)/vol_U
    return T_u

def get_T_at_v(THETA, face_connections,ECCOgrid):
    from extract_boundarys import llc_shift
    Tdown = llc_shift( THETA, face_connections, shift_x=0, shift_y=1,method='isel' )
    vol_V = (ECCOgrid.rAs*ECCOgrid.drF*ECCOgrid.hFacS).transpose('k','face','j_g','i').rename({'j_g':'j'})
    vol = (ECCOgrid.rA*ECCOgrid.drF*ECCOgrid.hFacC).transpose('k','face','j','i')
    voldown=llc_shift( vol, face_connections, shift_x=0, shift_y=1,method='isel' )

    T_v = (Tdown*voldown/2 + THETA*vol/2)/vol_V
    return T_v

def get_T_at_w(THETA, ECCOgrid, grid_type='k_l'):
    """
    Interpolate ECCO grid center variable (e.g. THETA) to w-grid (Zl),
    accounting for hFacC (partial cells) and skipping land columns
    using surface maskC for speed.
    """

    # === 1. Use surface maskC to skip land columns ===
    ocean_mask = ECCOgrid.maskC.isel(k=0).astype(bool)
    THETA = THETA.where(ocean_mask)

    # === 2. Fill vertical NaNs for partial cells ===
    THETA_filled = THETA.ffill(dim='k')

    # === 3. Effective layer thickness ===
    drF = ECCOgrid.drF
    hFacC = ECCOgrid.hFacC
    dr_eff = drF * hFacC

    # === 4. Interpolate to w-grid interfaces ===
    THETA_upper = THETA_filled.shift(k=1)
    dr_eff_upper = dr_eff.shift(k=1).fillna(0)

    num = (THETA_upper * dr_eff_upper / 2) + (THETA_filled * dr_eff / 2)
    denom = (dr_eff_upper / 2) + (dr_eff / 2)
    T_w = num / denom

    # === 5. Top interface = surface value ===
    T_w = T_w.where(T_w['k'] != 0, THETA_filled.isel(k=0))

    # === 6. Mask land columns ===
    T_w = T_w.where(ocean_mask)
    T_w = T_w.where(denom>0)
    
    # === 7. Add extra bottom interface if k_pl ===
    if grid_type == 'k_p1':
        # bottom cell 
        THETA_bottom = THETA_filled.isel(k=-1)
        dr_bottom = dr_eff.isel(k=-1)
    
        # bottom layer = bottom center
        bottom_layer = THETA_bottom.where(dr_bottom > 0)
        
        # extend k-dim
        bottom_layer = bottom_layer.expand_dims(dim='k')
        bottom_layer = bottom_layer.assign_coords({'k': [T_w['k'][-1]+1]})
        T_w = xr.concat([T_w, bottom_layer], dim='k')

    # === 8. Rename and reorder ===
    T_w = T_w.transpose('time','k','face','j', 'i')
    
    return T_w


    
def cal_GMlike_prime_transport(THETA, sTHETA, u, v, w, face_connections, ECCOgrid, rho_pot=None):
    u = u.reset_coords(drop=True)
    v = v.reset_coords(drop=True)
    w = w.reset_coords(drop=True)
    THETA = THETA.reset_coords(drop=True)
    sTHETA = sTHETA.reset_coords(drop=True)
    ECCOgrid = ECCOgrid.reset_coords(drop=True)
    
    Tu = get_T_at_u(sTHETA, face_connections,ECCOgrid)
    Tv = get_T_at_v(sTHETA, face_connections,ECCOgrid)
    Tw = get_T_at_w(THETA,ECCOgrid)
    
    
    utrans = (u*ECCOgrid.dyG*ECCOgrid.drF).rename({'i_g':'i'})*Tu
    vtrans = (v*ECCOgrid.dxG*ECCOgrid.drF).rename({'j_g':'j'})*Tv
    wtrans = (w*ECCOgrid.rA).rename({'k_l':'k'})*Tw
    
    # u/v/wtrans should be m^3/s
    # 将 NaN 替换为0, 使得陆地/无效区域对输运贡献=0
    utrans = utrans.fillna(0.0)
    vtrans = vtrans.fillna(0.0)
    wtrans = wtrans.fillna(0.0)
       
    utrans_right   = llc_uv_shift(utrans, vtrans, face_connections, shift_x=-1, shift_y=0)
    vtrans_up      = llc_uv_shift(vtrans, utrans, face_connections, shift_x=0, shift_y=-1)
    wtrans_below = wtrans.shift(k=-1).fillna(0.)
    
    
    adv_hConvH2_x = utrans_right - utrans
    adv_hConvH2_y = vtrans_up    - vtrans
    # Convergence of horizontal advection ( m^3/s)
    adv_hConvH2 = -( adv_hConvH2_x  +  adv_hConvH2_y )
        
    adv_vConvH2 = -(wtrans - wtrans_below )
            
    if rho_pot is None:
        return adv_hConvH2, adv_vConvH2
    else:
        return (adv_hConvH2*rho_pot),  (adv_vConvH2*rho_pot)

def cal_GMlike_prime_transport_ds(ds_block: xr.Dataset, face_connections, ECCOgrid, rho_pot=None) -> xr.Dataset:
    """
    包装函数，用于 xr.map_blocks.
    ds_block 是一个时间子块 (time=1)，包含 THETA, sTHETA, u, v, w.
    """
    # 调用你原来的函数
    ECCOgrid = ECCOgrid.chunk({dim: -1 for dim in ECCOgrid.dims})

    G_Hadv, G_Vadv = cal_GMlike_prime_transport(
        ds_block["THETA"], ds_block["sTHETA"],
        ds_block["u"], ds_block["v"], ds_block["w"],
        face_connections, ECCOgrid, rho_pot=rho_pot)
    
    # 返回一个 Dataset
    return xr.Dataset( {   "G_Hadv": G_Hadv,
                           "G_Vadv": G_Vadv,} )

def save_G_adv_surfacevolume_ds(ds_block: xr.Dataset, ECCOgrid, face_connections, var_prefix='MASS') -> xr.Dataset:
    """
    计算水平体积输运（m^3/s），用于水体质量收支。
    支持不同变量前缀（如 MASS, STAR）。
    """
    ECCOgrid = ECCOgrid.chunk({dim: -1 for dim in ECCOgrid.dims})

    ukey = f'UVEL{var_prefix}'
    vkey = f'VVEL{var_prefix}'
    #wkey = f'WVEL{var_prefix}'  # 未使用

    # 水平体积通量（m^3/s）
    utrans =(ds_block[ukey].astype('float64') * ECCOgrid.drF * ECCOgrid.dyG).reset_coords(drop=True)
    vtrans =(ds_block[vkey].astype('float64') * ECCOgrid.drF * ECCOgrid.dxG).reset_coords(drop=True)
    #wtrans = volcache.WVELMASS * ECCOgrid.rA

    # 清理无效区域
    utrans = utrans.fillna(0.0).rename({'i_g': 'i'})
    vtrans = vtrans.fillna(0.0).rename({'j_g': 'j'})
    #wtrans = wtrans.fillna(0.0).rename({'k_l': 'k'})

    # 跨面偏移（用于边界通量计算）
    utrans_right = llc_uv_shift(utrans, vtrans, face_connections, shift_x=-1, shift_y=0)
    vtrans_up    = llc_uv_shift(vtrans, utrans, face_connections, shift_x=0, shift_y=-1)

    return xr.Dataset({
        'utrans':       utrans,
        'utrans_right': utrans_right,
        'vtrans':       vtrans,
        'vtrans_up':    vtrans_up })


def cal_T_adv_shift(ds, face_connections):
    DFxE_TH = ds.ADVx_TH.astype('float64').fillna(0.).rename({'i_g': 'i'}).reset_coords(drop=True)
    DFyE_TH = ds.ADVy_TH.astype('float64').fillna(0.).rename({'j_g': 'j'}).reset_coords(drop=True)
    DFr_TH  = ds.ADVr_TH.astype('float64').fillna(0.).rename({'k_l': 'k'}).reset_coords(drop=True)
    del ds
    DFxE_TH_right   = llc_uv_shift(DFxE_TH, DFyE_TH, face_connections, shift_x=-1, shift_y=0)
    DFyE_TH_up      = llc_uv_shift(DFyE_TH, DFxE_TH, face_connections, shift_x=0, shift_y=-1)

    adv_hConvH2_x = DFxE_TH_right - DFxE_TH
    adv_hConvH2_y = DFyE_TH_up    - DFyE_TH

    # Convergence of horizontal advection (degC m^3/s)
    adv_hConvH2 = -( adv_hConvH2_x  +  adv_hConvH2_y )
    del DFxE_TH, DFyE_TH

    DFr_TH_below = DFr_TH.shift(k=-1, fill_value=0.)
    adv_vConvH2 = -(DFr_TH - DFr_TH_below )
    del DFr_TH
    return adv_hConvH2.compute(), adv_vConvH2.compute()
        

def cal_T_diffusion(ds, face_connections):
    DFxE_TH = ds.DFxE_TH.astype('float64').fillna(0.).rename({'i_g': 'i'}).reset_coords(drop=True)
    DFyE_TH = ds.DFyE_TH.astype('float64').fillna(0.).rename({'j_g': 'j'}).reset_coords(drop=True)
    DFrE_TH = ds.DFrE_TH.astype('float64').fillna(0.).rename({'k_l': 'k'}).reset_coords(drop=True)
    DFrI_TH = ds.DFrI_TH.astype('float64').fillna(0.).rename({'k_l': 'k'}).reset_coords(drop=True)
    DFr_TH  = DFrE_TH + DFrI_TH
    
    DFxE_TH_right   = llc_uv_shift(DFxE_TH, DFyE_TH, face_connections, shift_x=-1, shift_y=0)
    DFyE_TH_up      = llc_uv_shift(DFyE_TH, DFxE_TH, face_connections, shift_x=0, shift_y=-1)

    DFrI_TH_below = DFrI_TH.shift(k=-1).fillna(0.)
    DFrE_TH_below = DFrE_TH.shift(k=-1).fillna(0.)
    DFr_TH_below  = DFrE_TH_below + DFrI_TH_below
    
    dif_hConvH_x = DFxE_TH_right - DFxE_TH
    dif_hConvH_y = DFyE_TH_up    - DFyE_TH
    # Convergence of horizontal advection (degC m^3/s)
    dif_hConvH = -( dif_hConvH_x  +  dif_hConvH_y )
    dif_vConvH = -(DFr_TH - DFr_TH_below)

    return dif_hConvH, dif_vConvH
    
def cal_T_diffusion_ds(ds_block: xr.Dataset, face_connections) -> xr.Dataset:
    """
    用于 xr.map_blocks，ds_block 是一个时间子块 (time=1)，包含 DFxE_TH, DFyE_TH, DFrE_TH, DFrI_TH
    """
    # 调用原来的函数
    dif_hConvH, dif_vConvH = cal_T_diffusion(ds_block, face_connections)
    print(type(dif_vConvH))

    # 返回 Dataset
    return xr.Dataset({
        "dif_hConvH": dif_hConvH.astype("float64"),
        "dif_vConvH": dif_vConvH.astype("float64")
    })


def cal_T_forcing(hflux, ecco_grid, GEOFLX=None):
    import numpy as np
    hflux = hflux.reset_coords(drop=True)
    # Seawater density (kg/m^3)
    rhoconst = 1029.
    ## needed to convert surface mass fluxes to volume fluxes

    # Heat capacity (J/kg/K)
    c_p = 3994

    # Constants for surface heat penetration (from Table 2 of Paulson and Simpson, 1977)
    R = 0.62
    zeta1 = 0.6
    zeta2 = 20.0

    Z = ecco_grid.Z.compute()
    RF = np.concatenate([ecco_grid.Zp1.values[:-1],[np.nan]])
    q1 = R*np.exp(1.0/zeta1*RF[:-1]) + (1.0-R)*np.exp(1.0/zeta2*RF[:-1])
    q2 = R*np.exp(1.0/zeta1*RF[1:]) + (1.0-R)*np.exp(1.0/zeta2*RF[1:])
    # Correction for the 200m cutoff
    zCut = np.where(Z < -200)[0][0]
    q1[zCut:] = 0
    q2[zCut-1:] = 0
    # Create xarray data arrays
    q1 = xr.DataArray(q1,coords=[Z.k],dims=['k'])
    q2 = xr.DataArray(q2,coords=[Z.k],dims=['k'])

    ## Land masks
    # Make copy of hFacC
    mskC = ecco_grid.hFacC.copy(deep=True).compute()

    # Change all fractions (ocean) to 1. land = 0
    mskC.values[mskC.values>0] = 1

    # Shortwave flux below the surface (W/m^2)
    forcH_subsurf = ((q1*(mskC==1)-q2*(mskC.shift(k=-1)==1))*hflux.oceQsw.astype('float64') ).transpose('time','face','k','j','i')

    # Surface heat flux (W/m^2)
    forcH_surf = ((hflux.TFLUX.astype('float64') - (1-(q1[0]-q2[0]))*hflux.oceQsw.astype('float64') )*mskC[0]).transpose('time','face','j','i').assign_coords(k=0).expand_dims('k')

    # Full-depth sea surface forcing (W/m^2)
    forcH = xr.concat([forcH_surf,forcH_subsurf[:,:,1:]], dim='k').transpose('time','face','k','j','i')

    # Add geothermal heat flux to forcing field and convert from W/m^2 to degC/s m*3
    if GEOFLX is None:
        GEOFLX =0.
    vol = (ecco_grid.rA * ecco_grid.drF * ecco_grid.hFacC).transpose('face', 'k', 'j', 'i').compute()
    # Add geothermal heat flux to forcing field and convert from W/m^2 to degC/s
    G_forcing = ((forcH + GEOFLX)/(rhoconst*c_p))/(ecco_grid.hFacC*ecco_grid.drF)
    
    return G_forcing.reset_coords(drop=True).chunk({'k': len(Z)})


def add_bottom_layer_from_cell(T_k_l, T_k, k_p1, Zp1):
    """
    Add a bottom boundary layer to T_k_l, using the bottom cell value
    from T_k. The new layer corresponds to the lowest interface level (k_p1[-1]).
    
    - Removes the old Z coordinate (cell centers)
    - Assigns Zp1 (interface depths) instead
    - Dask-friendly (no computation until .compute()/.load())
    """

    # 1️⃣ 取 Tsnap_prime 最底层的温度
    T_bottom = T_k.isel(k=-1)  # shape: (time, face, j, i)

    # 2️⃣ 赋给一个新层（k_l 对应 k_p1 的最后一个界面）
    k_l_new = [k_p1[-1].item()]
    T_bottom_expanded = T_bottom.expand_dims(dim={"k_l": k_l_new})

    # 3️⃣ 拼接到底部
    T_extended = xr.concat( [T_k_l, T_bottom_expanded], dim="k_l"  )
    T_extended = T_extended.drop_vars(["Z","k"], errors="ignore").rename(k_l='k_p1')

    # 4️⃣ 更新 Z 坐标为对应的 Zp1
    T_extended = T_extended.assign_coords(Zp1=Zp1 ).chunk(chunks={"k_p1":-1})

    return T_extended





def get_T_at_w_old(THETA, face_connections,ECCOgrid):
    import numpy as np
    # 1. 获取深度和层厚度
    z_values = ECCOgrid.Z.values  # 中心点深度，形状 [k]，例如 [-5, -15, ..., -5000]
    zl_values= ECCOgrid.Zl.values 
    drF_values = ECCOgrid.drF.values  # 层厚度，形状 [k]，例如 [10, 20, ..., 200]

    # 3. 扩充 k 维度（仅顶部）
    k_values = THETA.k.values  # [0, 1, ..., 49]
    k_min = k_values[0]
    # 扩展深度和 k 坐标
    z_extended = np.concatenate([np.array([zl_values[0]]), z_values])  # [Zl[0], Z[0], ..., Z[-1]]
    k_extended = np.concatenate([np.array([k_min - 1]), k_values])  # [-1, 0, ..., 49]

    # 5. 创建顶部层
    top_layer = THETA.isel(k=0).drop_vars('k', errors='ignore')  # 移除 k 坐标
    # 6. 合并扩充层
    # 将 THETA 的 k 维度替换为 Z
    THETA_with_z = THETA.rename({'k': 'Z'}).assign_coords(Z=z_values).drop_vars('k', errors='ignore')
    top_layer_with_z = top_layer.expand_dims(Z=[zl_values[0]])  # 添加 Z 维度
    THETA_extended = xr.concat([top_layer_with_z, THETA_with_z], dim='Z')
    
    THETA_extended = THETA_extended.assign_coords(Z=z_extended)
    
    # 7. 插值到 w 网格
    T_w = THETA_extended.interp(Z=zl_values, method='linear')
    
      # 8. 重命名维度并分配坐标
    T_w = T_w.rename({'Z': 'k'}).assign_coords(k=('k', np.arange(len(zl_values))),Z=('k', zl_values))
    return T_w.transpose('time','face','k','j','i')



def get_T_at_w_old2(THETA, ECCOgrid):
    """
    Interpolate ECCO grid center temperature (THETA) to w-grid (Zl)
    Maintains dask chunks for parallel computation.
    Top layer (k=0) is taken directly from the original THETA.
    NaN values are forward-filled along k to avoid NaNs in T_w.
    """
    # Fill NaN along vertical (k) dimension
    THETA_filled = THETA.ffill(dim='k')

    # Center layer thickness and thickness of the layer above
    drF = ECCOgrid.drF
    drF_upper = drF.shift(k=1)

    # Temperature of the layer above
    THETA_upper = THETA.shift(k=1)

    # Vertical weighted average to w-grid
    T_w = (THETA_upper * drF_upper / 2 + THETA * drF / 2) / (drF_upper / 2 + drF / 2)
    
    # Top w-layer uses original k=0 temperature
    T_w = xr.where(T_w['k'] == 0, THETA.isel(k=0), T_w)
    
    # Replace NaNs with forward-filled values
    T_w = xr.where(T_w.isnull(), THETA_filled, T_w)

    # Transpose to standard order
    return T_w.transpose('time', 'face', 'k', 'j', 'i')