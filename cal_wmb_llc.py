import xarray as xr
from llc_uv_shift import llc_uv_shift
from extract_boundarys import *
import numpy as np

def cal_G_diffusion(ds, face_connections, rho_pot=None, boundary_diffs=None):
    DFxE_TH = ds.DFxE_TH.astype('float64').fillna(0.).rename({'i_g': 'i'})
    DFyE_TH = ds.DFyE_TH.astype('float64').fillna(0.).rename({'j_g': 'j'})
    DFrE_TH = ds.DFrE_TH.astype('float64').fillna(0.).rename({'k_l': 'k'})
    DFrI_TH = ds.DFrI_TH.astype('float64').fillna(0.).rename({'k_l': 'k'})
    DFr_TH  = DFrE_TH + DFrI_TH
    
    DFxE_TH_right   = llc_uv_shift(DFxE_TH, DFyE_TH, face_connections, shift_x=-1, shift_y=0)
    DFyE_TH_up      = llc_uv_shift(DFyE_TH, DFxE_TH, face_connections, shift_x=0, shift_y=-1)
    
    DFrI_TH_below = DFrI_TH.shift(k=-1, fill_value=0.)
    DFrE_TH_below = DFrE_TH.shift(k=-1, fill_value=0.)
    DFr_TH_below  = DFrE_TH_below + DFrI_TH_below

    if boundary_diffs is None:
        dif_hConvH_x = DFxE_TH_right - DFxE_TH
        dif_hConvH_y = DFyE_TH_up    - DFyE_TH
        # Convergence of horizontal advection (degC m^3/s)
        dif_hConvH = -( dif_hConvH_x  +  dif_hConvH_y )
        dif_vConvH = -(DFr_TH - DFr_TH_below)

        G_diffusion = (dif_hConvH + dif_vConvH).compute()

    else:
        #from extract_boundarys import extract_boundary_use_mask
        #boundary_test, tile_or_face, interior_test, boundary_diffs = extract_boundary_use_mask(region_mask, face_connections)
        G_diffusion = -( DFxE_TH      *boundary_diffs.sel(direction='left')  + \
                         DFxE_TH_right*boundary_diffs.sel(direction='right') + \
                         DFyE_TH      *boundary_diffs.sel(direction='down')  + \
                         DFyE_TH_up   *boundary_diffs.sel(direction='up')    + \
                         DFr_TH       *boundary_diffs.sel(direction='above') + \
                         DFr_TH_below *boundary_diffs.sel(direction='below') )
    if rho_pot is None:
        return G_diffusion
    else:
        return (G_diffusion*rho_pot).compute()    



def cal_G_forcing(hflux, ecco_grid, GEOFLX=None, rho_pot=None):
    import numpy as np
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

    G_forcing = ((forcH + GEOFLX.astype('float64'))/(rhoconst*c_p))/(ecco_grid.hFacC*ecco_grid.drF)*vol
    
    if rho_pot is None:
        return G_forcing
    else:
        return (G_forcing*rho_pot).compute()


def cal_G_transformation(ds_diff, hflux, THETA, ecco_grid, face_connections, rho_pot, velGM=None):
    vol = (ecco_grid.rA * ecco_grid.drF * ecco_grid.hFacC).transpose('face', 'k', 'j', 'i').compute()

    G_diffusion = cal_G_dif_shift(ds_diff, face_connections) * rho_pot
    G_forcing   = cal_G_forcing(hflux, ecco_grid, vol) * rho_pot

    if velGM is not None:
        G_GM = cal_GM_transport(THETA, velGM, face_connections, ecco_grid) * rho_pot
        return G_diffusion, G_forcing, G_GM
    else:
        return G_diffusion, G_forcing


def mask3d(data, tracer_threshold, more_or_less ):
    #
    if more_or_less=='>':
        mask = (data >= tracer_threshold) & (~data.isnull())
    elif more_or_less=='<':
        mask = (data <= tracer_threshold) & (~data.isnull())
    #
    return mask

def cal_G_adv_shift(volcache, ECCOgrid, face_connections, rho_pot=None, boundary_diffs=None):
    ### note!!! there is no vertical transport in the watermass budget!!!
    utrans = volcache.UVELMASS.astype('float64') * ECCOgrid.drF * ECCOgrid.dyG
    vtrans = volcache.VVELMASS.astype('float64') * ECCOgrid.drF * ECCOgrid.dxG 
    #wtrans = volcache.WVELMASS * ECCOgrid.rA
    
    # u/v/wtrans should be m^3/s
    # 将 NaN 替换为0, 使得陆地/无效区域对输运贡献=0
    utrans = utrans.fillna(0.0).rename({'i_g': 'i'})
    vtrans = vtrans.fillna(0.0).rename({'j_g': 'j'})
    #wtrans = wtrans.fillna(0.0).rename({'k_l': 'k'})
    
    utrans_right   = llc_uv_shift(utrans, vtrans, face_connections, shift_x=-1, shift_y=0)
    vtrans_up      = llc_uv_shift(vtrans, utrans, face_connections, shift_x=0, shift_y=-1)

    if boundary_diffs is None:
        adv_hConvH2_x = utrans_right - utrans
        adv_hConvH2_y = vtrans_up    - vtrans
        # Convergence of horizontal advection ( m^3/s)
        adv_hConvH2 = -( adv_hConvH2_x  +  adv_hConvH2_y )
        #wtrans_below = wtrans.shift(k=-1, fill_value=0.)
        #adv_vConvH2 = -(wtrans - wtrans_below )
        #G_advection2 = (adv_hConvH2 + adv_vConvH2).compute()
        G_advection2 = adv_hConvH2.compute()
        
    else:
        G_advection2 = -( utrans      *boundary_diffs.sel(direction='left')  + \
                          utrans_right*boundary_diffs.sel(direction='right') + \
                          vtrans      *boundary_diffs.sel(direction='down')  + \
                          vtrans_up   *boundary_diffs.sel(direction='up')    ).compute()
        
    assert G_advection2.dtype == 'float64', "utrans is not float64!"

    if rho_pot is None:
        return G_advection2
    else:
        return (G_advection2*rho_pot).compute()
def save_G_adv_surfacevolume(volcache, ECCOgrid, face_connections, var_prefix='MASS'):
    """
    计算水平体积输运（m^3/s），用于水体质量收支。
    支持不同变量前缀（如 MASS, STAR）。
    """
    ukey = f'UVEL{var_prefix}'
    vkey = f'VVEL{var_prefix}'
    #wkey = f'WVEL{var_prefix}'  # 未使用

    # 水平体积通量（m^3/s）
    utrans = volcache[ukey].astype('float64') * ECCOgrid.drF * ECCOgrid.dyG
    vtrans = volcache[vkey].astype('float64') * ECCOgrid.drF * ECCOgrid.dxG
    #wtrans = volcache.WVELMASS * ECCOgrid.rA

    # 清理无效区域
    utrans = utrans.fillna(0.0).rename({'i_g': 'i'})
    vtrans = vtrans.fillna(0.0).rename({'j_g': 'j'})
    #wtrans = wtrans.fillna(0.0).rename({'k_l': 'k'})

    # 跨面偏移（用于边界通量计算）
    utrans_right = llc_uv_shift(utrans, vtrans, face_connections, shift_x=-1, shift_y=0)
    vtrans_up    = llc_uv_shift(vtrans, utrans, face_connections, shift_x=0, shift_y=-1)

    return xr.Dataset({
        'utrans':       utrans.compute(),
        'utrans_right': utrans_right.compute(),
        'vtrans':       vtrans.compute(),
        'vtrans_up':    vtrans_up.compute()
    })

def save_G_adv_surfacevolume_old(volcache, ECCOgrid, face_connections):
    ### note!!! there is no vertical transport in the watermass budget!!!
    utrans = volcache.UVELMASS.astype('float64') * ECCOgrid.drF * ECCOgrid.dyG 
    vtrans = volcache.VVELMASS.astype('float64') * ECCOgrid.drF * ECCOgrid.dxG 
    #wtrans = volcache.WVELMASS * ECCOgrid.rA
    
    # u/v/wtrans should be m^3/s
    # 将 NaN 替换为0, 使得陆地/无效区域对输运贡献=0
    utrans = utrans.fillna(0.0).rename({'i_g': 'i'})
    vtrans = vtrans.fillna(0.0).rename({'j_g': 'j'})
    #wtrans = wtrans.fillna(0.0).rename({'k_l': 'k'})
    
    utrans_right   = llc_uv_shift(utrans, vtrans, face_connections, shift_x=-1, shift_y=0)
    vtrans_up      = llc_uv_shift(vtrans, utrans, face_connections, shift_x=0, shift_y=-1)

    #if rho_pot is None:
    #    return xr.Dataset({'utrans':utrans.compute(),'utrans_right': utrans_right.compute(), 'vtrans': vtrans.compute(), 'vtrans_up':vtrans_up.compute()})
    #else:
    return xr.Dataset({'utrans':utrans.compute(),'utrans_right': utrans_right.compute(), 'vtrans': vtrans.compute(), 'vtrans_up':vtrans_up.compute()})

        
def cal_G_adv_surfaceIntegrate(ds, boundary_diffs, rho_pot=None):
    G_advection2 = -(  ds.utrans      *boundary_diffs.sel(direction='left')  + \
                       ds.utrans_right*boundary_diffs.sel(direction='right') + \
                       ds.vtrans      *boundary_diffs.sel(direction='down')  + \
                       ds.vtrans_up   *boundary_diffs.sel(direction='up')    )
    
    if rho_pot is None:
        return G_advection2.compute()
    else:
        return (G_advection2*rho_pot).compute()
def cal_G_adv_surfaceIntegrate2(ds, boundary_diffs, rho_pot=None):
    # 先对各方向做乘积，同时利用 where 快速剔除零点
    terms = []
    for flux_name, bd_direction in zip(
        ['utrans', 'utrans_right', 'vtrans', 'vtrans_up'],
        ['left', 'right', 'down', 'up']
    ):
        bd_diff = boundary_diffs.sel(direction=bd_direction)
        # 只在有边界diff的地方做乘积（mask作用）
        term = (ds[flux_name] * bd_diff).where(bd_diff != 0, other=0.)
        terms.append(term)

    # 合并四个方向结果
    G_advection2 = -(terms[0] + terms[1] + terms[2] + terms[3])

    # 如果有 rho_pot，则直接乘上
    if rho_pot is not None:
        G_advection2 = G_advection2 * rho_pot

    return G_advection2.compute()
     
        
def get_T_at_u(THETA, face_connections,ECCOgrid):
    from extract_boundarys import llc_shift
    Tleft = llc_shift( THETA, face_connections, shift_x=1, shift_y=0,method='isel' )
    vol_U = (ECCOgrid.rAw*ECCOgrid.drF*ECCOgrid.hFacW).transpose('face','k','j','i_g').rename({'i_g':'i'}).compute()
    vol = (ECCOgrid.rA*ECCOgrid.drF*ECCOgrid.hFacC).transpose('face','k','j','i').compute()
    volleft=llc_shift( vol, face_connections, shift_x=1, shift_y=0,method='isel' )

    T_u = (Tleft*volleft/2 + THETA*vol/2)/vol_U
    return T_u.compute()

def get_T_at_v(THETA, face_connections,ECCOgrid):
    from extract_boundarys import llc_shift
    Tdown = llc_shift( THETA, face_connections, shift_x=0, shift_y=1,method='isel' )
    vol_V = (ECCOgrid.rAs*ECCOgrid.drF*ECCOgrid.hFacS).transpose('face','k','j_g','i').rename({'j_g':'j'}).compute()
    vol = (ECCOgrid.rA*ECCOgrid.drF*ECCOgrid.hFacC).transpose('face','k','j','i').compute()
    voldown=llc_shift( vol, face_connections, shift_x=0, shift_y=1,method='isel' )

    T_v = (Tdown*voldown/2 + THETA*vol/2)/vol_V
    return T_v.compute()

def get_T_at_w(THETA, face_connections,ECCOgrid):
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
    return T_w.compute()


def cal_GM_transport(THETA, velGM, face_connections, ECCOgrid, rho_pot=None, boundary_diffs=None):
    
    Tu = get_T_at_u(THETA, face_connections,ECCOgrid)
    Tv = get_T_at_v(THETA, face_connections,ECCOgrid)
    Tw = get_T_at_w(THETA, face_connections,ECCOgrid)
    
    u = velGM.UVELSTAR.astype('float64')
    v = velGM.VVELSTAR.astype('float64')
    w = velGM.WVELSTAR.astype('float64')
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
    wtrans_below = wtrans.shift(k=-1, fill_value=0.)
    
    if boundary_diffs is None:
        adv_hConvH2_x = utrans_right - utrans
        adv_hConvH2_y = vtrans_up    - vtrans
        # Convergence of horizontal advection ( m^3/s)
        adv_hConvH2 = -( adv_hConvH2_x  +  adv_hConvH2_y )
        
        adv_vConvH2 = -(wtrans - wtrans_below )
    
        G_GM_adv = (adv_hConvH2 + adv_vConvH2).compute()
        
    else:
        G_GM_adv = -(     utrans      *boundary_diffs.sel(direction='left')  + \
                          utrans_right*boundary_diffs.sel(direction='right') + \
                          vtrans      *boundary_diffs.sel(direction='down')  + \
                          vtrans_up   *boundary_diffs.sel(direction='up')    + \
                          wtrans      *boundary_diffs.sel(direction='above') + \
                          wtrans_below*boundary_diffs.sel(direction='below') ).compute()
    if rho_pot is None:
        return G_GM_adv
    else:
        return (G_GM_adv*rho_pot).compute()   


def cal_GMlike_transport(THETA, sTHETA, u, v, w, face_connections, ECCOgrid, rho_pot=None, boundary_diffs=None):
    
    Tu = get_T_at_u(sTHETA, face_connections,ECCOgrid)
    Tv = get_T_at_v(sTHETA, face_connections,ECCOgrid)
    Tw = get_T_at_w(THETA, face_connections,ECCOgrid)
    
    #u = velGM.UVELSTAR.astype('float64')
    #v = velGM.VVELSTAR.astype('float64')
    #w = velGM.WVELSTAR.astype('float64')
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
    wtrans_below = wtrans.shift(k=-1, fill_value=0.)
    
    if boundary_diffs is None:
        adv_hConvH2_x = utrans_right - utrans
        adv_hConvH2_y = vtrans_up    - vtrans
        # Convergence of horizontal advection ( m^3/s)
        adv_hConvH2 = -( adv_hConvH2_x  +  adv_hConvH2_y )
        
        adv_vConvH2 = -(wtrans - wtrans_below )
    
        G_GM_adv = (adv_hConvH2 + adv_vConvH2).compute()
        
    else:
        G_GM_adv = -(     utrans      *boundary_diffs.sel(direction='left')  + \
                          utrans_right*boundary_diffs.sel(direction='right') + \
                          vtrans      *boundary_diffs.sel(direction='down')  + \
                          vtrans_up   *boundary_diffs.sel(direction='up')    + \
                          wtrans      *boundary_diffs.sel(direction='above') + \
                          wtrans_below*boundary_diffs.sel(direction='below') ).compute()
    if rho_pot is None:
        return G_GM_adv
    else:
        return (G_GM_adv*rho_pot).compute()  
        

def cal_g_budgets_all(mass, freshwater, adv, g_mix, g_heat, tcenters, dt, THETA, more_or_less,maskregion=None,g_gm=None):
    import numpy as np
    # note!!! we want Mass data with 2 more days since it use center difference.
    rhoconst = 1029.
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
    
    tright = tcenters + dt/2.
    tleft  = tcenters - dt/2.
    g_gm_all   =[]
    g_mix_all  =[]
    g_heat_all =[]
    g_tend_all =[]
    g_fresh_all=[]
    g_adv_all  = []

    if maskregion is not None:
        boundary_region,  tile_or_face, interior_region, boundary_diffs_region = extract_boundary_use_mask(maskregion, face_connections)

    for i in range( len(tright) ):
        print(tcenters[i])
        mask_right = mask3d(THETA.isel(time=slice(0,-1)), tright[i], more_or_less )
        mask_left  = mask3d(THETA.isel(time=slice(0,-1)), tleft[i] , more_or_less )
        # 如果指定了 maskregion, 就 AND 上去:
        if maskregion is not None:
            mask_right = mask_right & maskregion
            mask_left  = mask_left  & maskregion
        #
        g_mix_right = (g_mix*mask_right).sum(['face','k','j','i']).compute()
        g_mix_left  = (g_mix*mask_left ).sum(['face','k','j','i']).compute()
        g_mix_one = -(g_mix_right - g_mix_left) / (tright[i]-tleft[i])
        g_mix_one = g_mix_one.expand_dims({'tcenters': [tcenters[i]]})
        g_mix_all.append(g_mix_one)
        #
        #
        g_gm_right   = (g_gm*mask_right).sum(['face','k','j','i']).compute()
        g_gm_left    = (g_gm*mask_left ).sum(['face','k','j','i']).compute()
        g_gm_one    = -(g_gm_right - g_gm_left) / (tright[i]-tleft[i])
        g_gm_one      = g_gm_one.expand_dims({'tcenters': [tcenters[i]]})
        g_gm_all.append(g_gm_one)
        #
        g_heat_right = (g_heat*mask_right).sum(['face','k','j','i']).compute()
        g_heat_left  = (g_heat*mask_left ).sum(['face','k','j','i']).compute()
        g_heat_one = -(g_heat_right - g_heat_left) / (tright[i]-tleft[i])
        g_heat_one = g_heat_one.expand_dims({'tcenters': [tcenters[i]]})
        g_heat_all.append(g_heat_one)
        #
        maskTracer  = mask3d(THETA, tcenters[i] , more_or_less )    
        if maskregion is not None:
            mask = maskTracer & maskregion
        else:
            mask = maskTracer
        M = (mass*mask).sum(['face','k','j','i']).compute()
        dmdt = (M[1:].values - M[0:-1].values)/86400.
        dmdt = xr.DataArray(dmdt,dims=['time'],coords={'time': M.time[0:-1]})        
        #dmdt = (M.isel( time=slice(2,None) ) - M.isel( time=slice(0,-2) ))/2./86400.         # here M is 1-dim with time
        #dmdt = ( M.shift(time=-1, fill_value=np.nan) - M.shift(time=1, fill_value=np.nan) )/2./86400.
        dmdt = dmdt.expand_dims({'tcenters': [tcenters[i]]})
        g_tend_all.append(dmdt)
        #
        tendency_mass_surf = (mask.isel(k=0)*freshwater).astype("float64").sum(['face','j','i']).compute()
        g_fresh_all.append(tendency_mass_surf.expand_dims({'tcenters': [tcenters[i]]}))
        #
        #
        maskTracer  = mask3d(THETA.isel(time=slice(0,-1)), tcenters[i] , more_or_less )    
        if maskregion is not None:
            mask = maskTracer & maskregion
        else:
            mask = maskTracer
        boundary,    tile_or_face, interior, boundary_diffs = extract_boundary_use_mask(mask, face_connections)
        boundary_tracer, tile_or_face, interior_tracer, boundary_diffs_tracer  = extract_boundary_use_mask(maskTracer, face_connections)
        boundarymask_adv = (boundary_diffs_region != 0) & (boundary_diffs != 0)
        region_and_tracer_mask = boundarymask_adv & (boundary_diffs_tracer!=0)
        boundarymask_adv = boundarymask_adv.where(~region_and_tracer_mask, other=False) # region boundary without tracer
        boundary_adv = boundary_diffs.where(boundarymask_adv, other=0.).compute()
        #
        #g_adv = (adv*mask).sum(['face','k','j','i']).compute()
        g_adv = cal_G_adv_surfaceIntegrate2(adv, boundary_adv, rho_pot=rhoconst).sum(['face','k','j','i']).compute()
        #g_adv = cal_G_adv_surfaceIntegrate( adv, boundary_adv, rho_pot=adv.rho_pot).sum(['face','k','j','i']).compute()
        g_adv_all.append( g_adv.expand_dims({'tcenters': [tcenters[i]]}) )
        #
        
    # 转换列表为 xarray DataArray
    g_mix_da   = xr.concat(g_mix_all, dim='tcenters').transpose('tcenters', 'time')
    g_heat_da  = xr.concat(g_heat_all, dim='tcenters').transpose('tcenters', 'time')
    g_tend_da  = xr.concat(g_tend_all, dim='tcenters').transpose('tcenters', 'time')
    g_fresh_da = xr.concat(g_fresh_all, dim='tcenters').transpose('tcenters', 'time')
    g_adv_da   = xr.concat(g_adv_all, dim='tcenters').transpose('tcenters', 'time')
    g_gm_da    = xr.concat(g_gm_all, dim='tcenters').transpose('tcenters', 'time')
        
    return xr.Dataset({'g_tend':g_tend_da,'g_mix': g_mix_da, 'g_heat': g_heat_da, 'g_salt':g_fresh_da, 'g_adv':g_adv_da, 'g_gm':g_gm_da})



def compute_flux_difference(var: xr.DataArray,
                            mask_left: xr.DataArray,
                            mask_right: xr.DataArray,
                            tleft: float,
                            tright: float,
                            tcenter: float,
                            dims=['face', 'k', 'j', 'i']) -> xr.DataArray:
    """
    Compute flux difference across left/right masks and normalize by time interval.

    Parameters:
    - var: xarray.DataArray, the flux variable (e.g., g_mix_h, g_heat)
    - mask_left, mask_right: xarray.DataArray, boolean masks
    - tleft, tright: float, time boundaries
    - tcenter: float, center time for output coordinate
    - dims: list of str, dimensions to sum over

    Returns:
    - xarray.DataArray with dimensions ('tcenters','time')
    """
    right_sum = (var * mask_right).sum(dims)
    left_sum  = (var * mask_left ).sum(dims)
    flux = -(right_sum - left_sum) / (tright - tleft)
    return flux.expand_dims({'tcenters': [tcenter]})


def cal_wmb_anomaly_monthly(ds_budgets, adv, gm, tcenters, dt, THETA, more_or_less, maskregion=None):
    """
    Calculate monthly water mass budget anomaly terms.

    Parameters
    ----------
    ds_budgets : xr.Dataset
        ECCO budget variables (mass, g_mix_h, g_heat, g_adv, etc.)
    tcenters : array-like
        Central times for averaging
    dt : float
        Time interval (same unit as tcenters)
    THETA : xr.DataArray
        Potential temperature field
    more_or_less : str
        Threshold operator for mask3d
    maskregion : xr.DataArray, optional
        Regional mask to apply

    Returns
    -------
    xr.Dataset
        Dataset with all anomaly terms
    """
    rhoconst = 1029.0

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
    
    # 时间边界
    tcenters = np.array(tcenters)
    tright = tcenters + dt/2.
    tleft  = tcenters - dt/2.
    
    # 保存结果的容器
    results = {name: [] for name in [
        "g_mix_h","g_mix_v","g_UpTb_h","g_UpTb_v","g_UbTp_h","g_UbTp_v","g_uptp_h","g_uptp_v","g_UpTp_h","g_UpTp_v",
        "g_heat","g_tend","g_salt","g_adv","g_gm","Mass"
    ]}
    
    # 区域边界差分
    if maskregion is not None:
        _, _, _, boundary_diffs_region = extract_boundary_use_mask(maskregion, face_connections)
    
    for i in range(len(tright)):
        print(f"Processing tcenter = {tcenters[i]}")
        
        # === mask: 左右边界
        mask_right = mask3d(THETA.isel(time=slice(0,-1)), tright[i], more_or_less)
        mask_left  = mask3d(THETA.isel(time=slice(0,-1)), tleft[i] , more_or_less)
        if maskregion is not None:
            mask_right &= maskregion
            mask_left  &= maskregion
        
        # === 混合 & 热通量项
        for varname in ["g_mix_h","g_mix_v","g_heat","g_UpTb_h","g_UpTb_v","g_UbTp_h","g_UbTp_v","g_uptp_h","g_uptp_v","g_UpTp_h","g_UpTp_v"]:
            cache = compute_flux_difference(ds_budgets[varname], mask_left, mask_right,
                                            tleft[i], tright[i], tcenters[i])
            results[varname].append(cache)
        
        # === 水体质量倾向项 (dM/dt)
        maskTracer = mask3d(THETA, tcenters[i], more_or_less)
        mask = maskTracer & maskregion if maskregion is not None else maskTracer
        M = (ds_budgets["Mass"]*mask.rename({"time":"time_mass"})).sum(['face','k','j','i'])
        results["Mass"].append( M.expand_dims({'tcenters':[tcenters[i]]}) )
        
        # 用真实时间差计算 dM/dt
        #dt_seconds = (M.time_mass[1:].values - M.time_mass[:-1].values) / np.timedelta64(1,'s')
        dt_seconds = 86400.
        dmdt = (M[1:].values - M[:-1].values) / dt_seconds
        dmdt = xr.DataArray(dmdt, dims=['time'], coords={'time': ds_budgets.time})
        results["g_tend"].append(dmdt.expand_dims({'tcenters':[tcenters[i]]}))
        
        # === 淡水通量 (surface fw)
        tendency_mass_surf = (mask.isel(k=0)*ds_budgets["g_fw"]).astype("float64").sum(['face','j','i'])
        results["g_salt"].append(tendency_mass_surf.expand_dims({'tcenters': [tcenters[i]]}))
        
        
        # === 平流 & GM 项
        if maskregion is None:
        # 无区域定义，水团无平流输运
            g_adv = xr.zeros_like(tendency_mass_surf)
            results["g_adv"].append(g_adv.expand_dims({'tcenters':[tcenters[i]]}))
            g_gm = xr.zeros_like(tendency_mass_surf)
            results["g_gm"].append(g_gm.expand_dims({'tcenters':[tcenters[i]]}))

        else:
            maskTracer2 = mask3d(THETA.isel(time=slice(0,-1)), tcenters[i], more_or_less)
            mask2 = maskTracer2 & maskregion
            _, _, _, boundary_diffs = extract_boundary_use_mask(mask2, face_connections)
            _, _, _, boundary_diffs_tracer = extract_boundary_use_mask(maskTracer2, face_connections)

            # 构造边界
            boundarymask_adv = (boundary_diffs_region != 0) & (boundary_diffs != 0)
            region_and_tracer_mask = boundarymask_adv & (boundary_diffs_tracer != 0)
            boundarymask_adv = boundarymask_adv.where(~region_and_tracer_mask, other=False)
            boundary_adv = boundary_diffs.where(boundarymask_adv, other=0.)

            g_adv = cal_G_adv_surfaceIntegrate2(adv, boundary_adv, rho_pot=None).sum(['face','k','j','i'])
            results["g_adv"].append(g_adv.expand_dims({'tcenters':[tcenters[i]]}))
            g_gm = cal_G_adv_surfaceIntegrate2(gm, boundary_adv, rho_pot=None).sum(['face','k','j','i'])
            results["g_gm"].append(g_gm.expand_dims({'tcenters':[tcenters[i]]}))

#        maskTracer2 = mask3d(THETA.isel(time=slice(0,-1)), tcenters[i], more_or_less)
#        mask2 = maskTracer2 & maskregion if maskregion is not None else maskTracer2
#        _, _, _, boundary_diffs = extract_boundary_use_mask(mask2, face_connections)
#        _, _, _, boundary_diffs_tracer = extract_boundary_use_mask(maskTracer2, face_connections)
#        
#        # 构造边界
#        boundarymask_adv = (boundary_diffs_region != 0) & (boundary_diffs != 0)
#        region_and_tracer_mask = boundarymask_adv & (boundary_diffs_tracer!=0)
#        boundarymask_adv = boundarymask_adv.where(~region_and_tracer_mask, other=False)
#        boundary_adv = boundary_diffs.where(boundarymask_adv, other=0.).compute()
#        
#        g_adv = cal_G_adv_surfaceIntegrate2(adv, boundary_adv, rho_pot=None) \
#                    .sum(['face','k','j','i']).compute()
#        results["g_adv"].append(g_adv.expand_dims({'tcenters':[tcenters[i]]}))
#        
#        g_gm = cal_G_adv_surfaceIntegrate2(gm, boundary_adv, rho_pot=None) \
#                    .sum(['face','k','j','i']).compute()
#        results["g_gm"].append(g_gm.expand_dims({'tcenters':[tcenters[i]]}))
        
    # 拼接所有变量
    return xr.Dataset({k: xr.concat(v, dim="tcenters") for k,v in results.items()})


def vertical_pairwise_avg(data, tcenters, dtcenters, dim="tcenter", sum_dims=("face", "j", "i")):
    """
    Compute vertically integrated pairwise averages along `tcenter`, 
    handling the first layer separately (single dtcenters[0]).
    
    Parameters
    ----------
    data : xr.DataArray
        Input data with dimensions (..., tcenter, face, j, i).
    tcenters : array-like
        Midpoint positions along the tcenter dimension.
    dtcenters : array-like
        Thickness or spacing corresponding to each tcenter.
    dim : str, default "tcenter"
        Name of the vertical coordinate in the input data.
    sum_dims : tuple of str, default ("face", "j", "i")
        Dimensions to sum over before averaging.
        
    Returns
    -------
    xr.DataArray
        Output DataArray with dimension (time, itcenter),
        representing pairwise layer averages divided by 
        (dt[i] + dt[i-1]) for i>=1, and /dt[0] for i=0.
        Includes coordinate `tcenter_mid` (midpoints for plotting).
    """
    newdim="itcenter"
    
    # Step 1️⃣: sum over spatial dimensions
    data_sum = data.sum(list(sum_dims))

    # Step 2️⃣: create new interval indices
    it = np.arange(len(tcenters))

    # Step 3️⃣: pairwise slices with shared coords
    left = data_sum.isel({dim: slice(1, None)}).rename({dim: newdim}).assign_coords({newdim: it[1:]})
    right = data_sum.isel({dim: slice(None, -1)}).rename({dim: newdim}).assign_coords({newdim: it[1:]})

    # Step 4️⃣: denominator for averaging
    den = xr.DataArray(dtcenters[1:] + dtcenters[:-1],dims=[newdim],coords={newdim: it[1:]})

    # Step 5️⃣: compute averaged part
    avg = (left + right) / den

    # Step 6️⃣: handle the first layer separately
    avg0 = data_sum.isel({dim: 0}).rename({dim: newdim}).expand_dims(newdim).assign_coords({newdim: [0]}) / dtcenters[0]

    # Step 7️⃣: concatenate
    output = xr.concat([avg0, avg], dim=newdim)

    output = output.rename({newdim: dim}).assign_coords({dim: tcenters}).chunk(chunks={dim:-1}).transpose("time",dim)

    return output



def cal_wmb_anomaly_monthly_onecenter(ds_budgets, adv, gm, tcenter, dt, THETA, more_or_less, maskregion=None):
    """
    Calculate water mass budget anomaly terms for a single tcenter.

    Parameters
    ----------
    ds_budgets : xr.Dataset
        ECCO budget variables (mass, g_mix_h, g_heat, g_adv, etc.)
    adv : xr.DataArray
        Advection fluxes
    gm : xr.DataArray
        GM fluxes
    tcenter : float
        Central time for averaging
    dt : float
        Time interval
    THETA : xr.DataArray
        Potential temperature field
    more_or_less : str
        Threshold operator for mask3d
    maskregion : xr.DataArray, optional
        Regional mask to apply

    Returns
    -------
    xr.Dataset
        Dataset with all anomaly terms at this tcenter
    """
    rhoconst = 1029.0

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

    # 时间边界
    tright = tcenter + dt/2.
    tleft  = tcenter - dt/2.

    # 保存结果的容器
    results = {}

    # 区域边界差分
    if maskregion is not None:
        _, _, _, boundary_diffs_region = extract_boundary_use_mask(maskregion, face_connections)

    # === mask: 左右边界
    mask_right = mask3d(THETA.isel(time=slice(0,-1)), tright, more_or_less)
    mask_left  = mask3d(THETA.isel(time=slice(0,-1)), tleft , more_or_less)
    if maskregion is not None:
        mask_right &= maskregion
        mask_left  &= maskregion

    # === 混合 & 热通量项
    for varname in ["g_mix_h","g_mix_v","g_heat","g_UpTb_h","g_UpTb_v",
                    "g_UbTp_h","g_UbTp_v","g_uptp_h","g_uptp_v","g_UpTp_h","g_UpTp_v"]:
        cache = compute_flux_difference(ds_budgets[varname], mask_left, mask_right,
                                        tleft, tright, tcenter)
        results[varname] = cache.expand_dims({'tcenters':[tcenter]})

    # === 水体质量倾向项 (dM/dt)
    maskTracer = mask3d(THETA, tcenter, more_or_less)
    mask = maskTracer & maskregion if maskregion is not None else maskTracer
    M = (ds_budgets["Mass"]*mask.rename({"time":"time_mass"})).sum(['face','k','j','i'])
    results["Mass"] = M.expand_dims({'tcenters':[tcenter]})

    # 用真实时间差计算 dM/dt
    dt_seconds = 86400.  # 如果你要更精确，可以改成差分
    dmdt = (M[1:].values - M[:-1].values) / dt_seconds
    dmdt = xr.DataArray(dmdt, dims=['time'], coords={'time': ds_budgets.time})
    results["g_tend"] = dmdt.expand_dims({'tcenters':[tcenter]})

    # === 淡水通量 (surface fw)
    tendency_mass_surf = (mask.isel(k=0)*ds_budgets["g_fw"]).astype("float64").sum(['face','j','i'])
    results["g_salt"] = tendency_mass_surf.expand_dims({'tcenters':[tcenter]})

    # === 平流 & GM 项
    if maskregion is None:
        g_adv = xr.zeros_like(tendency_mass_surf)
        g_gm  = xr.zeros_like(tendency_mass_surf)
    else:
        maskTracer2 = mask3d(THETA.isel(time=slice(0,-1)), tcenter, more_or_less)
        mask2 = maskTracer2 & maskregion
        _, _, _, boundary_diffs = extract_boundary_use_mask(mask2, face_connections)
        _, _, _, boundary_diffs_tracer = extract_boundary_use_mask(maskTracer2, face_connections)

        boundarymask_adv = (boundary_diffs_region != 0) & (boundary_diffs != 0)
        region_and_tracer_mask = boundarymask_adv & (boundary_diffs_tracer != 0)
        boundarymask_adv = boundarymask_adv.where(~region_and_tracer_mask, other=False)
        boundary_adv = boundary_diffs.where(boundarymask_adv, other=0.)

        g_adv = cal_G_adv_surfaceIntegrate2(adv, boundary_adv, rho_pot=None).sum(['face','k','j','i'])
        g_gm  = cal_G_adv_surfaceIntegrate2(gm , boundary_adv, rho_pot=None).sum(['face','k','j','i'])

    results["g_adv"] = g_adv.expand_dims({'tcenters':[tcenter]})
    results["g_gm"]  = g_gm.expand_dims({'tcenters':[tcenter]})

    return xr.Dataset(results)

