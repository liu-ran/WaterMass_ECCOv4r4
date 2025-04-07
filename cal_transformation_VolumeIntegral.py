def cal_g_mat(ds): 
    DFxE_TH = ds.DFxE_TH  #  degree_C m3 s-1
    DFyE_TH = ds.DFyE_TH  #  degree_C m3 s-1
    DFrE_TH = ds.DFrE_TH  #  degree_C m3 s-1
    DFrI_TH = ds.DFrI_TH  #  degree_C m3 s-1

    
    DFxE_TH_right   = llc_shift(DFxE_TH.rename({'i_g': 'i'}), face_connections, shift_x=1, shift_y=0,method='isel')
    DFyE_TH_up      = llc_shift(DFyE_TH.rename({'j_g': 'j'}), face_connections, shift_x=0, shift_y=1,method='isel')

    # Convergence of horizontal advection (degC m^3/s)
    dif_hConvH = (-( DFxE_TH_right - DFxE_TH.rename({'i_g': 'i'})  + DFyE_TH_up - DFyE_TH.rename({'j_g': 'j'})))
    
    DFrI_TH_below = DFrI_TH.rename({'k_l': 'k'}).shift(k=-1, fill_value=0.)
    DFrE_TH_below = DFrE_TH.rename({'k_l': 'k'}).shift(k=-1, fill_value=0.)

    dif_vConvH = -(DFrE_TH.rename({'k_l': 'k'}) - DFrI_TH_below + DFrI_TH.rename({'k_l': 'k'}) - DFrE_TH_below)

    g_mat = (dif_hConvH + dif_vConvH)*rho_pot
  
    return g_mat
    
