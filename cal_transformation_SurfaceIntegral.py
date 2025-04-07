def cal_transformation_SurfaceIntegral(ds,diff_left_boundary,diff_right_boundary,diff_up_boundary,diff_down_boundary,diff_above_boundary,diff_below_boundary):
        
    DFxE_TH = ds.DFxE_TH  #  degree_C m3 s-1
    DFyE_TH = ds.DFyE_TH  #  degree_C m3 s-1
    DFrE_TH = ds.DFrE_TH  #  degree_C m3 s-1
    DFrI_TH = ds.DFrI_TH  #  degree_C m3 s-1
    
    DFxE_TH_shift  = llc_shift(DFxE_TH.rename({'i_g': 'i'}), face_connections, shift_x=-1, shift_y=0,method='isel')
    DFyE_TH_shift  = llc_shift(DFyE_TH.rename({'j_g': 'j'}), face_connections, shift_x=0, shift_y=-1,method='isel')

    lateral_transport = DFxE_TH.rename({'i_g': 'i'})*diff_left_boundary + \
         DFxE_TH_shift*diff_right_boundary + \
         DFyE_TH.rename({'j_g': 'j'})*diff_down_boundary +\
         DFyE_TH_shift*diff_up_boundary


    DFrE_TH_below = DFrE_TH.rename({'k_l': 'k'}).shift(k=-1, fill_value=0.)
    DFrI_TH_below = DFrI_TH.rename({'k_l': 'k'}).shift(k=-1, fill_value=0.)
    
    vertical_transport = DFrE_TH.rename({'k_l': 'k'})*diff_above_boundary + DFrE_TH_below*diff_below_boundary

    g_mat = -(lateral_transport + vertical_transport)*rho_pot
    
    return g_mat
