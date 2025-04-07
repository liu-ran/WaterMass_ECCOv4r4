def extract_boundary_gradient_method(data, face_connections, tracer_threshold):
    #data = data.load()
    
    # recognize 'tile' or 'face'
    if 'tile' in data.dims or 'tile' in data.coords:
        tile_or_face = 'tile'
    elif 'face' in data.dims or 'face' in data.coords:
        tile_or_face = 'face'
    else:
        raise ValueError("Input data must have either 'tile' or 'face' as a dimension or coordinate")
    
    # create a mask：tracer >= threshold  ocean mask = 1; othervise = 0
    mask = (data >= tracer_threshold) & (~data.isnull())
    mask_numeric = mask.astype(float)  # into  1 or 0

    shift_up   = llc_shift(mask_numeric, face_connections, shift_x=0, shift_y=1,method='isel')
    shift_down = llc_shift(mask_numeric, face_connections, shift_x=0, shift_y=-1,method='isel')
    shift_left = llc_shift(mask_numeric, face_connections, shift_x=-1, shift_y=0,method='isel')
    shift_right= llc_shift(mask_numeric, face_connections, shift_x=1, shift_y=0,method='isel')

    diff_left = mask_numeric - shift_left
    diff_right= shift_right - mask_numeric
    diff_up   = shift_up - mask_numeric
    diff_down = mask_numeric - shift_down
    
    # using xgcm's diff, which can be across face boundary
    #diff_left = grid.diff(mask_numeric, 'X', )  # diff i = T_i - T_i-1
    #diff_down = grid.diff(mask_numeric, 'Y', )  # diff j = T_j - T_j-1
    
    shift_below = mask_numeric.shift(k=-1, fill_value=0.)
    shift_above = mask_numeric.shift(k=1, fill_value=0.)

    diff_above   = shift_above - mask_numeric
    diff_below   = mask_numeric - shift_below

    
    # convert diff into neighbour's mask
    # diff = 0 means same ( both are 1 or 0), diff != 0 
    shift_left = (diff_left == 0) & mask
    shift_right = (diff_right == 0) & mask
    shift_up = (diff_up == 0) & mask
    shift_down = (diff_down == 0) & mask
    shift_above = (diff_above == 0) & mask
    shift_below = (diff_below == 0) & mask
    
    # surface boundary
    surface_mask = mask.isel(k=0)
    surface_boundary = surface_mask & ~(shift_up.isel(k=0) & shift_down.isel(k=0) & 
                                        shift_left.isel(k=0) & shift_right.isel(k=0) )
    # deep ocean boundary
    deep_mask = mask.isel(k=slice(1, None))
    deep_boundary = deep_mask & ~(shift_up.isel(k=slice(1, None)) & 
                                  shift_down.isel(k=slice(1, None)) & 
                                  shift_left.isel(k=slice(1, None)) & 
                                  shift_right.isel(k=slice(1, None)) & 
                                  shift_above.isel(k=slice(1, None)) & 
                                  shift_below.isel(k=slice(1, None)))
    boundary = xr.concat([surface_boundary, deep_boundary], dim='k')

    # 
    diff_left_boundary = diff_left.where(boundary, 0)
    diff_right_boundary = diff_right.where(boundary, 0)
    diff_up_boundary = diff_up.where(boundary, 0)
    diff_down_boundary = diff_down.where(boundary, 0)

    boundary_vertical = mask & ~( shift_above & shift_below & shift_up & shift_down & shift_left & shift_right )
    
    diff_above_boundary = diff_above.where(boundary_vertical, 0)
    diff_below_boundary = diff_below.where(boundary_vertical, 0)
  
    return boundary, tile_or_face, mask, diff_left_boundary, diff_right_boundary, diff_up_boundary, diff_down_boundary, diff_above_boundary, diff_below_boundary
