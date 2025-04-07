def extract_boundary_gradient_method(data, face_connections, tracer_threshold):
    #data = data.load()
    
    # 自动识别 'tile' 或 'face'
    if 'tile' in data.dims or 'tile' in data.coords:
        tile_or_face = 'tile'
    elif 'face' in data.dims or 'face' in data.coords:
        tile_or_face = 'face'
    else:
        raise ValueError("Input data must have either 'tile' or 'face' as a dimension or coordinate")
    
    # 创建数值型掩码：tracer >= threshold 的海洋点为 1，否则为 0
    mask = (data >= tracer_threshold) & (~data.isnull())
    mask_numeric = mask.astype(float)  # 转换为 1 和 0

    shift_up   = llc_shift(mask_numeric, face_connections, shift_x=0, shift_y=1,method='isel')
    shift_down = llc_shift(mask_numeric, face_connections, shift_x=0, shift_y=-1,method='isel')
    shift_left = llc_shift(mask_numeric, face_connections, shift_x=-1, shift_y=0,method='isel')
    shift_right= llc_shift(mask_numeric, face_connections, shift_x=1, shift_y=0,method='isel')

    diff_left = mask_numeric - shift_left
    diff_right= shift_right - mask_numeric
    diff_up   = shift_up - mask_numeric
    diff_down = mask_numeric - shift_down
    
    # 使用 xgcm 的 diff 检查邻域，跨越 face 边界
    #diff_left = grid.diff(mask_numeric, 'X', )  # diff i = T_i - T_i-1
    #diff_down = grid.diff(mask_numeric, 'Y', )  # diff j = T_j - T_j-1
    
    # 垂直方向 (k) 仍使用 shift，因为 k 不涉及 face 连接
    # k=-1  向上shift； k=1  向下shift
    shift_below = mask_numeric.shift(k=-1, fill_value=0.)
    shift_above = mask_numeric.shift(k=1, fill_value=0.)

    diff_above   = shift_above - mask_numeric
    diff_below   = mask_numeric - shift_below
    
    
    
    # 将 diff 转换为邻域掩码（布尔型）
    # diff = 0 表示邻点与当前点一致（都为 1 或 0），diff != 0 表示不同
    shift_left = (diff_left == 0) & mask
    shift_right = (diff_right == 0) & mask
    shift_up = (diff_up == 0) & mask
    shift_down = (diff_down == 0) & mask
    shift_above = (diff_above == 0) & mask
    shift_below = (diff_below == 0) & mask
    
    # 分层处理边界
    surface_mask = mask.isel(k=0)
    surface_boundary = surface_mask & ~(shift_up.isel(k=0) & shift_down.isel(k=0) & 
                                        shift_left.isel(k=0) & shift_right.isel(k=0) )
    
    deep_mask = mask.isel(k=slice(1, None))
    deep_boundary = deep_mask & ~(shift_up.isel(k=slice(1, None)) & 
                                  shift_down.isel(k=slice(1, None)) & 
                                  shift_left.isel(k=slice(1, None)) & 
                                  shift_right.isel(k=slice(1, None)) & 
                                  shift_above.isel(k=slice(1, None)) & 
                                  shift_below.isel(k=slice(1, None)))
    
    boundary = xr.concat([surface_boundary, deep_boundary], dim='k')

    diff_left_boundary = diff_left.where(boundary, 0)
    diff_right_boundary = diff_right.where(boundary, 0)
    diff_up_boundary = diff_up.where(boundary, 0)
    diff_down_boundary = diff_down.where(boundary, 0)

    boundary_vertical = mask & ~( shift_above & shift_below & shift_up & shift_down & shift_left & shift_right )
    
    diff_above_boundary = diff_above.where(boundary_vertical, 0)
    diff_below_boundary = diff_below.where(boundary_vertical, 0)
  
    return boundary, tile_or_face, mask, diff_left_boundary, diff_right_boundary, diff_up_boundary, diff_down_boundary, diff_above_boundary, diff_below_boundary
