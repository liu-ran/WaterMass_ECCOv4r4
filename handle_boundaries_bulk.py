    import xarray as xr
    #### using xr.where
    def handle_boundaries_bulk(ds, ds_shift, face_connections, axis, direction, shift_len):
        #参数:
        #- ds: ordional llc xarray DataArray
        #- ds_shift: first-step shifted xarray DataArray
        #- face_connections: llc face
        #- axis: 'X' 或 'Y'
        #- direction: 'left', 'right', 'up', 'down'
        #- shift_len: shift length

        # llc C-cell grid using "i" & "j"
        axisname_map = {'X': 'i', 'Y': 'j'}
        axisname = axisname_map.get(axis)

        if not axisname:
            raise ValueError("axis must be 'X' or 'Y'")
        if axisname=='i':
            NL = len(ds['i'])
        else:
            NL = len(ds['j'])
        
        # 初始化一个全NaN的DataArray用于替换
        replacement = xr.full_like(ds_shift, np.nan)
    
        # 收集所有需要替换的数据和目标位置
        for face, connections in face_connections['face'].items():
            # 根据方向获取连接信息
            if direction in ['left', 'down']:
                conn = connections.get(axis, [None, None])[1]  # 对应方向的连接
            else:
                conn = connections.get(axis, [None, None])[0]  # 其他方向的连接
        
            if conn is not None:
                connected_face, connected_axis, flip = conn
                connected_axisname = axisname_map.get(connected_axis)
                if not connected_axisname:
                    raise ValueError("Connected_axis must be 'X' or 'Y'")

                # 定义选择器
                if direction in ['left', 'down']:
                    data_sel = {'face': connected_face, connected_axisname: slice(0, shift_len-1)}
                    target_sel = {'face': face, axisname: slice(NL-shift_len, None)}
                else:  # 'right', 'up'
                    data_sel = {'face': connected_face, connected_axisname: slice(NL-shift_len, None)}
                    target_sel = {'face': face, axisname: slice(0, shift_len-1)}
            
                # 提取相邻面的数据
                connected_data = ds.loc[data_sel]
            
                #print(connected_data)
                # 如果连接的轴与目标轴不同，则交换轴名并转置
                if connected_axisname != axisname:
                    # 翻转 连接面非连接轴 对应的维度
                    connected_data = connected_data.isel(**{axisname: slice(None, None, -1)})
                
                    new_dims = {axisname: connected_axisname, connected_axisname: axisname}
                    connected_data = connected_data.rename(new_dims)

                    all_dims = [d if d not in new_dims else new_dims[d] for d in connected_data.dims]
                    connected_data = connected_data.transpose(*all_dims)
                    ## 重新赋值坐标
                    #print(target_sel)

                    target_coords = ds.loc[target_sel]
                    #print(target_coords)
                    connected_data = connected_data.assign_coords({axisname: target_coords[axisname],
                                   connected_axisname: target_coords[connected_axisname]} )
            
                replacement.loc[target_sel] = connected_data.values
            
            else:
                # 无连接时，保持replacement为NaN，无需额外操作
                pass
    
        # use xr.where to revise the boundaries of the shifted xarray 
        ds_shift = xr.where(~replacement.isnull(), replacement, ds_shift)
    
        return ds_shift
