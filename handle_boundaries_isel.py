    import xarray as xr
    def handle_boundaries_isel(ds, ds_shift, face_connections, axis, direction, shift_len):
        # 确定轴对应的轴名
        if axis == 'X':
            axisname = 'i'
        elif axis == 'Y':
            axisname = 'j'
        else:
            raise ValueError("axis must be 'X' or 'Y'") 
            
        for face, connections in face_connections['face'].items():
            #print(face)
            conn = connections[axis][1] if direction in ['left', 'down'] else connections[axis][0]
            if conn is not None:
                connected_face, connected_axis, flip = conn
                # 确定连接面的轴对应的轴名
                if connected_axis == 'X':
                    connected_axisname = 'i'
                elif connected_axis == 'Y':
                    connected_axisname = 'j'
                else:
                    raise ValueError("Connected_axis must be 'X' or 'Y'")
                # 选择相邻面上的数据切片
                if direction in ['left', 'down']:
                    data_sel = {'face': connected_face, connected_axisname: slice(0, shift_len)}
                    target_sel = {'face': face, axisname: slice(-shift_len, None)}
                elif direction in ['right', 'up']:
                    data_sel = {'face': connected_face, connected_axisname: slice(-shift_len, None)}
                    target_sel = {'face': face, axisname: slice(0, shift_len)}
                        
                # 提取相邻面的数据
                connected_data = ds.isel(**data_sel)
                # 如果连接的轴与目标轴不同，需要flip and transpose
                #if (connected_axis == 'X' and axis == 'Y') or (connected_axis == 'Y' and axis == 'X'):
                if connected_axis!= axis:
                    # 翻转 连接面非连接轴 对应的维度
                    connected_data = connected_data.isel(**{axisname: slice(None, None, -1)})
        
                    # 重命名并转置，确保维度顺序正确
                    new_dims = {axisname: connected_axisname, connected_axisname: axisname}
                    connected_data = connected_data.rename(new_dims)
                    # 保留所有维度顺序
                    all_dims = [d if d not in new_dims else new_dims[d] for d in connected_data.dims]
                    connected_data = connected_data.transpose(*all_dims)
                    ## 重新赋值坐标 (不再需要)
                    #target_coords = ds.loc[target_sel]
                    #connected_data = connected_data.assign_coords({axisname: target_coords[axisname].values,
                    #                       connected_axisname: target_coords[connected_axisname].values} )
                #else: #不矫正坐标则不需要
                    #target_coords = ds.loc(**target_sel)
                    #target_coords = ds.loc[target_sel]
                    #print(target_sel)
                    #connected_data = connected_data.assign_coords({axisname: target_coords[axisname].values,
                    #                       connected_axisname: target_coords[connected_axisname].values} )
            else:
                # 选择相邻面上的数据切片
                if direction in ['left', 'down']:
                    target_sel = {'face': face, axisname: slice(-shift_len, None)}
                elif direction in ['right', 'up']:
                    target_sel = {'face': face, axisname: slice(0, shift_len)}
                connected_data = ds.isel(**target_sel)
                
                connected_data[:] = np.nan
           
            # ds_shift.isel(**target_sel)[:] = connected_data.values  ### 这样不能给xarry赋值
            # Step 1: 用 isel 提取目标区域的坐标
            target_subset = ds_shift.isel(**target_sel)
            target_labels = {
                'face': target_subset['face'].values.item(),  # 转换为标量
                axisname: slice(target_subset[axisname].values[0], target_subset[axisname].values[-1], None)
            }
            # Step 2: 用 loc 赋值
            ds_shift.loc[target_labels] = connected_data.values
            
        return ds_shift
