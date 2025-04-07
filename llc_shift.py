    def llc_shift(ds, face_connections, shift_x=0, shift_y=0,method='where'):
        if (shift_x != 0) & (shift_y==0):
            ds_shift = ds.shift(i=shift_x, j=shift_y)
            shift_len_x = abs(shift_x)
            direction_x = 'left' if shift_x < 0 else 'right'
            if method == 'loc':
                ds_shift = handle_boundaries_loc(ds, ds_shift, face_connections, 'X', direction_x, shift_len_x)
            elif method== 'isel':
                ds_shift = handle_boundaries_isel(ds, ds_shift, face_connections, 'X', direction_x, shift_len_x)
            elif method== 'where':
                ds_shift = handle_boundaries_bulk(ds, ds_shift, face_connections, 'X', direction_x, shift_len_x)
                
        elif (shift_y != 0)&(shift_x == 0):
            ds_shift = ds.shift(i=shift_x, j=shift_y)
            shift_len_y = abs(shift_y)
            direction_y = 'down' if shift_y < 0 else 'up'
            if method == 'loc':
                ds_shift = handle_boundaries_loc(ds, ds_shift, face_connections, 'Y', direction_y, shift_len_y)
            elif method== 'isel':
                ds_shift = handle_boundaries_isel(ds, ds_shift, face_connections, 'Y', direction_y, shift_len_y)
            elif method== 'where':
                ds_shift = handle_boundaries_bulk(ds, ds_shift, face_connections, 'Y', direction_y, shift_len_y)
                        
        elif (shift_y != 0)&(shift_x != 0):
            raise ValueError("Do not use since face_connections has changed once")
            ds_shift = ds.shift(i=shift_x)
            shift_len_x = abs(shift_x)
            direction_x = 'left' if shift_x < 0 else 'right'
            if method == 'loc':
                ds_shift = handle_boundaries_loc(ds, ds_shift, face_connections, 'X', direction_x, shift_len_x)
            elif method== 'isel':
                ds_shift = handle_boundaries_isel(ds, ds_shift, face_connections, 'X', direction_x, shift_len_x)
            elif method== 'where':
                ds_shift = handle_boundaries_bulk(ds, ds_shift, face_connections, 'X', direction_x, shift_len_x)
                
            ds_shift_y = ds.shift(j=shift_y)
            shift_len_y = abs(shift_y)
            direction_y = 'down' if shift_y < 0 else 'up'
            if method == 'loc':
                ds_shift = handle_boundaries_loc(ds_shift, ds_shift_y, face_connections, 'Y', direction_y, shift_len_y)
            elif method== 'isel':
                ds_shift = handle_boundaries_isel(ds_shift, ds_shift_y, face_connections, 'Y', direction_y, shift_len_y)
            elif method== 'where':
                ds_shift = handle_boundaries_bulk(ds_shift, ds_shift_y, face_connections, 'Y', direction_y, shift_len_y)
        else:
            # 未进行任何平移，返回原数据的副本
            ds_shift = ds.copy()
    
        return ds_shift
