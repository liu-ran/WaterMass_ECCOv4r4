import numpy as np
import xarray as xr

def handle_boundaries_uv_isel(uds, uds_shift, vds, face_connections, axis, direction, shift_len):
    """
    Handle boundary conditions for shifted U velocity component in LLC grid.
    
    Parameters:
    uds (xarray.Dataset): Original U velocity dataset
    uds_shift (xarray.Dataset): Shifted U velocity dataset
    vds (xarray.Dataset): V velocity dataset
    face_connections (dict): Dictionary describing face connections
    axis (str): 'X' or 'Y', indicating the axis of shift
    direction (str): 'left', 'right', 'up', or 'down'
    shift_len (int): Length of the shift
    
    Returns:
    xarray.Dataset: Updated shifted U velocity dataset
    """
    axisname = 'i' if axis == 'X' else 'j'
    
    for face, connections in face_connections['face'].items():
        conn = connections[axis][1] if direction in ['left', 'down'] else connections[axis][0]
        if conn is not None:
            connected_face, connected_axis, flip = conn
            connected_axisname = 'i' if connected_axis == 'X' else 'j'
            
            data_sel = {'face': connected_face, connected_axisname: slice(0, shift_len)}
            target_sel = {'face': face, axisname: slice(-shift_len, None)}
            if direction in ['right', 'up']:
                data_sel[connected_axisname] = slice(-shift_len, None)
                target_sel[axisname] = slice(0, shift_len)
            
            connected_data = vds.isel(**data_sel) if connected_axis != axis else uds.isel(**data_sel)
            if connected_axis!= axis:
                # 翻转 连接面非连接轴 对应的维度
                connected_data = connected_data.isel(**{axisname: slice(None, None, -1)})
                # 重命名并转置，确保维度顺序正确
                new_dims = {axisname: connected_axisname, connected_axisname: axisname}
                connected_data = connected_data.rename(new_dims)
                # 保留所有维度顺序
                all_dims = [d if d not in new_dims else new_dims[d] for d in connected_data.dims]
                connected_data = connected_data.transpose(*all_dims)
        else:
            target_sel = {'face': face, axisname: slice(-shift_len, None) if direction in ['left', 'down'] else slice(0, shift_len)}
            connected_data = uds.isel(**target_sel).copy()
            connected_data[:] = np.nan
        
        target_subset = uds_shift.isel(**target_sel)
        target_labels = {
            'face': target_subset['face'].values.item(),
            axisname: slice(target_subset[axisname].values[0], target_subset[axisname].values[-1], None)
        }
        uds_shift.loc[target_labels] = connected_data.values
    
    return uds_shift

def llc_uv_shift(uds, vds, face_connections, shift_x=0, shift_y=0):
    """
    Shift first input component (uds) in LLC grid and handle boundary conditions.

    This function is designed to shift the grid to obtain points around each cell
    (up, down, left, right). It handles the complex boundary conditions of the LLC grid,
    ensuring proper data continuity across different faces of the grid. 
    
    Parameters:
    uds (xarray.Dataset): U velocity dataset
    vds (xarray.Dataset): V velocity dataset
    face_connections (dict): Dictionary describing face connections
    shift_x (int): Shift in X direction
    shift_y (int): Shift in Y direction
    
    Returns:
    xarray.Dataset: Shifted U velocity dataset
    """
    if shift_x != 0 and shift_y == 0:
        uds_shift = uds.shift(i=shift_x)
        direction_x = 'left' if shift_x < 0 else 'right'
        u_shift = handle_boundaries_uv_isel(uds, uds_shift, vds, face_connections, 'X', direction_x, abs(shift_x))
    elif shift_y != 0 and shift_x == 0:
        uds_shift = uds.shift(j=shift_y)
        direction_y = 'down' if shift_y < 0 else 'up'
        u_shift = handle_boundaries_uv_isel(uds, uds_shift, vds, face_connections, 'Y', direction_y, abs(shift_y))
    elif shift_y != 0 and shift_x != 0:
        raise ValueError("Cannot shift in both X and Y directions simultaneously")
    else:
        u_shift = uds.copy()
    
    return u_shift