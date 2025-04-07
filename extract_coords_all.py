def extract_coords_kernel(boundary):
    # 1. 计算编码后的坐标
    coords = xr.where(boundary, 
                      boundary.face * 1000000 + boundary.k * 10000 + boundary.i * 100 + boundary.j, 
                      np.nan)
    boundary_flat = coords.stack(points=['face', 'k', 'i', 'j'])

    # 2. 计算布尔掩码并立即计算为 NumPy 数组
    isnan_mask = xr.apply_ufunc(np.isnan, boundary_flat, dask="parallelized").compute()
    boundary_valid = boundary_flat.where(~isnan_mask, drop=True)

    # 3. 计算结果
    boundary_valid = boundary_valid.compute()

    # 4. 解码 tile, level, i, j
    boundary_tile = (boundary_valid // 1000000).astype(int)
    boundary_level = ((boundary_valid % 1000000) // 10000).astype(int)
    boundary_i = ((boundary_valid % 10000) // 100).astype(int)
    boundary_j = (boundary_valid % 100).astype(int)
    
    return boundary_tile, boundary_level, boundary_i, boundary_j

# 处理所有时间步的坐标并存储为 pandas DataFrame
def extract_coords_all(boundary):
    import pandas as pd
    time_vals = boundary.time.values
    n_times = len(time_vals)
    
    # 存储所有时间步的坐标
    all_data = []
    
    for t in range(n_times):
        time_val = time_vals[t]
        tile, level, i, j = extract_coords_kernel(boundary.isel(time=t))
        
        # 为当前时间步创建 DataFrame
        df_t = pd.DataFrame({
            'time': np.full(len(tile), time_val),
            'tile': tile,
            'level': level,
            'i': i,
            'j': j
        })
        all_data.append(df_t)
    
    # 合并所有时间步的数据
    boundary_df = pd.concat(all_data, ignore_index=True)
    return boundary_df
