
def plot_tiles(tiles, cmap=None, 
               layout='llc', rotate_to_latlon=False,
               Arctic_cap_tile_location = 2,
               show_colorbar=False,  
               show_cbar_label=False, 
               show_tile_labels= True,
               cbar_label = '', 
               fig_size = 9,  
               less_output=True,
               **kwargs):
    """

    Plots the 13 tiles of the lat-lon-cap (LLC) grid
    
    Parameters
    ----------
    tiles : numpy.ndarray or dask.array.core.Array or xarray.core.dataarray.DataArray
        an array of n=1..13 tiles of dimension n x llc x llc 

            - If *xarray DataArray* or *dask Array* tiles are accessed via *tiles.sel(tile=n)*
            - If *numpy ndarray* tiles are acceed via [tile,:,:] and thus n must be 13.

    cmap : matplotlib.colors.Colormap, optional
        see plot_utils.assign_colormap for default
        a colormap for the figure

    layout : string, optional, default 'llc'
        a code indicating the layout of the tiles

        :llc:    situates tiles in a fan-like manner which conveys how the tiles 
                 are oriented in the model in terms of x an y
    
        :latlon: situates tiles in a more geographically recognizable manner.  
                 Note, this does not rotate tiles 7..12, it just places tiles 
                 7..12 adjacent to tiles 0..5.  To rotate tiles 7..12 
                 specifiy *rotate_to_latlon* as True
                     
    rotate_to_latlon : boolean, default False
        rotate tiles 7..12 so that columns correspond with
        longitude and rows correspond to latitude.  Note, this rotates
        vector fields (vectors positive in x in tiles 7..12 will be -y 
        after rotation).  

    Arctic_cap_tile_location : int, default 2
        integer, which lat-lon tile to place the Arctic tile over. can be 
        2, 5, 7 or 10.
        
    show_colorbar : boolean, optional, default False
        add a colorbar
        
    show_cbar_label : boolean, optional, default False
        add a label on the colorbar
        
    show_tile_labels : boolean, optional, default True
        show tiles numbers in subplot titles
        
    cbar_label : str, optional, default '' (empty string)
        the label to use for the colorbar
      
    less_output : boolean, optional, default True
        A debugging flag.  True = less debugging output
                
    cmin/cmax : floats, optional, default calculate using the min/max of the data
        the minimum and maximum values to use for the colormap
        
    fig_size : float, optional, default 9 inches
        size of the figure in inches
        
    fig_num : int, optional, default none
        the figure number to make the plot in.  By default make a new figure.
        
    Returns
    -------
    f : matplotlib figure object

    cur_arr : numpy ndarray
        numpy array of size:
            (llc*nrows, llc*ncols)
        where llc is the size of tile for llc geometry (e.g. 90)
        nrows, ncols refers to the subplot size
        For now, only implemented for llc90, otherwise None is returned
    """
    import dask
    import numpy as np
    # 自动识别 'tile' 或 'face'
    if 'tile' in tiles.dims or 'tile' in tiles.coords:
        tile_or_face = 'tile'
    elif 'face' in tiles.dims or 'face' in tiles.coords:
        tile_or_face = 'face'
    else:
        raise ValueError("Input data must have either 'tile' or 'face' as a dimension or coordinate")
        
    # processing for dask array (?)
    if isinstance(tiles, dask.array.core.Array):
        tiles = np.asarray(tiles.squeeze())

    # get default colormap
    cmap, (cmin,cmax) = assign_colormap(tiles,cmap)

    #%%
    fig_num = -1
    for key in kwargs:
        if key == "cmin":
            cmin = kwargs[key]
        elif key == "cmax":
            cmax =  kwargs[key]
        elif key == 'fig_num':
            fig_num = kwargs[key]
        else:
            print("unrecognized argument ", key)
    
    # if llc90, return array otherwise not implemented
    get_array = True
    nx = tiles.shape[-1]
    if nx != 90:
        get_array = False
        warnings.warn('Will not return array for non llc90 data')

    # set sizing for subplots
    fac1 = 1; fac2=1
    if show_tile_labels and show_colorbar:
        fac2 = 1.15

    if show_tile_labels==False:
        if show_colorbar:
            fac2 =  0.8766666666666666
        else:
            fac2 = 9.06/9
        
    if layout == 'llc' :
        nrows=5
        ncols=5

        # plotting of the tiles happens in a 5x5 grid
        # which tile to plot for any one of the 25 spots is indicated with a list
        # a value of negative one means do not plot anything in that spot.
        tile_order = np.array([-1, -1, 10, 11, 12, \
                               -1,  6,  7,  8,  9, \
                                2,  5, -1, -1, -1, \
                                1,  4, -1, -1, -1, \
                                0,  3, -1, -1, -1])
    
    elif layout == 'latlon':
        ncols = 4
        nrows = 4

        # plotting of the tiles happens in a 4x4 grid
        # which tile to plot for any one of the 16 spots is indicated with a list
        # a value of negative one means do not plot anything in that spot.
        # the top row will have the Arctic tile.  You can choose where the 
        # Arctic tile goes.  By default it goes in the second column.
        if Arctic_cap_tile_location not in [2,5,7,10]:
            print('Arctic Cap Alignment is not one of 2,5,7,10, using 2')
            Arctic_cap_tile_location  = 2    
        
        if  Arctic_cap_tile_location == 2: # plot in 1st position, column 1
            tile_order_top_row = [6, -1, -1, -1]
        elif Arctic_cap_tile_location == 5:
            tile_order_top_row = [-1, 6, -1, -1]
        elif Arctic_cap_tile_location == 7:# plot in 3rd position, column 3
            tile_order_top_row = [-1, -1, 6, -1]
        elif Arctic_cap_tile_location == 10:# plot in 4th position, column 4
            tile_order_top_row = [-1, -1, -1, 6]
            
        # the order of the rest of the tile is fixed.  four columns each with 
        # three rows.
        tile_order_bottom_rows =[2, 5, 7, 10, \
                                 1, 4, 8, 11, \
                                 0, 3, 9, 12]
        
        # these are lists so to combine tile_orde_first and tile_order_rest 
        # you just add them in python (wierd).  If these were numpy arrays 
        # one would use np.concatenate()
        tile_order = tile_order_top_row + tile_order_bottom_rows

    # create fig object
    if fig_num > 0:
        f, axarr = plt.subplots(nrows, ncols, num=fig_num, figsize=(18,18), constrained_layout=True)
    else:
        f, axarr = plt.subplots(nrows, ncols, num=fig_num, figsize=(18,18), constrained_layout=True)

    #%%
    f.set_size_inches(fac1*fig_size, fig_size*fac2)

    if show_tile_labels==False:
        f.subplots_adjust(wspace=0, hspace=0)
    
    # loop through the axes array and plot tiles where tile_order != -1
    cur_arr = np.zeros((nrows*nx,ncols*nx)) if get_array else None
    cur_tile = -1
    for i, ax in enumerate(axarr.ravel()):
        ax.axis('off')

        cur_tile_num = tile_order[i]
        have_tile = False

        if cur_tile_num >= 0:
            if type(tiles) == np.ndarray:
                have_tile = True
                cur_tile = tiles[cur_tile_num ]
                    
            elif isinstance(tiles, dask.array.core.Array) or \
                 isinstance(tiles, xr.core.dataarray.DataArray):

                if cur_tile_num in tiles[tile_or_face] :
                    have_tile = True
                    cur_tile = tiles.sel(**{tile_or_face: cur_tile_num})
                    #cur_tile = tiles.sel(tile=cur_tile_num)
    
            if have_tile:
                if (layout == 'latlon' and rotate_to_latlon and cur_tile_num == 6):
                    if Arctic_cap_tile_location == 2:
                        cur_tile = np.rot90(cur_tile,-1)
                    elif Arctic_cap_tile_location == 7:
                        cur_tile = np.rot90(cur_tile,-3)
                    elif Arctic_cap_tile_location == 10:
                        cur_tile = np.rot90(cur_tile,2)
                
                if (layout == 'latlon' and rotate_to_latlon and 
                    cur_tile_num > 6):
        
                    cur_tile = np.rot90(cur_tile)
        
                im=ax.imshow(cur_tile, vmin=cmin, vmax=cmax, cmap=cmap, 
                             origin='lower')

            # axis handling
            ax.set_aspect('equal')
            ax.axis('on')
            if show_tile_labels:
                ax.set_title('Tile ' + str(cur_tile_num))
                
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Generate array from this process
            colnum = ncols-1-int(i/ncols)
            rownum = i%nrows
            rownump1 = int(rownum + 1)
            colnump1 = int(colnum + 1)
            if not less_output:
                print('i=',i,rownum, colnum)
            
            if cur_tile_num>=0 and get_array:
                cur_arr[colnum*nx:colnump1*nx, rownum*nx:rownump1*nx] = cur_tile

    # show the colorbar
    if show_colorbar:
        if show_tile_labels:
            f.subplots_adjust(left=None, bottom=None, right=0.8)
        else:
            f.subplots_adjust(right=0.8, left=None, bottom=None,
                              top=None, wspace=0, hspace=0)
            
        #[left, bottom, width, height]
        h=.6;w=.025
        cbar_ax = f.add_axes([0.85, (1-h)/2, w, h])
        cbar = f.colorbar(im, cax=cbar_ax)#, format='%.0e')        
        if show_cbar_label:
            cbar.set_label(cbar_label)

    return f, cur_arr, axarr
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def assign_colormap(fld,user_cmap=None):
    """assign a default colormap based on input field
    following xarray defaults

    Sequential fld: viridis
    Divergent fld : RdBu_r
    

    Parameters
    ----------
    fld : xarray.DataArray or numpy.ndarray
        must be in tiled format

    user_cmap : str or None, optional
        if user_cmap is specified, use this
        None if not specified, and default cmaps selected

    Returns
    -------
    cmap : str
        assigned colormap depending on diverging or sequential
        data in field

    (cmin,cmax) : tuple of floats
        minimum and maximum values for colormap
    """

    #%%    
    # If fld span positive/negative, default to normalize about 0
    # otherwise, regular (sequential). Assign cmap accordingly.
    import numpy as np
    
    cmin = np.nanmin(fld)
    cmax = np.nanmax(fld)
    if cmin*cmax<0:
        cmax=np.nanmax(np.abs(fld))
        cmin=-cmax
        cmap = 'RdBu_r'
    else:
        cmap = 'viridis'

    # override if user_cmap is not None
    cmap = cmap if user_cmap is None else user_cmap

    return cmap, (cmin,cmax)

