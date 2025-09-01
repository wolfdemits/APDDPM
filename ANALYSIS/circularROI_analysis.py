import numpy as np

def roi_circ_sub(scan, xctr, yctr, zctr, Sph_Diam_Vxl, wflg):

    """
    Circular ROI at (xctr, yctr) in slice zctr with diam in voxel units
    
    -> calculates the average and standard deviation of voxel intensities within
    a circular region of interest (ROI) using a sub-voxel search. The ROI can be 
    defined in a single axial plane or spanning two slices. 

    Parameters:
    -----------
    scan: 3D numpy array
        3D scan with SUV data.
    xctr, yctr, zctr : float
        Center coordinates of the circular ROI center in voxel units.
    Sph_Diam_Vxl : float
        Diameter of the circular ROI in in voxel units
    wflg : int
        Weighting flag:
        - 0: No additional weighting.
        - 1: Weight based on the area of the VOI/ROI in each voxel.

        
    Returns:
    --------
    tuple
        - ave_circ: Average of voxel intensities within the circular ROI.
        - sd_circ: Standard deviation of voxel intensities within the circular ROI.
    
    """

    # Define subsampling properties for finer resolution
    nsub = 10  # Number of subsamples on each side of a voxel
    nsubstep = 1.0 / nsub
    nsubhalf = nsubstep / 2.0

    # Define sphere properties
    rad = Sph_Diam_Vxl / 2.0
    radsq = rad ** 2
    epsrad = 0.01        # Small buffer to avoid expanding the ROI near boundaries
    rade = rad - epsrad  # Effective radius
    
    # Define the voxel range that encloses the circular ROI
    xmin = round(xctr - rade)
    xmax = round(xctr + rade)
    ymin = round(yctr - rade)
    ymax = round(yctr + rade)

    zctr0 = round(np.fix(zctr))

    # Calculate total number of voxels in the ROI region
    x_roivoxels = (xmax - xmin + 1)
    y_roivoxels = (ymax - ymin + 1)
    roivoxels = x_roivoxels * y_roivoxels

    # Initialize arrays for weights and signal values for ROI 
    wt = np.zeros(roivoxels)
    circ = np.zeros(roivoxels)

    # Account when the ROI (=sphere coords) spans between two slices
    wt_ax1 = 1 - np.mod(zctr, 1)
    wt_ax2 = 1 - wt_ax1


    ind = 0     # Index for tracking voxels in the ROI

    
    ### START ###

    if (wt_ax2 < 0.01): # Only within one slice ###############################

        # Iterate through each voxel in the ROI
        for iy in range(ymin, ymax+1):
            for ix in range(xmin, xmax+1):
                ind += 1

                # Extract signal of current voxel from single slice
                circ[ind - 1] = scan[zctr0, iy, ix]

                # Calculate subsample ranges for the current voxel
                iy1 = iy - 0.5 + nsubhalf - yctr
                iy2 = iy + 0.5 - yctr
                ix1 = ix - 0.5 + nsubhalf - xctr
                ix2 = ix + 0.5 - xctr

                # Subsample within the voxel
                for ysub in np.arange(iy1, iy2, nsubstep):
                    ysq = ysub ** 2

                    for xsub in np.arange(ix1, ix2, nsubstep):
                        xsq = xsub ** 2
                            
                        rsub = ysq + xsq
                            
                        if rsub <= radsq:
                            wt[ind - 1] += 1


    else: # between two slices ###############################

        # Iterate through each voxel in the ROI
        for iy in range(ymin, ymax+1):
            for ix in range(xmin, xmax+1):
                ind += 1

                # Extract signal of current voxel from two slices and calculate weighted average
                data1 = scan[zctr0, iy, ix]
                data2 = scan[zctr0+1, iy, ix]
                circ[ind - 1] = data1 * wt_ax1 + data2 * wt_ax2

                # Calculate subsample ranges for the current voxel
                iy1 = iy - 0.5 + nsubhalf - yctr
                iy2 = iy + 0.5 - yctr
                ix1 = ix - 0.5 + nsubhalf - xctr
                ix2 = ix + 0.5 - xctr

                # Subsample within the voxel
                for ysub in np.arange(iy1, iy2, nsubstep):
                    ysq = ysub ** 2

                    for xsub in np.arange(ix1, ix2, nsubstep):
                        xsq = xsub ** 2
                            
                        rsub = ysq + xsq
                            
                        if rsub <= radsq:
                            wt[ind - 1] += 1


    if wflg == 1:   # Apply additional weighting if requested
        wt = wt ** 2 

    # Calculate the average and standard deviation
    ave_circ = np.sum(circ * wt) / np.sum(wt)
    sd_circ = np.sqrt(np.average((circ - ave_circ) ** 2, weights=wt))

    return ave_circ, sd_circ