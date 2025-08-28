import numpy as np

def voi_sph_sub(scan, xctr, yctr, zctr, Sph_Diam_Vxl, wflg):
    
    """
    Spherical VOI at (xctr, yctr, zctr) with diam in voxel units. 
    
    -> calculates the average and standard deviation of voxel intensities
    within a spherical volume of interest (VOI) using a sub-voxel search.

    Parameters:
    -----------
    scan: 3D numpy array
        3D scan with SUV data
    xctr, yctr, zctr : float
        Center coordinates of the spherical VOI in voxel units.
    Sph_Diam_Vxl : float
        Diameter of the spherical VOI in voxel units
    wflg : int
        Weighting flag:
        - 0: No additional weighting.
        - 1: Weight based on the area of the VOI/ROI in each voxel.

        
    Returns:
    --------
    tuple
        - ave_sph: Average of voxel intensities within the spherical VOI.
        - sd_sph: Standard deviation of of voxel intensities within the VOI.
    
    """

    # Define subsampling properties for finer resolution
    nsub = 10  # Number of subsamples along each voxel dimension
    nsubstep = 1.0 / nsub 
    nsubhalf = nsubstep / 2.0

    # Define sphere properties
    rad = Sph_Diam_Vxl / 2.0 
    radsq = rad ** 2
    epsrad = 0.01             # Small buffer to avoid expanding the ROI near boundaries
    rade = rad - epsrad       # Effective radius

    # Define the voxel range that encloses the spherical VOI
    xmin = round(xctr - rade)
    xmax = round(xctr + rade)
    ymin = round(yctr - rade)
    ymax = round(yctr + rade)
    zmin = round(zctr - rade)
    zmax = round(zctr + rade)

    # Calculate total number of voxels in the VOI region
    x_voivoxels = (xmax - xmin + 1)
    y_voivoxels = (ymax - ymin + 1)
    z_voivoxels = (zmax - zmin + 1)
    voivoxels = x_voivoxels * y_voivoxels * z_voivoxels

    # Initialize arrays for weights and signal values in the spherical VOI
    wt = np.zeros(voivoxels)
    sph = np.zeros(voivoxels)


    ind = 0  # Index for tracking voxels in the VOI

    # Loop through each voxel in the VOI
    for iz in range(zmin, zmax + 1):
        for iy in range(ymin, ymax + 1):
            for ix in range(xmin, xmax + 1):
                ind += 1 

                # Store the signal value of the current voxel
                sph[ind - 1] = scan[iz, iy, ix]

                # Calculate subsample ranges for the current voxel
                iz1 = iz - 0.5 + nsubhalf - zctr
                iz2 = iz + 0.5 - zctr
                iy1 = iy - 0.5 + nsubhalf - yctr
                iy2 = iy + 0.5 - yctr
                ix1 = ix - 0.5 + nsubhalf - xctr
                ix2 = ix + 0.5 - xctr

                # Subsample within the voxel
                for zsub in np.arange(iz1, iz2, nsubstep):
                    zsq = zsub ** 2 

                    for ysub in np.arange(iy1, iy2, nsubstep):
                        ysq = ysub ** 2
                        
                        for xsub in np.arange(ix1, ix2, nsubstep):
                            xsq = xsub ** 2

                            rsub = zsq + ysq + xsq

                            if rsub <= radsq:
                                wt[ind - 1] += 1


    if wflg == 1:   # Apply additional weighting if requested
        wt = wt ** 2 

    # Calculate the average and standard deviation
    ave_sph = np.sum(sph * wt) / np.sum(wt)
    sd_sph = np.sqrt(np.average((sph - ave_sph) ** 2, weights=wt))

    return ave_sph, sd_sph