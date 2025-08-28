import matplotlib.pyplot as plt

def plot_coordinate(scan, coords):
    x, y, z = coords

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # coronal
    ax[0].imshow(scan[:,y,:], cmap='gray_r')
    ax[0].set_title('Coronal')
    ax[0].axhline(x, linestyle='--', c='red')
    ax[0].axvline(z, linestyle='--', c='orange')
    # Sagittal
    ax[1].imshow(scan[x,:,:], cmap='gray_r')
    ax[1].set_title('Sagittal')
    ax[1].axhline(y, linestyle='--', c='green')
    ax[1].axvline(z, linestyle='--', c='orange')
    # Transaxial
    ax[2].imshow(scan[:,:,z], cmap='gray_r')
    ax[2].set_title('Transaxial')
    ax[2].axhline(x, linestyle='--', c='red')
    ax[2].axvline(y, linestyle='--', c='green')

    return fig

def plot_divisions(scans, plane, slice_idx):
    # TODO
    return