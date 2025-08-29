import matplotlib.pyplot as plt

class PlotCoordinate:
    def __init__(self, scan, coords):
        """
        Interactive plot to display 3D scan as seen from 3 anatomical planes.

        COORDINATE SYSTEM USED: 

        Global coordinate system used: arr = [z,y,x]
        z: 0 -> 712 = Cranial -> Caudal; 
        y: 0 -> 300 = Ventral -> Dorsal; 
        x: 0 -> 300 = Anatomical Left -> Anatomical Right; 

        Parameters:
        -----------
        scan: 3D numpy array
            3D scan to display
        coords: tuple as defined: (x,y,z)
            initial coordinate to plot
        
        Returns:
        --------
        PlotCoordinate Object, calling plt.show() should display the interactive plot. 

        """
        self.scan = scan
        self.coords = coords

        # init fig, ax
        self.fig, self.ax = plt.subplots(1, 3, figsize=(15, 5))

        # initial render
        self.render()

        # listens for click events
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        return

    def render(self):
        x, y, z = self.coords

        # clear previous ax
        for a in self.ax:
            a.clear()

        # coronal
        self.ax[0].imshow(self.scan[:,y,:], cmap='gray_r')
        self.ax[0].set_title('Coronal', c='green')
        self.ax[0].axhline(z, linestyle='--', c='blue', alpha=0.2)
        self.ax[0].axvline(x, linestyle='--', c='red', alpha=0.2)
        # Sagittal
        self.ax[1].imshow(self.scan[:,:,x], cmap='gray_r')
        self.ax[1].set_title('Sagittal', c='red')
        self.ax[1].axhline(z, linestyle='--', c='blue', alpha=0.2)
        self.ax[1].axvline(y, linestyle='--', c='green', alpha=0.2)

        # Transaxial
        self.ax[2].imshow(self.scan[z,:,:], cmap='gray_r')
        self.ax[2].set_title('Transaxial', c='blue')
        self.ax[2].axhline(y, linestyle='--', c='green', alpha=0.2)
        self.ax[2].axvline(x, linestyle='--', c='red', alpha=0.2)

        # redraw
        plt.draw()

        return
    
    def onclick(self, event):
        if event.inaxes is None:
            return

        x_, y_ = round(event.xdata), round(event.ydata)
        # GLOBAL coordinates: [z,y,x], plotcoordinates: (x',y')
        if event.inaxes is self.ax[0]:
            # CORONAL
            # x = x'
            # y = y
            # z = y'
            self.coords = (x_, self.coords[1], y_)
        elif event.inaxes is self.ax[1]:
            # SAGITTAL
            # x = x
            # y = x'
            # z = y'
            self.coords = (self.coords[0], x_, y_)
        elif event.inaxes is self.ax[2]:
            # TRANSAXIAL
            # x = x'
            # y = y'
            # z = z
            self.coords = (x_, y_, self.coords[2])
        else:
            return

        self.render()
        return

def plot_divisions(scans, plane, slice_idx):
    # TODO
    return

#wDM#