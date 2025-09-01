import matplotlib.pyplot as plt
import numpy as np

# utility class for colored print outputs
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

##TODO
# - write docs: ctrl + scroll, normal scroll, double click = reset, ALSO: default matpltlib pan/zoom function, right click = change coordinate
# - add circular VOI -> click + space

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
        coords: tuple as defined: (z,y,x)
            initial coordinate to plot
        
        Returns:
        --------
        PlotCoordinate Object, calling plt.show() should display the interactive plot. 

        """
        if coords[0] >= scan.shape[0] or coords[1] >= scan.shape[1] or coords[2] >= scan.shape[2]:
            print(bcolors.FAIL + 'Invalid coordinates' + bcolors.ENDC)
            return

        self.scan = scan
        self.coords = coords
        self.xlim = [None, None, None]
        self.ylim = [None, None, None]

        # init fig, ax
        self.fig, self.ax = plt.subplots(1, 3, figsize=(12, 5))

        # initial render
        self.render()

        # listens for click events
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        # listens for scroll events
        cid = self.fig.canvas.mpl_connect('scroll_event', self.onscroll) 

        return

    def render(self):
        z, y, x = self.coords

        # clear previous ax
        for a in self.ax:
            a.clear()

        vmax = max(float(np.max(self.scan[:,y,:])), float(np.max(self.scan[z,:,:])), float(np.max(self.scan[:,:,x])))

        # coronal
        self.ax[0].imshow(self.scan[:,y,:], cmap='gray_r', vmin=0, vmax=vmax)
        self.ax[0].set_title('Coronal', c='green')
        self.ax[0].axhline(z, linestyle='--', c='blue', alpha=0.2)
        self.ax[0].axvline(x, linestyle='--', c='red', alpha=0.2)

        # view logic
        if not self.xlim[0] is None:
            self.ax[0].set_xlim(self.xlim[0])
        if not self.ylim[0] is None:
            self.ax[0].set_ylim(self.ylim[0])

        # Sagittal
        self.ax[1].imshow(self.scan[:,:,x], cmap='gray_r', vmin=0, vmax=vmax)
        self.ax[1].set_title('Sagittal', c='red')
        self.ax[1].axhline(z, linestyle='--', c='blue', alpha=0.2)
        self.ax[1].axvline(y, linestyle='--', c='green', alpha=0.2)

        # view logic
        if not self.xlim[1] is None:
            self.ax[1].set_xlim(self.xlim[1])
        if not self.ylim[1] is None:
            self.ax[1].set_ylim(self.ylim[1])

        # Transaxial
        self.ax[2].imshow(self.scan[z,:,:], cmap='gray_r', vmin=0, vmax=vmax)
        self.ax[2].set_title('Transaxial', c='blue')
        self.ax[2].axhline(y, linestyle='--', c='green', alpha=0.2)
        self.ax[2].axvline(x, linestyle='--', c='red', alpha=0.2)

        # view logic
        if not self.xlim[2] is None:
            self.ax[2].set_xlim(self.xlim[2])
        if not self.ylim[2] is None:
            self.ax[2].set_ylim(self.ylim[2])

        # redraw
        plt.draw()

        return
    
    def onclick(self, event):
        # if not on one of graphs -> return
        if event.inaxes is None:
            return

        # double click -> reset view
        if event.dblclick:
            if event.inaxes is self.ax[0]:
                # CORONAL
                self.xlim[0] = None
                self.ylim[0] = None
            elif event.inaxes is self.ax[1]:
                # SAGITTAL
                self.xlim[1] = None
                self.ylim[1] = None
            elif event.inaxes is self.ax[2]:
                # TRANSAXIAL
                self.xlim[2] = None
                self.ylim[2] = None
            else:
                return
            
            self.render()
            return
        
        # right click -> change coordinate
        if event.button == 3:
            x_, y_ = round(event.xdata), round(event.ydata)
            # GLOBAL coordinates: [z,y,x], plotcoordinates: (x',y')
            if event.inaxes is self.ax[0]:
                # CORONAL
                # x = x'
                # y = y
                # z = y'
                self.coords = (y_, self.coords[1], x_)
            elif event.inaxes is self.ax[1]:
                # SAGITTAL
                # x = x
                # y = x'
                # z = y'
                self.coords = (y_, x_, self.coords[2])
            elif event.inaxes is self.ax[2]:
                # TRANSAXIAL
                # x = x'
                # y = y'
                # z = z
                self.coords = (self.coords[0], y_, x_)
            else:
                return

            self.render()
            return
        return
    
    def onscroll(self, event):
        # if not on one of graphs -> return
        if event.inaxes is None:
            return
        
        # control scroll -> zoom in / out
        if event.key == 'control':
            if event.button=='up':
                SCROLL_FRACTION = -0.1
            else:
                SCROLL_FRACTION = 0.1

            x_, y_ = round(event.xdata), round(event.ydata)

            # GLOBAL coordinates: [z,y,x], plotcoordinates: (x',y')
            if event.inaxes is self.ax[0]:
                # CORONAL
                curr_width = self.ax[0].get_xlim()[1] - self.ax[0].get_xlim()[0]
                curr_height = self.ax[0].get_ylim()[1] - self.ax[0].get_ylim()[0]

                new_width, new_height = curr_width + round(curr_width*SCROLL_FRACTION), curr_height + round(curr_height*SCROLL_FRACTION)
                self.xlim[0] = (x_ - new_width//2, x_ + new_width//2)
                self.ylim[0] = (y_ - new_height//2, y_ + new_height//2)
            elif event.inaxes is self.ax[1]:
                # SAGITTAL
                curr_width = self.ax[1].get_xlim()[1] - self.ax[1].get_xlim()[0]
                curr_height = self.ax[1].get_ylim()[1] - self.ax[1].get_ylim()[0]

                new_width, new_height = curr_width + round(curr_width*SCROLL_FRACTION), curr_height + round(curr_height*SCROLL_FRACTION)
                self.xlim[1] = (x_ - new_width//2, x_ + new_width//2)
                self.ylim[1] = (y_ - new_height//2, y_ + new_height//2)
            elif event.inaxes is self.ax[2]:
                # TRANSAXIAL
                curr_width = self.ax[2].get_xlim()[1] - self.ax[2].get_xlim()[0]
                curr_height = self.ax[2].get_ylim()[1] - self.ax[2].get_ylim()[0]

                new_width, new_height = curr_width + round(curr_width*SCROLL_FRACTION), curr_height + round(curr_height*SCROLL_FRACTION)
                self.xlim[2] = (x_ - new_width//2, x_ + new_width//2)
                self.ylim[2] = (y_ - new_height//2, y_ + new_height//2)
            else:
                return
            
        # other scroll -> slice up/down
        else:
            # GLOBAL coordinates: [z,y,x], plotcoordinates: (x',y')
            if event.inaxes is self.ax[0]:
                # CORONAL
                if event.button == 'up':
                    if self.coords[1] >= self.scan.shape[1] - 1:
                        return
                    self.coords = (self.coords[0], self.coords[1] + 1, self.coords[2])
                else:
                    if self.coords[1] <= 0:
                        return
                    self.coords = (self.coords[0], self.coords[1] - 1, self.coords[2])
            elif event.inaxes is self.ax[1]:
                # SAGITTAL
                if event.button == 'up':
                    if self.coords[2] >= self.scan.shape[2] - 1:
                        return
                    self.coords = (self.coords[0], self.coords[1], self.coords[2] + 1)
                else:
                    if self.coords[2] <= 0:
                        return
                    self.coords = (self.coords[0], self.coords[1], self.coords[2] - 1)
            elif event.inaxes is self.ax[2]:
                # TRANSAXIAL
                if event.button == 'up':
                    if self.coords[0] >= self.scan.shape[0] - 1:
                        return
                    self.coords = (self.coords[0] + 1, self.coords[1], self.coords[2])
                else:
                    if self.coords[0] <= 0:
                        return
                    self.coords = (self.coords[0] - 1, self.coords[1], self.coords[2])
            else:
                return
            
        self.render()
        return

def plot_divisions(scans, plane, slice_idx, divisions=None):
    fig, axs = plt.subplots(1, scans.shape[0], figsize=(3*scans.shape[0], 5))

    def slice_scan(scan, plane, slice_idx):
        if not (plane == 'Transaxial' or plane == 'Coronal' or plane == 'Transaxial'):
            print(bcolors.FAIL + 'Invalid plane name!' + bcolors.ENDC)
            return
        
        if plane == 'Transaxial':
            if slice_idx >= scan.shape[0]:
                print(bcolors.FAIL + 'Invalid coordinates' + bcolors.ENDC)
                return
            img_arr = scan[slice_idx,:,:]
        elif plane == 'Coronal':
            if slice_idx >= scan.shape[1]:
                print(bcolors.FAIL + 'Invalid coordinates' + bcolors.ENDC)
                return
            img_arr = scan[:, slice_idx, :]
        else:
            if slice_idx >= scan.shape[2]:
                print(bcolors.FAIL + 'Invalid coordinates' + bcolors.ENDC)
                return
            img_arr = scan[:, :, slice_idx]

        return img_arr
    
    vmax = 0

    for i in range(scans.shape[0]):
        arr = slice_scan(scans[i], plane, slice_idx)
        if float(np.max(arr) > vmax):
            vmax = np.max(arr)

    for i in range(scans.shape[0]):
        arr = slice_scan(scans[i], plane, slice_idx)
        axs[i].imshow(arr, cmap='gray_r', vmin=0, vmax=vmax)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        if not divisions is None:
            axs[i].set_title(f'Division: {divisions[i]}')

    plt.tight_layout()

    return fig

#wDM#