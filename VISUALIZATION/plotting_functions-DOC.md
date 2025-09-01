# PLOTTING_FUNCTIONS DOCUMENTATION

### COORDINATE SYSTEM USED: 
Global coordinate system used: arr = [z,y,x] <br>
z: 0 -> 712 = Cranial -> Caudal <br>
y: 0 -> 300 = Ventral -> Dorsal <br>
x: 0 -> 300 = Anatomical Left -> Anatomical Right <br>

## PlotCoordinate:
Interactive plot to display 3D scan as seen from 3 anatomical planes.<br>
<br>
Parameters:<br>
-----------<br>
scan: 3D numpy array<br>
    3D scan to display<br>
coords: tuple as defined: (z,y,x)<br>
    initial coordinate to plot<br>
<br>
Returns:<br>
--------<br>
PlotCoordinate Object, calling plt.show() should display the interactive plot. <br>
<br>
### Example usage:<br>
from plotting_functions import PlotCoordinate

plotcoordinate = PlotCoordinate(scan=[3D numpy arr], coords=([z],[y],[x]))<br>
plt.show()<br>

## Usage of interactive plot:
- Pan/Zoom: use default matplotlib functions. An additional zoom function can be used by ctrl + scrolling to zoom in/out. <br>
- Double click: reset viewbox
- Scroll: move slice up/down
- Right Click: change coordinate to mouse position