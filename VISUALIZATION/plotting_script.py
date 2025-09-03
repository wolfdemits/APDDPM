import pathlib
import matplotlib.pyplot as plt

from VISUALIZATION.plotting_functions import PlotCoordinate, plot_divisions, view_preprocessed, view_residual
from PREPROCESSING.datamanager import Datamanager

# path relative to terminal path
DATAPATH=pathlib.Path('./DATA')
PATIENT_ID = 'r001'

datamanager = Datamanager(DATAPATH)
scan, divisions = datamanager.load_scan(PATIENT_ID)

# plotcoordinate = PlotCoordinate(scan=scan[0], coords=(100,150,150), roi_diam=10, print_mean_SUV=True, SUV_threshold=1.5)

# fig = plot_divisions(scan, plane='Coronal', slice_idx=150, divisions=divisions)

# fig = view_preprocessed('r001', 'Coronal', 0)

fig = view_residual('r001', 'Coronal', 100)

plt.show()