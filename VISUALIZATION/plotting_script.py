import pathlib
import matplotlib.pyplot as plt

from VISUALIZATION.plotting_functions import PlotCoordinate
from PREPROCESSING.datamanager import Datamanager

# path relative to terminal path
DATAPATH=pathlib.Path('./DATA')
PATIENT_ID = 'r001'

datamanager = Datamanager(DATAPATH)
scan, _ = datamanager.load_scan(PATIENT_ID)

plotcoordinate = PlotCoordinate(scan=scan[0], coords=(100,150,150))

plt.show()