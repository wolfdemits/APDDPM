import pathlib
import matplotlib.pyplot as plt

from VISUALIZATION.plotting_functions import plot_coordinate
from PREPROCESSING.datamanager import Datamanager

# path relative to terminal path
DATAPATH=pathlib.Path('./DATA')
PATIENT_ID = 'r001'

datamanager = Datamanager(DATAPATH)
scan, _ = datamanager.load_scan(PATIENT_ID)

fig = plot_coordinate(scan=scan[0], coords=(100,150,150))
plt.show()