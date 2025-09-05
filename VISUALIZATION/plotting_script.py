import pathlib
import matplotlib.pyplot as plt

from VISUALIZATION.plotting_functions import PlotCoordinate, plot_divisions, view_preprocessed, view_residual
from PREPROCESSING.datamanager import Datamanager

# path relative to terminal path
LOCAL = True

if LOCAL:
    PATH = pathlib.Path('./')
else:
    PATH = pathlib.Path('/kyukon/data/gent/vo/000/gvo00006/vsc48955/APDDPM')

RESULTPATH = PATH / 'RESULTS'
DATAPATH = PATH / 'DATA'
DATAPATH_PREPROCESSED = PATH / 'DATA_PREPROCESSED'

PATIENT_ID = 'r177'

datamanager = Datamanager(DATAPATH)
scan, divisions = datamanager.load_scan(PATIENT_ID)

# plotcoordinate = PlotCoordinate(scan=scan[0], coords=(100,150,150), roi_diam=10, print_mean_SUV=True, SUV_threshold=1.5)

# fig = plot_divisions(scan, plane='Coronal', slice_idx=150, divisions=divisions)

fig = view_preprocessed(PATIENT_ID, 'Coronal', 100, DATAPATH=DATAPATH_PREPROCESSED)

# fig = view_residual('r001', 'Coronal', 100)

if LOCAL:
    plt.show()
else:
    path = RESULTPATH / 'PLOT_SCRIPT' / PATIENT_ID
    fig.savefig(str(path))