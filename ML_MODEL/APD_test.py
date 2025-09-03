import matplotlib.pyplot as plt
import pathlib
import torch
import numpy as np
import json

from ML_MODEL.APD import APD

LOCAL = True

if LOCAL:
    PATH = pathlib.Path('./')
else:
    PATH = pathlib.Path('/kyukon/data/gent/vo/000/gvo00006/vsc48955/APDDPM')

DATAPATH = PATH / 'DATA_PREPROCESSED'

# init APD instance
APD = APD(DATAPATH)

# batch size
B = 10

# dataloader
# load info object
datasets_obj = {}
with open(PATH / 'datasets.json') as f:
    datasets_obj = json.load(f)

# Load training data
TrainSet = APD.Dataset(
            PatientList=datasets_obj['train'],
            DATAPATH = DATAPATH,
            Planes=["Coronal"],
            RandomFlip = False,
            divisions=[1, 5, 10, 20])

TrainLoader = torch.utils.data.DataLoader(TrainSet, batch_size=B, collate_fn=APD.CollateFn2D(), shuffle=True)
# -> batch output shape: (T, B, Width, Height), T=time (divisions), B=Batch

# # Load validation data
# ValSet = APD.Dataset(
#             PatientList=datasets_obj['val'],
#             DATAPATH = DATAPATH,
#             Planes=["Coronal"],
#             RandomFlip = False,
#             divisions=[1, 5, 10, 20])

# ValLoader = torch.utils.data.DataLoader(ValSet, batch_size=B, collate_fn=APD.CollateFn2D(), shuffle=True)

# # Load test data
# TestSet = APD.Dataset(
#             PatientList=datasets_obj['test'],
#             DATAPATH = DATAPATH,
#             Planes=["Coronal"],
#             RandomFlip = False,
#             divisions=[1, 5, 10, 20])

# TestLoader = torch.utils.data.DataLoader(TestSet, batch_size=B, collate_fn=APD.CollateFn2D(), shuffle=True)

# inputs
T = 4
t = torch.ones((B), dtype=torch.uint8) * 4
beta = np.sqrt(0.01)

# training batch
trainBatch = next(iter(TrainLoader))

# run forward diffusion
x0 = trainBatch['Images'][0]
xT = trainBatch['Images'][3]
x_t = APD.diffuse(t=t, T=T, x0=x0, R=(xT-x0), beta=beta, convergence_verbose=True)

vmax = max(torch.max(x0), torch.max(xT), torch.max(x_t)) / 2

vmax_frac = 0.1

# plot
fig, ax = plt.subplots(1,3, figsize=(12,4))
ax[0].imshow(x_t[0],cmap="gray_r", vmin=0, vmax=vmax*vmax_frac)
ax[0].set_title('x_t')
ax[1].imshow(x0[0],cmap="gray_r", vmin=0, vmax=vmax*vmax_frac)
ax[1].set_title('x_0')
ax[2].imshow(xT[0],cmap="gray_r", vmin=0, vmax=vmax*vmax_frac)
ax[2].set_title('x_T')

plt.show()