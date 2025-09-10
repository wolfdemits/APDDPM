import matplotlib.pyplot as plt
import torch
import pathlib
import json
import numpy as np
import zarr

from ML_MODEL.UNet import UNet
from ML_MODEL.APD import APD

### QUICK SETUP ############################################################################
LOCAL = True
CHECKPOINT = 'APDDPM_2025-09-08_22-36-30'

PATIENT = 'r001'
PLANE = 'Coronal'
SLICE_IDX = 100
DIV = 20
############################################################################################

### GENERAL SETUP ##########################################################################
# Path setup
if LOCAL:
    PATH = pathlib.Path('./')
else:
    PATH = pathlib.Path('/kyukon/data/gent/vo/000/gvo00006/vsc48955/APDDPM')

RESULTPATH = PATH / 'RESULTS'

# init APD instance
APD = APD(PATH)

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

# setup device
if torch.cuda.is_available():
    print(bcolors.OKCYAN  + 'Using CUDA' + bcolors.ENDC, flush=True)
    device = torch.device('cuda')
else:        
    print(bcolors.OKCYAN + 'Using cpu' + bcolors.ENDC, flush=True)
    device  = torch.device('cpu')

####################################################################################

### LOAD CHECKPOINT ################################################################
print(bcolors.OKBLUE + f'Loading checkpoint: {CHECKPOINT}' + bcolors.ENDC, flush=True)
checkpoint = torch.load(RESULTPATH / 'CHECKPOINTS' / '{}.pt'.format(CHECKPOINT))
start_epoch = checkpoint['current_epoch'] + 1
epoch_best = checkpoint['epoch_best']        
loss_best = checkpoint['loss_best']
state_best = checkpoint['state_best']

# APD_state = checkpoint['APD_state']
# network_hyperparam = checkpoint['network_hyperparam']
# training_hyperparam = checkpoint['training_hyperparam']

## TEMP
APD_state = {
    "DIFF_BETA": np.sqrt(0.005),
    "DIFF_T": 100,
    "divisions": [1, 5, 10, 20],
}

# Network hyperparameters
network_hyperparam = {
    "dim": '2d',
    "num_in_channels": 1,
    "features_main": [64, 128, 256, 512],
    "features_skip": [64, 128, 256],
    "conv_kernel_size": 3,
    "dilation": 1,
    "down_mode": 'maxpool',
    "up_mode": 'upsample',
    "normalization": 'batch_norm',
    "activation": 'PReLU',
    "attenGate": True,
    "residual_connection": True,
    "time_embed_dim": 64,
}

# training hyperparam
training_hyperparam = {
    "BATCH_SIZE": 32,
    "LEARN_RATE": 10**-5,
    "DECAY": 1,
    "MAX_EPOCHS": 100,
    "MIXED_PRECISION": False,
    "RANDOM_FLIP": True,
}
## TEMP

####################################################################################

### DIFFUSION PARAMETERS ############################################################
DIFF_BETA = APD_state['DIFF_BETA']
DIFF_T = APD_state['DIFF_T']
divisions = APD_state['divisions']

####################################################################################

### MODEL SETUP ####################################################################
model = UNet(
    dim = network_hyperparam['dim'], 
    num_in_channels = network_hyperparam['num_in_channels'], 
    features_main = network_hyperparam['features_main'], 
    features_skip = network_hyperparam['features_skip'], 
    conv_kernel_size = network_hyperparam['conv_kernel_size'], 
    dilation = network_hyperparam['dilation'],
    down_mode = network_hyperparam['down_mode'],
    up_mode = network_hyperparam['up_mode'], 
    normalization = network_hyperparam['normalization'], 
    activation = network_hyperparam['activation'], 
    attenGate = network_hyperparam['attenGate'], 
    residual_connection = network_hyperparam['residual_connection'],
    time_embed_dim = network_hyperparam['time_embed_dim'])

model.load_state_dict(checkpoint['model_state_dict'])

# send to device
model = model.to(device)

####################################################################################

# function to get image pair
def get_pair(patient, plane, slice_idx, div):
        DATAPATH = PATH / 'DATA_PREPROCESSED'
        root = zarr.open_group(str(DATAPATH / 'PATIENTS'), mode='r')

        try:
            x0 = root[patient][plane]['div' + str(1)][str(slice_idx)][:]
        except:
            print(bcolors.FAIL + f'Unable to find specified group in dataset: {patient} -> {plane} -> div{1} -> {slice_idx}' + bcolors.ENDC)

        try:
            xT = root[patient][plane]['div' + str(div)][str(slice_idx)][:]
        except:
            print(bcolors.FAIL + f'Unable to find specified group in dataset: {patient} -> {plane} -> div{div} -> {slice_idx}' + bcolors.ENDC)

        #convert to torch
        x0 = torch.as_tensor(x0, dtype=torch.float32)
        xT = torch.as_tensor(xT, dtype=torch.float32)

        item = {'x0': x0.unsqueeze(0), 'xT': xT.unsqueeze(0), 'Patient': np.array([patient]), 'Plane': np.array([plane]), 'SliceIndex': np.array([slice_idx])}

        return item

### INFERENCE ######################################################################
BATCH_SIZE = 1

batch = get_pair(PATIENT, PLANE, SLICE_IDX, DIV)
x0 = batch['x0'].to(device)
xT = batch['xT'].to(device)
batch_planes = batch['Plane']

# diffusion
div_idxs = np.array([3])
delta = 1/np.array(divisions)[div_idxs]

T = DIFF_T

# time vector
t = np.array([T])

# get x_t and x_t_1 ready
res_info = {
    'division_idxs': div_idxs,
    'all_divisions': divisions,
    'batch_planes': batch_planes,
    'all_planes': [PLANE],
    'patients': [PATIENT],
    'random_flip': True,
}
x_t, x_t_1 = APD.diffuse(t=t, T=T, x0=x0, R=(xT-x0), beta=DIFF_BETA, res_info=res_info) # for now: T constant

# loss
loss_criterion = torch.nn.MSELoss()

# disable gradient tracking
with torch.no_grad():
    x0_hat = model(x_t.unsqueeze(1), torch.as_tensor(t/T).to(device), torch.as_tensor(delta).to(device))
    x_t_1_hat, _ = APD.diffuse(t=t-1, T=T, x0=x0_hat.squeeze(1), R=(xT-x0_hat.squeeze(1)), beta=DIFF_BETA, res_info=res_info)

    # Loss
    loss = loss_criterion(x_t_1, x_t_1_hat)

## plot ##
VMAX_PERCENTILE = 99

img1 = x_t.squeeze().cpu()
img2 = x_t_1_hat.squeeze().cpu()
img3 = x_t_1.squeeze().cpu()
img4 = x0_hat.squeeze().cpu()
img5 = x0.squeeze().cpu()
img6 = xT.squeeze().cpu()

vmax = np.percentile(np.array([img1.detach(),img2.detach(),img3.detach(),img4.detach(),img5.detach(),img6.detach()]), VMAX_PERCENTILE)

fig, ax = plt.subplots(2, 3, figsize=(11, 7))
ax[0,0].imshow(img1, cmap='gray_r', vmin=0, vmax=vmax)
ax[0,0].set_title('IN: xt')
ax[0,1].imshow(img2.detach(), cmap='gray_r', vmin=0, vmax=vmax)
ax[0,1].set_title('OUT: x^_t-1')
ax[0,2].imshow(img3.detach(), cmap='gray_r', vmin=0, vmax=vmax)
ax[0,2].set_title('TARGET: x_t-1')
ax[1,0].imshow(img4.detach(), cmap='gray_r', vmin=0, vmax=vmax)
ax[1,0].set_title('UNet: x0^')
ax[1,1].imshow(img5.detach(), cmap='gray_r', vmin=0, vmax=vmax)
ax[1,1].set_title('HD: x0')
ax[1,2].imshow(img6.detach(), cmap='gray_r', vmin=0, vmax=vmax)
ax[1,2].set_title('LD: xT')
fig.suptitle(f'Patient {PATIENT}, slice: {SLICE_IDX}, \n timestep fraction: {t[0]/T}, dose delta: {delta[0]}, loss: {loss:.4f}')

if LOCAL:
    plt.show()
else:
    path = PATH / 'RESULTS' / 'INFERENCE'
    path.mkdir(exist_ok=True, parents=True)
    fig.savefig(str(path  / f'{PATIENT}_{PLANE}_{SLICE_IDX}_{t[0]}_{delta[0]}.png'))
    print(bcolors.OKGREEN + f"Succesfully saved result to {str(path / f'{PATIENT}_{PLANE}_{SLICE_IDX}_{t[0]}_{delta[0]}.png')}" + bcolors.ENDC, flush=True)
    plt.close()

####################################################################################