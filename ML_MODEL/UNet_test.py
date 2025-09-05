import matplotlib.pyplot as plt
import torch
import pathlib
import datetime

from ML_MODEL.UNet import UNet

LOCAL = True

if LOCAL:
    PATH = pathlib.Path('./')
else:
    PATH = pathlib.Path('/kyukon/data/gent/vo/000/gvo00006/vsc48955/APDDPM')

RESULTPATH = PATH / 'RESULTS'

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
    print('Using CUDA')
    device = torch.device('cuda')
else:        
    print('Using cpu')
    device  = torch.device('cpu')

random2Dtensor = torch.rand(32, 1, 93, 149)

gener = UNet(
    dim = '2d', 
    num_in_channels = 1, 
    features_main = [64, 128, 256, 512], 
    features_skip = [64, 128, 256], 
    conv_kernel_size = 3, 
    dilation = 1,
    down_mode = 'maxpool',
    up_mode = 'upsample', 
    normalization = 'batch_norm', 
    activation = 'PReLU', 
    attenGate = True, 
    residual_connection = True)

output = gener(random2Dtensor)
print(output.shape)

filename = f'UNet_test-{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.png'

# plot
fig, ax = plt.subplots(1,2, figsize=(12,4))
ax[0].imshow(random2Dtensor[0,0],cmap="gray_r")
ax[0].set_title('HN')
ax[1].imshow(output[0,0].detach().numpy(),cmap="gray_r")
ax[1].set_title('LN')

fig.savefig(str(RESULTPATH / 'UNET_TEST' / filename))

print(bcolors.OKCYAN + f'Saved fig to: {str(RESULTPATH / 'UNET_TEST' / filename)}' + bcolors.ENDC)