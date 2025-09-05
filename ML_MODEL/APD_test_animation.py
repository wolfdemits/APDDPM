from ML_MODEL.APD import APD
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import json
import torch
from matplotlib import animation
import datetime

LOCAL = True

if LOCAL:
    PATH = pathlib.Path('./')
else:
    PATH = pathlib.Path('/kyukon/data/gent/vo/000/gvo00006/vsc48955/APDDPM')

RESULTPATH = PATH / 'RESULTS'

# init instance
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

resIterator = None

def diffuse(t, T, x0, R, beta, res_info=None, convergence_verbose=False):
    """
    Apply Forward Anchored Path Diffusion Process to image.

    Note: x0 and R should always be normalized!

    Uses time schedule function: f(t) = 4*alpha_t(1-alpha_t), alpha_t = t/T

    Parameters:
    -----------
    t: torch time tensor: (B,)
        Stop forward process at time point t. (inclusive)
    T: int
        Amount of total diffusion steps in chain.
    x0: torch image tensor: (B, Width, Height)
        High-Dose images (ground truth or estimate).
    R: torch image tensor: (B, Width, Height)
        Low-Dose images.
    beta: float
        End point variance.
    convergence_verbose: Bool
        Wether or not to print out convergence warnings. (default: False)

    Returns:
    --------
    xt: torch image tensor: (B, Width, Height)
        Requested image at time vector t after forward diffusion. 
    """

    # TODO - make more memory efficient eg: stop at time vector

    device = x0.device
    # batch size
    B = x0.shape[0]

    # determine step-wise variance to reach the desired end point variance (beta)
    sigma =  np.sqrt(15/8 * (T**3)/(T**4 - 1)) * beta

    # Noise schedule function
    f = lambda t: 4* t/T * (1-t/T)

    # convergence check
    if convergence_verbose:
        critical_value_beta = np.sqrt(8/15 * (T**4 - 1)/(T**5))
        print(f'{bcolors.WARNING}Critical beta value: {critical_value_beta} {bcolors.ENDC}', flush=True)
        print(f'{bcolors.WARNING}Beta value: {beta} {bcolors.ENDC}', flush=True)
        if (beta > critical_value_beta*10):
            print(f'{bcolors.FAIL}Will not converge!{bcolors.ENDC}', flush=True)
        elif (beta > critical_value_beta):
            print(f'{bcolors.WARNING}WARN: in range of beta_crit, might not converge! (but is possible){bcolors.ENDC}', flush=True)

    # prepare torch tensors: (T, B, Width, Height) -> T = time axis, B = batch axis
    images = torch.zeros((T+1, B, x0.shape[1], x0.shape[2])).to(device)
    images[0,:,:,:] = x0

    # sample epsilon: (T, B, 300, 300) (T=time) -> N samples = T*B
    # Note epsilons are already variance-normalized

    global resIterator
    if not res_info is None:
        # use residuals, else use gaussian noise
        residualSet = APD.ResidualSet(PatientList=res_info['patients'], plane=res_info['plane'], division=res_info['division'], PATH=PATH, RandomFlip=True)
        sampler = torch.utils.data.RandomSampler(residualSet, replacement=True)
        resLoader = torch.utils.data.DataLoader(residualSet, sampler=sampler, batch_size=B)
        resIterator = iter(resLoader)

    # 1 diffusion step
    def step(xt_1, t):
        global resIterator
        if resIterator is None:
            # gaussian if no resloader created
            epsilon = torch.normal(0,1, size=xt_1.shape).to(device)

        else:
            try:
                epsilon = next(resIterator)['Residual'].to(device)
            except StopIteration:
                resIterator = iter(resLoader)
                epsilon = next(resIterator)['Residual'].to(device)

        # pad noise according to image to allow addition of noise
        W, H = xt_1.shape[1], xt_1.shape[2]
        X, Y = epsilon.shape[1]//2, epsilon.shape[2]//2
        
        epsilon = epsilon[:, X - W//2 : X + (W - W//2), Y - H//2 : Y + (H - H//2)]

        # push everything to device
        xt = xt_1 + 1/T * R + sigma * f(t) * epsilon
        return xt
    
    # run whole chain
    for i in range(T):
        images[i+1,:,:,:] = step(xt_1=images[i,:,:,:], t=i+1)

    return images

# batch size
B = 1

# dataloader
# load info object
datasets_obj = {}
with open(PATH / 'datasets.json') as f:
    datasets_obj = json.load(f)

TrainList = datasets_obj['train']

# Load training data
TrainSet = APD.Dataset(
            PatientList=TrainList,
            PATH = PATH,
            Planes=["Coronal"],
            RandomFlip = False,
            divisions=[1, 5, 10, 20])

TrainLoader = torch.utils.data.DataLoader(TrainSet, batch_size=B, collate_fn=APD.CollateFn2D(), shuffle=True)
# -> batch output shape: (T, B, Width, Height), T=time (divisions), B=Batch

# inputs
if LOCAL:
    T = 20
else: 
    T = 100
t = torch.ones((B), dtype=torch.uint8) * 4
beta = np.sqrt(0.005)
division = 20
division_idx = 3 # div20

print(bcolors.OKCYAN + f'Target endpoint standard deviation (beta**2): {beta**2}' + bcolors.ENDC)

# amount of samples
N = 20

# training batch
trainBatch = next(iter(TrainLoader))

# setup device
if torch.cuda.is_available():
    print('Using CUDA')
    device = torch.device('cuda')
else:        
    print('Using cpu')
    device  = torch.device('cpu')

# run forward diffusion
x0 = trainBatch['Images'][0].expand(N, -1, -1).to(device)
xT = trainBatch['Images'][division_idx].expand(N, -1, -1).to(device)
res_info = {
    'division': division,
    'plane': 'Coronal',
    'patients': TrainList,
}
images = diffuse(t=t, T=T, x0=x0, R=(xT-x0), beta=beta, convergence_verbose=True, res_info=res_info)

# back to cpu
images = images.cpu()
x0 = x0.cpu()
xT = xT.cpu()

figA, axA = plt.subplots(2, 3,figsize=(15, 10), num='Anchored Path Diffusion Visualization')

mean_sqdiff_arr = np.zeros(T)
std_arr = np.zeros(T)

def update_plot(t):
    axA[0,0].clear()
    axA[0,1].clear()
    axA[0,2].clear()
    axA[1,0].clear()
    axA[1,1].clear()
    axA[1,2].clear()

    # visual -> first sample
    vmax = max(torch.max(x0[0]), torch.max(xT[0]), torch.max(images[t][0]))*0.5

    axA[0,0].imshow(x0[0], cmap='gray_r', vmin=0, vmax=vmax)
    axA[0,0].set_title('x0')
    axA[0,1].imshow(images[t][0], cmap='gray_r', vmin=0, vmax=vmax)
    axA[0,1].set_title('xt')
    axA[0,2].imshow(xT[0], cmap='gray_r', vmin=0, vmax=vmax)
    axA[0,2].set_title('xT')
    axA[1,0].imshow(xT[0] - images[t][0], cmap='gray_r', vmin=0, vmax=vmax)
    axA[1,0].set_title(f'difference xT and xt')

    mean_sqdiff_arr[t] = torch.sum((xT[0] - torch.mean(images[t], axis=0))**2)
    axA[1,1].plot(np.arange(len(mean_sqdiff_arr[:t])), mean_sqdiff_arr[:t])
    axA[1,1].set_title(f'Mean Sum Square difference between \n xT and xt')

    # std between samples
    std_arr[t] = torch.mean(torch.std(images[t], axis=0)**2)
    axA[1,2].plot(np.arange(len(std_arr[:t])), std_arr[:t])
    axA[1,2].set_title(f'Standard deviation between \n samples of same diffusion process')
    return

ani = animation.FuncAnimation(fig=figA, func=update_plot, frames=T, interval=500, blit = False, repeat=True)

if LOCAL:
    plt.show()
else:
    writervideo = animation.FFMpegWriter(fps=10)
    filename = f'test_animation-{datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}.mp4'
    ani.save(str(RESULTPATH / 'APD_ANIMATION' / filename), writer=writervideo)
    plt.close()

    print(bcolors.OKCYAN + f'Succesfully written to: {filename}' + bcolors.ENDC, flush=True)

print(bcolors.OKCYAN + f'Final mean sqdiff: {mean_sqdiff_arr[-1]}' + bcolors.ENDC, flush=True)
print(bcolors.OKCYAN + f'Final mean sqdiff per pixel: {mean_sqdiff_arr[-1] / (x0[0].shape[0] * x0[0].shape[1])}' + bcolors.ENDC, flush=True)
print(bcolors.OKCYAN + f'Final stdev: {std_arr[-1]}' + bcolors.ENDC, flush=True)