import matplotlib.pyplot as plt
import torch
from torch.cuda.amp import GradScaler
import pathlib
import datetime
import json
import numpy as np

from ML_MODEL.UNet import UNet
from ML_MODEL.APD import APD
from ML_MODEL.helperfunctions import export_metrics

### QUICK SETUP ############################################################################
LOCAL = True
CHECKPOINT = None
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

## Run name
NAME_RUN = 'Diffusion_' + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) 
start_time_run = datetime.datetime.now()
print(bcolors.OKBLUE + f'RUN NAME: {NAME_RUN}' + bcolors.ENDC, flush=True)

####################################################################################

### NETWORK PARAMETERS #############################################################
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
residual_connection = True

####################################################################################

### DIFFUSION PARAMETERS ############################################################
# constant for now #
DIFF_BETA = np.sqrt(0.005)   # critical beta
DIFF_T = 100
divisions = [1, 5, 10, 20]
DELTA_INV = 20                  # Division = 1/delta
DIVISION_IDX = 3                # div20

print(bcolors.OKBLUE + f'Constant Delta: {1/divisions[DIVISION_IDX]}' + bcolors.ENDC, flush=True)

CONVERGENCE_VERBOSE = True

# convergence check
if CONVERGENCE_VERBOSE:
    critical_value_beta = np.sqrt(8/15 * (DIFF_T**4 - 1)/(DIFF_T**5))
    print(f'{bcolors.WARNING}Critical beta value: {critical_value_beta} {bcolors.ENDC}', flush=True)
    print(f'{bcolors.WARNING}Beta value: {DIFF_BETA} {bcolors.ENDC}', flush=True)

    if (DIFF_BETA > critical_value_beta*10):
        print(f'{bcolors.FAIL}Will not converge!{bcolors.ENDC}', flush=True)
    elif (DIFF_BETA > critical_value_beta):
        print(f'{bcolors.WARNING}WARN: in range of beta_crit, might not converge! (but is possible){bcolors.ENDC}', flush=True)
    else:
        print(bcolors.OKGREEN + 'APD Convergence OK' + bcolors.ENDC, flush=True)
        print(bcolors.OKGREEN + f'Target endpoint standard deviation (beta**2): {DIFF_BETA**2}' + bcolors.ENDC, flush=True)

####################################################################################

### TRAINING HYPERPARAMETERS #######################################################
BATCH_SIZE = 32
LEARN_RATE = 10**-5
DECAY = 1
MAX_EPOCHS = 100

MIXED_PRECISION = False

# in case of local
if LOCAL: BATCH_SIZE = 4
####################################################################################

### DATALOADERS ####################################################################
## Data lists
# load info object
datasets_obj = {}
with open(PATH / 'datasets.json') as f:
    datasets_obj = json.load(f)

TrainList = datasets_obj['train']
ValList = datasets_obj['val']

planes = ["Coronal"]

# Load training data
TrainSet = APD.Dataset(
            PatientList=TrainList,
            PATH = PATH,
            Planes=planes,
            RandomFlip = True,
            divisions=divisions)

TrainLoader = torch.utils.data.DataLoader(TrainSet, batch_size=BATCH_SIZE, collate_fn=APD.CollateFn2D(), shuffle=True)
# -> batch output shape: (T, B, Width, Height), T=time (divisions), B=Batch

# Load validation data
ValSet = APD.Dataset(
            PatientList=ValList,
            PATH = PATH,
            Planes=planes,
            RandomFlip = True,
            divisions=divisions)

ValLoader = torch.utils.data.DataLoader(ValSet, batch_size=BATCH_SIZE, collate_fn=APD.CollateFn2D(), shuffle=True)
# -> batch output shape: (T, B, Width, Height), T=time (divisions), B=Batch
####################################################################################

### CHECKPOINT SETUP/LOGIC ###############################################################
VIEW_SLICES_AMOUNT = 10
VIEW_PATIENTS_TRAIN = [TrainList[0]]
VIEW_PATIENTS_VAL = [ValList[0]]

if not CHECKPOINT is None:
    print(bcolors.OKBLUE + f'Continuing from checkpoint: {CHECKPOINT}', flush=True)
    checkpoint = torch.load(RESULTPATH / 'CHECKPOINTS' / '{}.pt'.format(CHECKPOINT))
    start_epoch = checkpoint['current_epoch'] + 1
    epoch_best = checkpoint['epoch_best']        
    loss_best = checkpoint['loss_best']
    state_best = checkpoint['state_best']
else:
    start_epoch = 0
    epoch_best = 0
    loss_best = np.inf

####################################################################################

### MODEL SETUP ####################################################################
model = UNet(
    dim = dim, 
    num_in_channels = num_in_channels, 
    features_main = features_main, 
    features_skip = features_skip, 
    conv_kernel_size = conv_kernel_size, 
    dilation = dilation,
    down_mode = down_mode,
    up_mode = up_mode, 
    normalization = normalization, 
    activation = activation, 
    attenGate = attenGate, 
    residual_connection = residual_connection)

# optimizer/scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=LEARN_RATE)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,DECAY)

# loss
loss_criterion = torch.nn.MSELoss()

# grad scaler
grad_scaler = GradScaler('cuda')

# if applicable, load checkpoint state dict to model/optimizer/scheduler
if not CHECKPOINT is None:
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# send to device
model = model.to(device)

### TRAINING ######################################################################
no_update_since = 0
epoch_loss = []

for current_epoch in range(start_epoch, MAX_EPOCHS):
    start_time_epoch = datetime.datetime.now()

    # set model to training mode
    model.train()
    train_loss_per_epoch = 0
    train_batch_number = 0
    batch_loss = []

    for trainBatch in TrainLoader:
        train_batch_number += 1

        if (train_batch_number % 1 == 0): # DEBUG: change to 10 or something
            print(bcolors.OKCYAN + f'Train Batch number: {train_batch_number}' + bcolors.ENDC, flush=True)

        # sample delta
        # div = np.floor(rng.uniform(0,3,100000)) + 1
        # delta = 1/divisions[div]
        # T = f(delta)
        T = DIFF_T
        delta = 1/divisions[DIVISION_IDX]

        # load data
        x0 = trainBatch['Images'][0].to(device)
        xT = trainBatch['Images'][DIVISION_IDX].to(device) # for now: DIVISON_IDX constant

        # sample time vector
        t = np.random.randint(1, T+1, size=BATCH_SIZE)

        # get x_t and x_t_1 ready
        res_info = {
            'division': DIVISION_IDX,
            'plane': 'Coronal', # for now, coronal only
            'patients': TrainList,
        }
        x_t, x_t_1 = APD.diffuse(t=t, T=T, x0=x0, R=(xT-x0), beta=DIFF_BETA, res_info=res_info) # for now: T constant

        ## actual training ##
        optimizer.zero_grad()

        if MIXED_PRECISION:
            with torch.amp.autocast('cuda'):
                x0_hat = UNet(x_t) #, t/T, delta)  # TODO: embedding
                x_t_1_hat = APD.diffuse(t=t-1, T=T, x0=x0_hat, R=(xT-x0_hat), beta=DIFF_BETA, res_info=res_info)

                # Loss
                loss = loss_criterion(x_t_1, x_t_1_hat)

            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

        else:
            x0_hat = UNet(x_t) #, t/T, delta)  # TODO: embedding
            x_t_1_hat = APD.diffuse(t=t-1, T=T, x0=x0_hat, R=(xT-x0_hat), beta=DIFF_BETA, res_info=res_info)

            # Loss
            loss = loss_criterion(x_t_1, x_t_1_hat)

            loss.backward()
            optimizer.step()
        #####################

        # update metrics
        batch_loss.append(loss.item())
        train_loss_per_epoch += batch_loss[-1]

        metrics_dict = {
            "current_batch": train_batch_number,
            "batch_loss": batch_loss,
            "xt-1": x_t_1,
            "xt-1_hat": x_t_1_hat,
            "x0": x0,
            "x0_hat": x0_hat,
        }

        export_metrics(metrics_dict, 'train-batch')

    # epoch ended
    end_time_epoch = datetime.datetime.now()
    time_train_epoch = end_time_epoch - start_time_epoch
    train_loss = train_loss_per_epoch / train_batch_number

    # update metrics
    epoch_loss.append(train_loss)

    metrics_dict = {
        "current_epoch": current_epoch,
        "epoch_loss": epoch_loss,
        "epoch_time": time_train_epoch,
    }

    export_metrics(metrics_dict, 'train-epoch')

    print(bcolors.WARNING + 'TRAINING: Epoch [{}] \t Run Time = {} \t Loss = {}'.format(current_epoch, str(time_train_epoch), round(train_loss, 6)) + bcolors.ENDC, flush=True)

    ## VALIDATION ##
    start_time_epoch = datetime.datetime.now()

    # set model to eval mode
    model.eval()
    val_loss_per_epoch = 0
    val_batch_number = 0
    batch_loss = []

    for valBatch in ValLoader:
        val_batch_number += 1

        if (val_batch_number % 1 == 0): # DEBUG: change to 10 or something
            print(bcolors.OKCYAN + f'Val Batch number: {val_batch_number}' + bcolors.ENDC, flush=True)

        # sample delta
        # div = np.floor(rng.uniform(0,3,100000)) + 1
        # delta = 1/divisions[div]
        # T = f(delta)
        T = DIFF_T
        delta = 1/divisions[DIVISION_IDX]

        # load data
        x0 = valBatch['Images'][0].to(device)
        xT = valBatch['Images'][DIVISION_IDX].to(device) # for now: DIVISON_IDX constant

        # sample time vector
        t = np.random.randint(1, T+1, size=BATCH_SIZE)

        # get x_t and x_t_1 ready
        res_info = {
            'division': DIVISION_IDX,
            'plane': 'Coronal', # for now, coronal only
            'patients': ValList,
        }
        x_t, x_t_1 = APD.diffuse(t=t, T=T, x0=x0, R=(xT-x0), beta=DIFF_BETA, res_info=res_info) # for now: T constant

        # disable gradient tracking
        with torch.no_grad():
            x0_hat = UNet(x_t) #, t/T, delta)  # TODO: embedding
            x_t_1_hat = APD.diffuse(t=t-1, T=T, x0=x0_hat, R=(xT-x0_hat), beta=DIFF_BETA, res_info=res_info)

            # Loss
            loss = loss_criterion(x_t_1, x_t_1_hat)

        # update metrics
        batch_loss.append(loss.item())
        val_loss_per_epoch += batch_loss[-1]

        metrics_dict = {
            "current_batch": val_batch_number,
            "batch_loss": batch_loss,
            "xt-1": x_t_1,
            "xt-1_hat": x_t_1_hat,
            "x0": x0,
            "x0_hat": x0_hat,
        }

        export_metrics(metrics_dict, 'val-batch')

    # epoch ended
    end_time_epoch = datetime.datetime.now()
    time_val_epoch = end_time_epoch - start_time_epoch
    val_loss = val_loss_per_epoch / val_batch_number

    # step scheduler
    scheduler.step(val_loss)

    # update metrics
    epoch_loss.append(val_loss)

    metrics_dict = {
        "current_epoch": current_epoch,
        "epoch_loss": epoch_loss,
        "epoch_time": time_val_epoch,
    }

    export_metrics(metrics_dict, 'val-epoch')

    print(bcolors.WARNING + 'VALIDATION: Epoch [{}] \t Run Time = {} \t Loss = {}'.format(current_epoch, str(time_val_epoch), round(val_loss, 6)) + bcolors.ENDC, flush=True)

    ## END OF EPOCH ##
    total_run_time = datetime.datetime.now() - start_time_run
    print(bcolors.OKGREEN + '##### EPOCH [{}] COMPLETED -- Run Time = {}'.format(current_epoch, str(total_run_time)) + bcolors.ENDC, flush=True)
    print(bcolors.OKBLUE + 'Best Epoch [{}]'.format(epoch_best) + bcolors.ENDC, flush=True)

    ## COMPARE EPOCH & SAVE ##
    if val_loss < loss_best:
        state_best = model.state_dict()
        epoch_best = current_epoch
        loss_best = val_loss

        runtime_atBestEpoch = total_run_time

        # save checkpoint
        checkpoint = {
            'current_epoch': current_epoch,
            'epoch_best': epoch_best,
            'loss_best': loss_best,
            'state_best': state_best,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()}
        
        path = RESULTPATH / 'CHECKPOINTS' / '{}.pt'.format(NAME_RUN)
        torch.save(checkpoint, path)
        print(bcolors.OKBLUE + f'Saved current epoch checkpoint to {path}' + bcolors.ENDC, flush=True)

        # reset early stopping counter
        no_update_since = 0
    else:
        # increment early stopping counter
        no_update_since += 1

    ## EARLY STOPPING ##
    if current_epoch > 2:
        if no_update_since > 3:
            print(bcolors.WARNING + '------------------------------------------------' + bcolors.ENDC, flush=True)  
            print(bcolors.WARNING + 'Early stopping criteria reached after EPOCH [{}]'.format(current_epoch) + bcolors.ENDC, flush=True)
            print(bcolors.WARNING + 'Best Epoch [{}] /// Run Time @Best Epoch {}'.format(epoch_best, runtime_atBestEpoch) + bcolors.ENDC, flush=True)
            
            break

####################################################################################