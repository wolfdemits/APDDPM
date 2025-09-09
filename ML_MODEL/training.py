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
NAME_RUN = 'APDDPM_' + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")) 
start_time_run = datetime.datetime.now()
print(bcolors.OKBLUE + f'RUN NAME: {NAME_RUN}' + bcolors.ENDC, flush=True)

####################################################################################

### NETWORK PARAMETERS #############################################################
dim = '2d'
num_in_channels = 1
features_main = [64, 128, 256, 512]
features_skip = [64, 128, 256]
conv_kernel_size = 3
dilation = 1
down_mode = 'maxpool'
up_mode = 'upsample'
normalization = 'batch_norm'
activation = 'PReLU'
attenGate = True
residual_connection = True
time_embed_dim = 64

####################################################################################

### DIFFUSION PARAMETERS ############################################################
# constant for now #
DIFF_BETA = np.sqrt(0.005)   # critical beta
DIFF_T = 100
divisions = [1, 5, 10, 20]
TEMP_DIV = 3

# for now TODO: make dynamic
print(bcolors.OKBLUE + f'Constant Delta: {1/divisions[TEMP_DIV]}' + bcolors.ENDC, flush=True)

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

RANDOM_FLIP = False

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
            RandomFlip = RANDOM_FLIP,
            divisions=divisions)

TrainLoader = torch.utils.data.DataLoader(TrainSet, batch_size=BATCH_SIZE, collate_fn=APD.CollateFn2D(), shuffle=True, drop_last=True)
# -> batch output shape: (T, B, Width, Height), T=time (divisions), B=Batch

# Load validation data
ValSet = APD.Dataset(
            PatientList=ValList,
            PATH = PATH,
            Planes=planes,
            RandomFlip = RANDOM_FLIP,
            divisions=divisions)

ValLoader = torch.utils.data.DataLoader(ValSet, batch_size=BATCH_SIZE, collate_fn=APD.CollateFn2D(), shuffle=True, drop_last=True)
# -> batch output shape: (T, B, Width, Height), T=time (divisions), B=Batch

####################################################################################

### CHECKPOINT SETUP/LOGIC ###############################################################
VIEW_SLICES = range(0,712,10)
VIEW_PATIENTS_TRAIN = [TrainList[0]]
VIEW_PATIENTS_VAL = [ValList[0], ValList[-1]]

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
    residual_connection = residual_connection,
    time_embed_dim=time_embed_dim)

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
epoch_val_loss = []
epoch_train_loss = []

for current_epoch in range(start_epoch, MAX_EPOCHS):
    start_time_epoch = datetime.datetime.now()

    # set model to training mode
    model.train()
    train_loss_per_epoch = 0
    train_batch_number = 0
    train_batch_loss = []

    for trainBatch in TrainLoader:
        train_batch_number += 1

        if (train_batch_number % 1 == 0): # DEBUG: change to 10 or something
            print(bcolors.OKCYAN + f'Train Batch number: {train_batch_number}' + bcolors.ENDC, flush=True)

        # sample division vector
        div_idxs = np.random.randint(0, len(divisions) - 1, size=BATCH_SIZE)

        # for now: all div 20 TODO: make dynamic
        div_idxs = np.ones_like(div_idxs) * TEMP_DIV
        delta = 1/np.array(divisions)[div_idxs]

        # T = f(delta) TODO: find function
        T = DIFF_T

        # load data
        x0 = trainBatch['Images'][0].to(device)

        batch_idx = torch.arange(BATCH_SIZE)
        xT = trainBatch['Images'][div_idxs,batch_idx].to(device)

        # sample time vector
        t = np.random.randint(1, T+1, size=BATCH_SIZE)

        # get x_t and x_t_1 ready
        res_info = {
            # 'division_idxs': div_idxs,
            # 'all_divisions': divisions,
            'division': divisions[div_idxs[0]], # TODO: for now same div
            'plane': 'Coronal', # for now, coronal only
            'patients': TrainList,
        }
        x_t, x_t_1 = APD.diffuse(t=t, T=T, x0=x0, R=(xT-x0), beta=DIFF_BETA, res_info=res_info)

        ## actual training ##
        optimizer.zero_grad()

        if MIXED_PRECISION:
            with torch.amp.autocast('cuda'):
                x0_hat = model(x_t.unsqueeze(1), torch.as_tensor(t/T).to(device), torch.as_tensor(delta).to(device))
                x_t_1_hat, _ = APD.diffuse(t=t-1, T=T, x0=x0_hat.squeeze(1), R=(xT-x0_hat.squeeze(1)), beta=DIFF_BETA, res_info=res_info)

                # Loss
                loss = loss_criterion(x_t_1, x_t_1_hat)

            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

        else:
            x0_hat = model(x_t.unsqueeze(1), torch.as_tensor(t/T).to(device), torch.as_tensor(delta).to(device))
            x_t_1_hat, _ = APD.diffuse(t=t-1, T=T, x0=x0_hat.squeeze(1), R=(xT-x0_hat.squeeze(1)), beta=DIFF_BETA, res_info=res_info)

            # Loss
            loss = loss_criterion(x_t_1, x_t_1_hat)

            loss.backward()
            optimizer.step()
        #####################

        # update metrics
        train_batch_loss.append(loss.item())
        train_loss_per_epoch += train_batch_loss[-1]

        metrics_dict = {
            "epoch": current_epoch,
            "batch": trainBatch,
            "current_batch": train_batch_number,
            "batch_loss": train_batch_loss,
            "xt-1": x_t_1,
            "xt-1_hat": x_t_1_hat,
            "xt": x_t,
            "xT": xT,
            "x0": x0,
            "x0_hat": x0_hat,
            "pathfrac": t/T,
            "delta": delta,
            "view_slices": VIEW_SLICES,
            "view_patients": VIEW_PATIENTS_TRAIN
        }

        FIGUREPATH = RESULTPATH / 'FIGURES' / str(NAME_RUN)
        FIGUREPATH.mkdir(exist_ok=True, parents=True)
        export_metrics(metrics_dict, 'train-batch', FIGUREPATH=FIGUREPATH)

    # epoch ended
    end_time_epoch = datetime.datetime.now()
    time_train_epoch = end_time_epoch - start_time_epoch
    train_loss = train_loss_per_epoch / train_batch_number

    # update metrics
    epoch_train_loss.append(train_loss)

    print(bcolors.WARNING + 'TRAINING: Epoch [{}] \t Run Time = {} \t Loss = {}'.format(current_epoch, str(time_train_epoch), round(train_loss, 6)) + bcolors.ENDC, flush=True)

    ## VALIDATION ##
    start_time_epoch = datetime.datetime.now()

    # set model to eval mode
    model.eval()
    val_loss_per_epoch = 0
    val_batch_number = 0
    val_batch_loss = []

    for valBatch in ValLoader:
        val_batch_number += 1

        if (val_batch_number % 1 == 0):
            print(bcolors.OKCYAN + f'Val Batch number: {val_batch_number}' + bcolors.ENDC, flush=True)

        # sample time vector
        div_idxs = np.random.randint(0, len(divisions) - 1, size=BATCH_SIZE)

        # for now: all div 20 TODO: make dynamic
        div_idxs = np.ones_like(div_idxs) * TEMP_DIV
        delta = 1/np.array(divisions)[div_idxs]

        # T = f(delta) TODO: find function
        T = DIFF_T

        # load data
        x0 = valBatch['Images'][0].to(device)

        batch_idx = torch.arange(BATCH_SIZE)
        xT = valBatch['Images'][div_idxs,batch_idx].to(device)

        # sample time vector
        t = np.random.randint(1, T+1, size=BATCH_SIZE)

        # get x_t and x_t_1 ready
        res_info = {
            # 'division_idxs': div_idxs,
            # 'all_divisions': divisions,
            'division': divisions[div_idxs[0]], # TODO: for now same div
            'plane': 'Coronal', # for now, coronal only
            'patients': ValList,
        }
        x_t, x_t_1 = APD.diffuse(t=t, T=T, x0=x0, R=(xT-x0), beta=DIFF_BETA, res_info=res_info) # for now: T constant

        # disable gradient tracking
        with torch.no_grad():
            x0_hat = model(x_t.unsqueeze(1), torch.as_tensor(t/T).to(device), torch.as_tensor(delta).to(device))
            x_t_1_hat, _ = APD.diffuse(t=t-1, T=T, x0=x0_hat.squeeze(1), R=(xT-x0_hat.squeeze(1)), beta=DIFF_BETA, res_info=res_info)

            # Loss
            loss = loss_criterion(x_t_1, x_t_1_hat)

        # update metrics
        val_batch_loss.append(loss.item())
        val_loss_per_epoch += val_batch_loss[-1]

        metrics_dict = {
            "epoch": current_epoch,
            "batch": valBatch,
            "current_batch": val_batch_number,
            "batch_loss": val_batch_loss,
            "xt-1": x_t_1,
            "xt-1_hat": x_t_1_hat,
            "x0": x0,
            "x0_hat": x0_hat,
            "xt": x_t,
            "xT": xT,
            "pathfrac": t/T,
            "delta": delta,
            "view_slices": VIEW_SLICES,
            "view_patients": VIEW_PATIENTS_VAL
        }

        FIGUREPATH = RESULTPATH / 'FIGURES' / str(NAME_RUN)
        FIGUREPATH.mkdir(exist_ok=True, parents=True)
        export_metrics(metrics_dict, 'val-batch', FIGUREPATH=FIGUREPATH)

    # epoch ended
    end_time_epoch = datetime.datetime.now()
    time_val_epoch = end_time_epoch - start_time_epoch
    val_loss = val_loss_per_epoch / val_batch_number

    # step scheduler
    scheduler.step(val_loss)

    # update metrics
    epoch_val_loss.append(val_loss)

    metrics_dict = {
        "epoch": current_epoch,
        "current_epoch": current_epoch,
        "epoch_val_loss": epoch_val_loss,
        "epoch_train_loss": epoch_train_loss,
        "epoch_val_time": time_val_epoch,
        "epoch_train_time": time_train_epoch,
    }

    FIGUREPATH = RESULTPATH / 'FIGURES' / str(NAME_RUN)
    FIGUREPATH.mkdir(exist_ok=True, parents=True)
    export_metrics(metrics_dict, 'epoch-done', FIGUREPATH=FIGUREPATH)

    print(bcolors.WARNING + 'VALIDATION: Epoch [{}] \t Run Time = {} \t Loss = {}'.format(current_epoch, str(time_val_epoch), round(val_loss, 6)) + bcolors.ENDC, flush=True)

    ## END OF EPOCH ##
    total_run_time = datetime.datetime.now() - start_time_run
    print(bcolors.OKGREEN + '##### EPOCH [{}] COMPLETED -- Run Time = {}'.format(current_epoch, str(total_run_time)) + bcolors.ENDC, flush=True)

    ## COMPARE EPOCH & SAVE ##
    epoch_obj = {
        'inter_epoch_train_loss': train_batch_loss,
        'inter_epoch_val_loss': val_batch_loss
    }

    # write out epoch data object
    path = RESULTPATH / 'EPOCH_DATA'
    path.mkdir(exist_ok=True, parents=True)
    with open(path / f'epoch_data_{current_epoch}.json', 'w') as f:
        json.dump(epoch_obj, f)

    loss_obj = {
        'train_loss': epoch_train_loss,
        'val_loss': epoch_val_loss
    }

    # write out loss object
    path = RESULTPATH / 'EPOCH_DATA'
    path.mkdir(exist_ok=True, parents=True)
    with open(path / f'loss.json', 'w') as f:
        json.dump(loss_obj, f)

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
        
        path = RESULTPATH / 'CHECKPOINTS'
        path.mkdir(exist_ok=True, parents=True)
        torch.save(checkpoint, path / '{}.pt'.format(NAME_RUN))
        print(bcolors.OKBLUE + f'Saved current epoch checkpoint to {path}' + bcolors.ENDC, flush=True)

        # reset early stopping counter
        no_update_since = 0
    else:
        # increment early stopping counter
        no_update_since += 1

    print(bcolors.OKBLUE + 'Best Epoch [{}]'.format(epoch_best) + bcolors.ENDC, flush=True)

    ## EARLY STOPPING ##
    if current_epoch > 2:
        if no_update_since > 3:
            print(bcolors.WARNING + '------------------------------------------------' + bcolors.ENDC, flush=True)  
            print(bcolors.WARNING + 'Early stopping criteria reached after EPOCH [{}]'.format(current_epoch) + bcolors.ENDC, flush=True)
            print(bcolors.WARNING + 'Best Epoch [{}] /// Run Time @Best Epoch {}'.format(epoch_best, runtime_atBestEpoch) + bcolors.ENDC, flush=True)
            
            break

####################################################################################