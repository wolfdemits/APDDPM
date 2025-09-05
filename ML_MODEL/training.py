import matplotlib.pyplot as plt
import torch
import pathlib
import datetime

from ML_MODEL.UNet import UNet
from ML_MODEL.APD import APD

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