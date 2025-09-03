import numpy as np
import zarr
import json
import pathlib
from scipy.ndimage import binary_erosion

inpath = pathlib.Path("./DATA")
outpath = pathlib.Path("./DATA_PREPROCESSED")

from PREPROCESSING.datamanager import Datamanager
datamanager = Datamanager(DATAPATH=inpath)

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

def crop(scan, bb=None, thresh=0.01):
    if (bb == None):
 
        mask = scan > thresh
        n_erosions = 5
        
        for _ in range(n_erosions):
            mask = binary_erosion(mask)
 
        nonzero_idx = np.nonzero(mask)

        if (len(nonzero_idx[0]) != 0):
            bb = [
                [nonzero_idx[0].min() - n_erosions, nonzero_idx[0].max() + n_erosions],
                [nonzero_idx[1].min() - n_erosions, nonzero_idx[1].max() + n_erosions],
                [nonzero_idx[2].min() - n_erosions, nonzero_idx[2].max() + n_erosions]]
            
        else:
            bb = [
                [0, scan.shape[0]],
                [0, scan.shape[1]],
                [0, scan.shape[2]]]
 
    cropped_image = scan[bb[0][0]:bb[0][1], bb[1][0]:bb[1][1], bb[2][0]:bb[2][1]]
 
    return cropped_image, bb

patients = datamanager.available_ids

scan = datamanager.scan

# open zarr root
root = zarr.open_group(str(outpath / 'PATIENTS'), mode='a')

data_obj = {}

for patient in patients:
    print(bcolors.OKCYAN + f'Processing patient: {patient}' + bcolors.ENDC)
    scan, divisions = datamanager.load_scan(patient)

    # create patient group
    patient_group = root.require_group(patient)

    # load info object
    info_obj = {}
    with open(inpath / 'PATIENTS' / patient / 'info.json') as f:
            info_obj = json.load(f)

    # update data_obj
    tracer = info_obj['tracer']
    if not tracer in data_obj:
        data_obj[tracer] = [patient]
    else:
        data_obj[tracer].append(patient)

    # write info object
    with open(outpath / 'PATIENTS' / patient / 'info.json', 'w') as f:
            json.dump(info_obj, f)

    # Iterate over 3 axis
    for plane in ['Coronal', 'Sagittal', 'Transaxial']:
        # create plane group
        plane_group = patient_group.require_group(plane)

        # define thresholds
        tracer = info_obj['tracer']
        if tracer == 'FDG' or tracer == 'FNOS' or tracer == 'CFN':
            threshold = 0.5
        elif tracer == 'FTT':
            threshold = 0.3
        elif tracer == 'FES':
            threshold = 0.2
        elif tracer == 'Zr89':
            threshold = 0.1
        else:
            threshold = 0.5
            print(bcolors.WARNING + 'Invalid tracer -> threshold not defined properly' + bcolors.ENDC)

        # crop scan in 3D based on full image
        # binary erosion, 5 iterations and threshold defined above
        _, bb = crop(scan[0], thresh=threshold)

        for i, div in enumerate(divisions):
            # create div group
            div_group = plane_group.require_group(f'div{str(div)}')

            img3D = scan[i][bb[0][0]:bb[0][1], bb[1][0]:bb[1][1], bb[2][0]:bb[2][1]]

            if plane == 'Coronal':
                axis_range = img3D.shape[1]
            elif plane == 'Sagittal':
                axis_range = img3D.shape[2]
            elif plane == 'Transaxial':
                axis_range = img3D.shape[0]

            for idx in range(axis_range):
                if plane == 'Coronal':
                    img = img3D[:,idx,:]
                elif plane == 'Sagittal':
                    img = img3D[:,:,idx]
                elif plane == 'Transaxial':
                    img = img3D[idx,:,:]

                arr = div_group.create_array(str(idx), shape=img.shape , chunks=img.shape, dtype='float32')
                arr[:] = img

# write out data.json
with open(outpath / 'data.json', 'w') as f:
    json.dump(data_obj, f)

# copy datasets.json
datasets_obj = {}
with open(inpath / 'datasets.json') as f:
    datasets_obj = json.load(f)

with open(outpath / 'datasets.json', 'w') as f:
    json.dump(datasets_obj, f)

# final data.json file
print(bcolors.OKGREEN + 'Pre-processing completed' + bcolors.ENDC)