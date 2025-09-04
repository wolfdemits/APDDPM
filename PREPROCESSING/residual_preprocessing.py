import numpy as np
import zarr
import json
import pathlib

LOCAL = True

if LOCAL:
    PATH = pathlib.Path("./")
else:
    PATH = pathlib.Path('/kyukon/data/gent/vo/000/gvo00006/vsc48955/APDDPM')

inpath = PATH / 'DATA'
outpath = PATH / 'RESIDUALS'

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

patients = datamanager.available_ids

scan = datamanager.scan

# open zarr root
root = zarr.open_group(str(outpath / 'PATIENTS'), mode='a')

data_obj = {}

for patient in patients[:5]: # only first 5 for local
    print(bcolors.OKCYAN + f'Processing patient: {patient}' + bcolors.ENDC, flush=True)
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

    info_obj['divisions'] = info_obj['divisions'][1:]

    # write info object
    with open(outpath / 'PATIENTS' / patient / 'info.json', 'w') as f:
            json.dump(info_obj, f)

    # Iterate over 3 axis
    for plane in ['Coronal', 'Sagittal', 'Transaxial']:
        # create plane group
        plane_group = patient_group.require_group(plane)

        for i, div in enumerate(divisions):
            # if full image -> don't compute residual 
            if i == 0:
                continue

            # create div group
            div_group = plane_group.require_group(f'div{str(div)}')

            if plane == 'Coronal':
                axis_range = scan[0].shape[1]
            elif plane == 'Sagittal':
                axis_range = scan[0].shape[2]
            elif plane == 'Transaxial':
                axis_range = scan[0].shape[0]

            res_arr = []

            for idx in range(axis_range):
                if plane == 'Coronal':
                    res_arr.append(scan[i][:,idx,:] - scan[0][:,idx,:])
                elif plane == 'Sagittal':
                    res_arr.append(scan[i][:,:,idx] - scan[0][:,:,idx])
                elif plane == 'Transaxial':
                    res_arr.append(scan[i][idx,:,:] - scan[0][idx,:,:])

            # variance normalization
            res_arr = np.stack(res_arr)

            population_std = np.std(res_arr)

            # in rare case of divide by zero
            if population_std == 0:
                population_std = 10**-6

            print(bcolors.OKBLUE + f'Population Std: {population_std}' + bcolors.ENDC, flush=True)

            # save to dir
            for idx in range(axis_range):
                res_norm = res_arr[idx] / population_std
                arr = div_group.create_array(str(idx), shape=res_norm.shape , chunks=res_norm.shape, dtype='float32')
                arr[:] = res_norm


# write out data.json
with open(outpath / 'data.json', 'w') as f:
    json.dump(data_obj, f)

# final data.json file
print(bcolors.OKGREEN + 'Pre-processing completed' + bcolors.ENDC, flush=True)