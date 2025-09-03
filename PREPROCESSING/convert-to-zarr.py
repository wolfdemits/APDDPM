import numpy as np
import zarr
import pathlib
import json

inpath = pathlib.Path("D://DATA/PPEx_irx7_Subsampl_BinaryFormat")
outpath = pathlib.Path("./DATA")

SHAPE = (712,300,300)

dirlist = sorted([f.name for f in inpath.iterdir() if f.is_dir()])
print(f'Found patient directories: {dirlist}', flush=True)

data_obj = {}

for dir_name in dirlist:
    frags = dir_name.split('_')

    tracer = frags[2]
    patient_id = frags[0]

    if not tracer in data_obj:
        data_obj[tracer] = [patient_id]
    else:
        data_obj[tracer].append(patient_id)

    # get subsampled image paths
    subsampled = sorted([f.name for f in (inpath / dir_name).iterdir()])
    
    # write to zarr
    # open zarr root
    root = zarr.open_group(str(outpath / 'PATIENTS'), mode='a')

    # create patient group
    patient_group = root.require_group(patient_id)

    info_obj = {}

    # add subsampled images
    for i, image_dir in enumerate(subsampled):
        if image_dir in patient_group:
            # overwrite if exists
            del patient_group[image_dir]

        filepath = inpath / dir_name / image_dir
        img = np.fromfile(filepath, dtype='float32').reshape(SHAPE)

        name = str(image_dir)

        # get info from filename
        namefrags = name.split('_')

        # load in scan info only once, except for div info
        if i == 0:
            info_obj['type'] = namefrags[1]
            info_obj['tracer'] = namefrags[2]
            info_obj['time'] = namefrags[4]
            info_obj['shuffled'] = (namefrags[5] == 'sh')
            info_obj['shape'] = SHAPE
    
            if namefrags[6] == 'NCts':
                division = int(namefrags[7][3:])
                info_obj['divisions'] = [division]
                info_obj['TOF'] = namefrags[8][:5]
                info_obj['blb'] = namefrags[9].split('-')[0][3:]
                info_obj['iteration'] = namefrags[9].split('-')[1].split('.')[0]
            else:
                division = 1
                info_obj['divisions'] = [division]
                info_obj['TOF'] = namefrags[6][:5]
                info_obj['blb'] = namefrags[7].split('-')[0][3:]
                info_obj['iteration'] = namefrags[7].split('-')[1].split('.')[0]
        else:
            if namefrags[6] == 'NCts':
                division = int(namefrags[7][3:])
                info_obj['divisions'].append(division)
            else:
                division = 1
                info_obj['divisions'].append(division)
        
        arr = patient_group.create_array(f'div{str(division)}', shape=img.shape , chunks=img.shape, dtype='float32')
        arr[:] = img

        # cleanup
        del img

    # sort division list
    info_obj['divisions'] = sorted(info_obj['divisions'])

    # write json
    with open(outpath / 'PATIENTS' / patient_id / 'info.json', 'w') as f:
        json.dump(info_obj, f)

    print(f'Completed converting: {name}', flush=True)


# write final data.json
# write json
with open(outpath / 'data.json', 'w') as f:
    json.dump(data_obj, f)

print(f'Conversion completed. ', flush=True)