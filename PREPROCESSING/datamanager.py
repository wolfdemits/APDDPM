import numpy as np
import zarr
import json
import pathlib

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

class Datamanager:
    def __init__(self, DATAPATH=pathlib.Path('./DATA')):
        self.DATAPATH = DATAPATH

        try:
            # get data.json object
            self.data_obj = {}
            with open(self.DATAPATH / 'data.json') as f:
                self.data_obj = json.load(f)
        except:
            print(bcolors.FAIL + "Failed to retreive data object" + bcolors.ENDC)
            return

        try:
            self.root = zarr.open_group(str(self.DATAPATH / 'PATIENTS'), mode='r')   
        except:
            print(bcolors.FAIL + "Failed to find zarr root" + bcolors.ENDC)
            return

        self.available_ids = []
        self.available_tracers = []
        for tracer, ids in self.data_obj.items():
            self.available_ids.extend(ids)
            self.available_tracers.append(tracer)

        self.scan = None
        self.info_obj = None

        # start at default scan: (first patient of first tracer)
        self.current_tracer = next(iter(self.data_obj.keys()))
        self.current_patient = self.data_obj[self.current_tracer][0]
        self.load_scan(self.current_patient)
        
        return

    def load_scan(self, patient_id=None):
        """Load scan from disk to memory, also returns: scan + divisions"""

        if patient_id is None:
            patient_id = self.current_patient

        # check if in the data.json
        if not patient_id in self.available_ids:
            print(bcolors.WARNING + "Incorrect patient id" + bcolors.ENDC)
            return

        # load info object
        self.info_obj = {}
        with open(self.DATAPATH / 'PATIENTS' / patient_id / 'info.json') as f:
                self.info_obj = json.load(f)

        # read in scan 
        divisions = self.info_obj["divisions"]
        self.scan = np.zeros(shape=[len(divisions)] + self.info_obj["shape"], dtype='float32')
        
        for i, div in enumerate(divisions):
            self.scan[i] = self.root[patient_id]['div' + str(div)]

        print(f'Size in memory of scan: {round(self.scan.nbytes / 1024 / 1024,2)} MB')
        
        return self.scan, divisions

    def next_patient(self):
        """Changes patient pointer (current_patient) to next patient, note: the image still needs to be loaded via load_scan()!"""
        patient_list = self.data_obj[self.current_tracer]
        idx = patient_list.index(self.current_patient)
        next_idx = (idx + 1) % len(patient_list)
        self.current_patient = patient_list[next_idx]
        
        return

    def previous_patient(self):
        """Changes patient pointer (current_patient) to previous patient, note: the image still needs to be loaded via load_scan()!"""
        patient_list = self.data_obj[self.current_tracer]
        idx = patient_list.index(self.current_patient)
        next_idx = (idx - 1) % len(patient_list)
        self.current_patient = patient_list[next_idx]
        
        return

    def set_tracer(self, tracer):
        """Changes tracer pointer (current_tracer) to new tracer, note: the image still needs to be loaded via load_scan()!"""
        if not tracer in self.available_tracers:
            print(bcolors.WARNING + "Incorrect tracer" + bcolors.ENDC)
            return
        
        self.current_tracer = tracer
        self.current_patient = self.data_obj[self.current_tracer][0]

        return
        