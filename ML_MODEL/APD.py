import pathlib
import numpy as np
import torch
import zarr

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

# TODO -: write docstrings: example for dataloaders: output shapes

class APD:
    def __init__(self, PATH=pathlib.Path('./')):
        self.PATH = PATH
        return
    
    def diffuse(self, t, T, x0, R, beta, res_info=None, convergence_verbose=False):
        """
        Apply Forward Anchored Path Diffusion Process to image.

        Note: x0 and R should always be normalized!

        Uses time schedule function: f(t) = 4*alpha_t(1-alpha_t), alpha_t = t/T

        Parameters:
        -----------
        t: torch time tensor: (B,)
            Stop forward process at time point t. (inclusive)
            ! t cannot contain timsetps 0 or lower !
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
        resIterator = None
        if not res_info is None:
            # use residuals, else use gaussian noise
            residualSet = APD.ResidualSet(PatientList=res_info['patients'], plane=res_info['plane'], division=res_info['division'], PATH=self.PATH, RandomFlip=True)
            sampler = torch.utils.data.RandomSampler(residualSet, replacement=True)
            resLoader = torch.utils.data.DataLoader(residualSet, sampler=sampler, batch_size=B)
            resIterator = iter(resLoader)

        # 1 diffusion step
        def step(xt_1, t):
            if resIterator is None:
                # gaussian if no resloader created
                epsilon = torch.normal(0,1, size=xt_1.shape).to(device)

            else:
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

        # output tensor
        x_t = torch.zeros((B, x0.shape[1], x0.shape[2])).to(device)
        x_t_1 = torch.zeros((B, x0.shape[1], x0.shape[2])).to(device)

        for b in range(B):
            x_t[b] = images[t[b].item(), b,:,:].squeeze()
            x_t_1[b] = images[t[b].item() - 1, b,:,:].squeeze()

        return x_t, x_t_1
    
    class Dataset(torch.utils.data.Dataset):
        """ Class function for preparing/preprocessing the data that we want to feed to the neural network
            * init     --> initialise all arguments, and calculate all possible image pairs that can be fed into the model
            * len      --> gives the length of the self.search list
            * getitem  --> defines a dictionary with all info about a certain pair
        """

        def __init__(self, PatientList, PATH=pathlib.Path('./'), Planes = ["Coronal", "Sagittal", "Transax"], RandomFlip=False, divisions=[1, 5, 10, 20, 60]):
            self.RandomFlip = RandomFlip
            self.DATAPATH = PATH / 'DATA_PREPROCESSED'

            self.rng = np.random.default_rng()

            self.root = zarr.open_group(str(self.DATAPATH / 'PATIENTS'), mode='r')   

            self.search = []

            # data shape
            self.shape = (712, 300, 300)

            # divisions
            self.divisions = divisions

            for patient in PatientList:
                for plane in Planes:
                    # find amount of slices
                    try:
                        slices = self.root[patient][plane]['div1'].keys()
                    except:
                        print(bcolors.FAIL + f'Unable to find specified group: {patient} -> {plane} -> div1' + bcolors.ENDC)
                        return
                    
                    for idx in slices:
                        self.search.append((patient, plane, idx))
            return
        
        def __getitem__(self, index):
            # assigned random patient, plane and slice
            patient, plane, slice_idx = self.search[index]

            images = []

            self.root = zarr.open_group(str(self.DATAPATH / 'PATIENTS'), mode='r')

            for i, div in enumerate(self.divisions):
                try:
                    img = self.root[patient][plane]['div' + str(div)][slice_idx][:]
                except:
                    print(bcolors.FAIL + f'Unable to find specified group: {patient} -> {plane} -> div{div} -> {slice_idx}' + bcolors.ENDC)
                    break

                #convert to torch
                Img = torch.as_tensor(img, dtype=torch.float32)

                if self.RandomFlip:
                    if self.rng.random() > 0.5: # vertical flip
                        Img = torch.flip(Img, dims=[0])
                    if self.rng.random() > 0.5: # horizontal flip
                        Img = torch.flip(Img, dims=[1])

                images.append(Img)

            images = torch.stack(images)

            item = {'Images': images, 'divisions': self.divisions, 'Patient': patient, 'Plane': plane, 'SliceIndex': slice_idx}

            return item

        def __len__(self):
            return len(self.search)
        
    class ResidualSet:
        def __init__(self, PatientList, plane, division, PATH=pathlib.Path('./'), RandomFlip=False):
            self.RandomFlip = RandomFlip
            self.DATAPATH = PATH / 'RESIDUALS'

            self.rng = np.random.default_rng()

            self.root = zarr.open_group(str(self.DATAPATH / 'PATIENTS'), mode='r')   

            self.search = []

            # division
            self.division = division

            for patient in PatientList:
                # find amount of slices
                try:
                    slices = self.root[patient][plane][f'div{division}'].keys()
                except:
                    print(bcolors.FAIL + f'Unable to find specified group: {patient} -> {plane} -> div{division}' + bcolors.ENDC)
                    return
                    
                for idx in slices:
                    self.search.append((patient, plane, division, idx))
            return
        
        def __getitem__(self, index):
            # assigned random patient, plane and slice
            patient, plane, division, slice_idx = self.search[index]

            self.root = zarr.open_group(str(self.DATAPATH / 'PATIENTS'), mode='r')

            try:
                img = self.root[patient][plane]['div' + str(division)][slice_idx][:]
            except:
                print(bcolors.FAIL + f'Unable to find specified group: {patient} -> {plane} -> div{division} -> {slice_idx}' + bcolors.ENDC)
                return

            #convert to torch
            Img = torch.as_tensor(img, dtype=torch.float32)

            if self.RandomFlip:
                if self.rng.random() > 0.5: # vertical flip
                    Img = torch.flip(Img, dims=[0])
                if self.rng.random() > 0.5: # horizontal flip
                    Img = torch.flip(Img, dims=[1])

            item = {'Residual': Img, 'division': self.division, 'Patient': patient, 'Plane': plane, 'SliceIndex': slice_idx}

            return item

        
        def __len__(self):
            return len(self.search)
        
    ## Collate function
    class CollateFn2D():

        """ Upon loading the individual data (i.e., slices of different sizes, if cropping was used) into one batch
            -> Input data may not necessarily be of the same matrix size

            of the same matrix size, but all slices in one batch need to be 
        of the same size, so collate function is used to pad the slices to equal size """

        def __call__(self, item_list):

            batch = {} # Initialise an empty dictionary
            

            # Iterate over all keys of an item (so over 'LR_Img', 'HR_Img', 'Subject', 'Plane' and 'SliceName')

            for key in item_list[0].keys():

                # If the key is 'Images' (= these contain the data arrays) -> They need to be padded
                
                if key == 'Images': 
                    
                    tensor_list = [item[key] for item in item_list]
                    shape_list  = [tensor.shape for tensor in tensor_list]

                    height_max  = max(shape[-2] for shape in shape_list) # Determines the max height of all tensors --> we have to padd al the other tensors to this height
                    width_max   = max(shape[-1] for shape in shape_list) # Determines the max width of all tensors --> we have to padd all the other tensors to this width
                    
                    pad_list    = [((width_max - shape[-1]) // 2,      # left padding
                                    -(-(width_max - shape[-1]) // 2),  # right padding
                                    (height_max - shape[-2]) // 2,     # top padding
                                    -(-(height_max - shape[-2]) // 2)) # bottom padding   
                                for shape in shape_list]
                    
                    tensor = torch.stack([
                                torch.nn.functional.pad(tensor, padding, value=-1) 
                                for tensor, padding in zip(tensor_list, pad_list)])
                    
                    # Add stacked & padded tensor data to batch dictionary using the exsisting key
                    batch[key] = torch.transpose(tensor, 0,1)
                
                else:
                    # If key is something other than 'LR_Img' or 'HR_Img' --> Merge data of all items in a list using stack command and add to batch
                    batch[key] = np.stack([item[key] for item in item_list])

            return batch
        
#wDM#