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

class APD:
    def __init__(self, DATAPATH=pathlib.Path('./DATA')):
        self.DATAPATH = DATAPATH
        return
    
    def diffuse(self, t, T, x0, R, beta, convergence_verbose=False):
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

        # TODO make more memory efficient

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
            print(f'{bcolors.WARNING}Critical beta value: {critical_value_beta} {bcolors.ENDC}')
            print(f'{bcolors.WARNING}Beta value: {beta} {bcolors.ENDC}')
            if (beta > critical_value_beta*10):
                print(f'{bcolors.FAIL}Will not converge!{bcolors.ENDC}')
            elif (beta > critical_value_beta):
                print(f'{bcolors.WARNING}WARN: in range of beta_crit, might not converge! (but is possible){bcolors.ENDC}')

        # prepare torch tensors: (T, B, Width, Height) -> T = time axis, B = batch axis
        images = torch.zeros((T+1, B, x0.shape[1], x0.shape[2])).to(device)
        images[0,:,:,:] = x0

        # TODO
        # sample epsilon: (T, B, 300, 300) (T=time) -> N samples = T*B
        # Note epsilons are already variance-normalized
        # residualSet = APD.ResidualSet(self.DATAPATH)
        # sampler = torch.utils.data.RandomSampler(residualSet, replacement=True)
        # resLoader = torch.utils.data.DataLoader(residualSet, sampler=sampler, batch_size=B)

        # 1 diffusion step
        def step(xt_1, t):

            # ------------------ TEMP --------------------------------
            epsilon = torch.normal(0,1, size=xt_1.shape).to(device)
            # --------------------------------------------------------
            #epsilon = next(iter(resLoader)).to(device)

            # push everything to device
            xt = xt_1 + 1/T * R + sigma * f(t) * epsilon
            return xt
        
        # run whole chain
        for i in range(T):
            images[i+1,:,:,:] = step(xt_1=images[i,:,:,:], t=i+1)

        # output tensor
        x_t = torch.zeros((B, x0.shape[1], x0.shape[2])).to(device)

        for b in range(B):
            x_t[b] = images[t[b].item(), b,:,:].squeeze()

        return x_t
    
    class Dataset(torch.utils.data.Dataset):
        """ Class function for preparing/preprocessing the data that we want to feed to the neural network
            * init     --> initialise all arguments, and calculate all possible image pairs that can be fed into the model
            * len      --> gives the length of the self.search list
            * getitem  --> defines a dictionary with all info about a certain pair
        """

        def __init__(self, PatientList, DATAPATH=pathlib.Path('./DATA'), Planes = ["Coronal", "Sagittal", "Transax"], RandomFlip=False, divisions=[1, 5, 10, 20, 60], T=4):
            self.RandomFlip = RandomFlip
            self.DATAPATH = DATAPATH

            self.rng = np.random.default_rng()

            self.root = zarr.open_group(str(self.DATAPATH / 'PATIENTS'), mode='r')   

            self.search = []

            # data shape
            shape = (712, 300, 300)

            # divisions
            self.divisions = divisions
            self.T = T

            for patient in PatientList:
                for plane in Planes:
                    if plane == 'Coronal':
                        for idx in range(shape[1]):
                            self.search.append((patient, plane, idx))
                    elif plane == 'Sagittal':
                        for idx in range(shape[2]):
                            self.search.append((patient, plane, idx))
                    elif plane == 'Transax':
                        for idx in range(shape[0]):
                            self.search.append((patient, plane, idx))
            return
        
        def normalise(self, x, mean=None, std=None):
            # Z-score normalisation
            if mean is None or std is None:
                mean = torch.mean(x)
                std = torch.std(x)
                return (x - mean) / std, mean, std
            else:
                return (x - mean) / std, mean, std
        
        def __getitem__(self, index):
            # assigned random patient, plane and slice
            patient, plane, slice_idx = self.search[index]

            # sample LD division:
            ld_div = np.floor(self.rng.uniform(0,self.T,1))

            self.root = zarr.open_group(str(self.DATAPATH / 'PATIENTS'), mode='r')
            ld_scan = self.root[patient]['div' + str(self.divisions[ld_div+1])][:]
            hd_scan = self.root[patient]['div' + str(self.divisions[0])][:]

            if plane == 'Coronal':
                ld_img = ld_scan[:,slice_idx,:]
                hd_img = hd_scan[:,slice_idx,:]
            elif plane == 'Sagittal':
                ld_img = ld_scan[:,:,slice_idx]
                hd_img = hd_scan[:,:,slice_idx]
            elif plane == 'Transax':
                ld_img = ld_scan[slice_idx:,:]
                hd_img = hd_scan[slice_idx:,:]

            # cleanup
            del ld_scan
            del hd_scan

            # images should be normalised when fed to network
            Img_LD = ld_img.astype(np.float32)
            Img_HD = hd_img.astype(np.float32)

            # Expand dimension
            Img_LD = np.expand_dims(Img_LD, axis=0)
            Img_HD = np.expand_dims(Img_HD, axis=0)

            Img_LD, mean, std = self.normalise(torch.FloatTensor(Img_LD))
            Img_HD, _, _ = self.normalise(torch.FloatTensor(Img_HD), mean, std)

            if self.RandomFlip:
                if self.rng.random() > 0.5: # vertical flip
                    LD_Img = torch.flip(LD_Img, dims=[0])    
                    HD_Img = torch.flip(HD_Img, dims=[0])  
                if self.rng.random() > 0.5: # horizontal flip
                    LD_Img = torch.flip(LD_Img, dims=[1])
                    HD_Img = torch.flip(HD_Img, dims=[1])  

            item = {'LD_Img': LD_Img, 'HD_Img': HD_Img, 'Patient': patient, 'Plane': plane, 'SliceIndex': slice_idx, 'LD division': ld_div, 'divisions': self.divisions}

            return item

        def __len__(self):
            return len(self.search)