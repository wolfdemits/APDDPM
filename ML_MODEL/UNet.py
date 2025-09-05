#%%
import torch
import torch.nn as nn

""" This script defines the blocks and classes used to build-up the CNN model """

def count_model_parameters(model):
    
    total_params = 0
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        total_params+=param

    return total_params


#### CONVOLUTIONAL BLOCKS ####################################################

class ConvBlock(nn.Module):
    
    """ Convolutional blocks can include: conv layers, dropout layers, batch
    normalization layers, and the activation function as last block """
    
    def __init__(self, dim, in_channel, out_channel, conv_kernel, dilation, normalization, activation):
        
        super().__init__()
        
        Conv = nn.Conv2d if dim == '2d' else nn.Conv3d

        if normalization == 'batch_norm':
            Norm = nn.BatchNorm2d if dim == '2d' else nn.BatchNorm3d
        elif normalization == 'instance_norm':
            Norm = nn.InstanceNorm2d if dim == '2d' else nn.InstanceNorm3d 
        
        self.conv  = Conv(in_channel, out_channel, conv_kernel, stride=1, padding="same", dilation=dilation)
        self.norm  = Norm(out_channel)
        self.activ = getattr(nn, activation)()
        
    def forward(self, x):
        
        x = self.conv(x)
        x = self.norm(x)
        x = self.activ(x)
        
        return x
    

class DoubleConvBlock(nn.Module):
    
    def __init__(self, dim, in_channel, mid_channel, out_channel, conv_kernel, dilation, normalization, activation):
        
        super().__init__()
        
        self.conv1 = ConvBlock(dim, in_channel, mid_channel, conv_kernel, dilation, normalization, activation)
        self.conv2 = ConvBlock(dim, mid_channel, out_channel, conv_kernel, dilation, normalization, activation)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


#### SINGLE CONV BLOCKS ###############################################################    
    
class Conv_1x1(nn.Module):
    
    def __init__(self, dim, in_channel, out_channel):
        
        super().__init__()
        Conv = nn.Conv2d if dim == '2d' else nn.Conv3d
        self.conv = Conv(in_channel, out_channel, kernel_size=1, stride=1, padding="same", dilation=1)
        
    def forward(self, x):
        x = self.conv(x)
        return x


#### ENCODER #################################################################

class DownBlock_Pool(nn.Module):
    
    def __init__(self, dim, in_channel, out_channel, conv_kernel, dilation, pool_mode, normalization, activation):
        
        super().__init__()
        
        if pool_mode == 'maxpool':
            pool_operation = nn.MaxPool2d if dim == '2d' else nn.MaxPool3d
        elif pool_mode == 'meanpool':
            pool_operation = nn.AvgPool2d if dim == '2d' else nn.AvgPool3d
        
        double_conv = DoubleConvBlock(dim, in_channel, out_channel, out_channel, conv_kernel, dilation, normalization, activation)
        self.down = nn.Sequential(pool_operation(2), double_conv)

    def forward(self, x):
        return self.down(x)


## Strided Conv. 
class DownBlock_ConvStride2(nn.Module):
    
    def __init__(self, dim, in_channel, out_channel, conv_kernel, dilation, normalization, activation):
        
        super().__init__()

        Conv = nn.Conv2d if dim == '2d' else nn.Conv3d
        
        if normalization == 'batch_norm':
            Norm = nn.BatchNorm2d if dim == '2d' else nn.BatchNorm3d
        elif normalization == 'instance_norm':
            Norm = nn.InstanceNorm2d if dim == '2d' else nn.InstanceNorm3d
        
        self.down = nn.Sequential(
            Conv(in_channel, in_channel, kernel_size=3, stride=2), 
            Norm(in_channel), 
            getattr(nn, activation)())
        
        self.double_conv = DoubleConvBlock(
            dim, in_channel, out_channel, out_channel, conv_kernel, dilation, normalization, activation)
       
    def forward(self, x):
        x = self.down(x)
        x = self.double_conv(x)
        return x


#### ATTENTION GATE BLOCK ####################################################

""" ADOPTED FROM OKTAY ET AL. (2018). 
    Instead of directly directly concatenating the feature maps, this approach adds an 
    attention gating mechanism that helps the neural network focus on relevant regions. 
    
    Step 1: input feature maps (size of x_encode) downsampled to resolution of gating signal
            & decrease the number of channels on gating_signal 
            => decouple the feature maps and map them to lower dimensional space
    Step 2: Additive Attention Gating Operation
    """

class AttentionBlock(nn.Module):
    
    def __init__(self, dim, num_encode_channels, num_gating_channels):
        
        super().__init__()
        self.dim = dim
        Conv = nn.Conv2d if dim == '2d' else nn.Conv3d
        mode = 'bilinear' if dim=='2d' else 'trilinear'
        
        # Conv with Stride 2
        self.spatial_Downsampler = Conv(num_encode_channels, num_encode_channels, kernel_size=1, stride=2)
        
        # Decrease the number of channels on gating_signal
        self.featuremap_Downsampler = Conv(num_gating_channels, num_encode_channels, kernel_size=1, stride=1)
        
        # Gating operation
        self.ReLU = nn.ReLU()
        self.attenCoeff_Downsampler = Conv(num_encode_channels, 1, kernel_size=1, stride=1)
        self.Sigmoid = nn.Sigmoid()
        
        # Resampler
        self.resampler = nn.Upsample(scale_factor=2, mode=mode)


    def forward(self, x_encode, gating_signal):

        x_encode_downsampled = self.spatial_Downsampler(x_encode)
        gating_signal_downfeatured = self.featuremap_Downsampler(gating_signal)

        ## Additive Gating Operation
        x_encode_downsampled = resizePadding(
            self.dim, x_to_resize=x_encode_downsampled, x_reference=gating_signal_downfeatured)
        atten_coeff = self.ReLU(x_encode_downsampled + gating_signal_downfeatured)
        atten_coeff = self.attenCoeff_Downsampler(atten_coeff)
        atten_coeff = self.Sigmoid(atten_coeff)

        atten_map = self.resampler(atten_coeff)
        
        ## Input features are scaled with attention coefficients computed in AG
        x_encode_resized = resizePadding(
            self.dim, x_to_resize=x_encode, x_reference=atten_map)
        attenGated_x_encode = atten_map * x_encode_resized 

        return attenGated_x_encode


#### HELPER FUNCTIONS FOR: ##################################
######## RESIZING 
######## CONCATENATE THE CHANNELS  

def resizePadding(dim, x_to_resize, x_reference, padValue=0):

    diff_w = x_reference.shape[-1] - x_to_resize.shape[-1]
    diff_h = x_reference.shape[-2] - x_to_resize.shape[-2]
        
    if dim == '2d':            
        x_padded = torch.nn.functional.pad(x_to_resize, (0, diff_w, 0, diff_h), value=padValue)
    
    elif dim == '3d':
        diff_z = x_reference.shape[-3] - x_to_resize.shape[-3]  
        x_padded = torch.nn.functional.pad(x_to_resize, (0, diff_w, 0, diff_h, 0, diff_z), value=padValue)

    return x_padded


def concat_skipConnection(dim, x_out_skip, x_decode):

    if (x_out_skip.shape[2:] != x_decode.shape[2:]):
        x_decode = resizePadding(dim, x_to_resize=x_decode, x_reference=x_out_skip) 

    x = torch.cat([x_decode, x_out_skip], dim=1)

    return x


#### DECODER #################################################################

""" Three cases: 
    1) No skip connections (so also no attention gate) 
    2) Normal skip connections without attention gates
    3) Skip connections with attention gates """

class UpSample(nn.Module):
    
    def __init__(
            self, dim, num_main_channel, num_skip_channel, num_channel_out, conv_kernel, 
            dilation, normalization, activation, attenGate):
        
        super().__init__()

        self.dim = dim
        mode = 'bilinear' if dim=='2d' else 'trilinear'
        self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)

        in_channel = num_main_channel + num_skip_channel
        
        # with skip connections
        if num_skip_channel != 0: 
            self.skip_connection = True
            self.attenGate = attenGate
            mid_channel = num_main_channel
            out_channel = num_channel_out
            
        # no skip connections (-> so no attention gating)
        elif num_skip_channel == 0: 
            self.skip_connection = False
            self.attenGate = False
            mid_channel = num_channel_out
            out_channel = num_channel_out
        
        if self.attenGate: 
            self.attenBlock = AttentionBlock(dim, num_skip_channel, num_main_channel)
        
        self.double_conv = DoubleConvBlock(dim, in_channel, mid_channel, out_channel, conv_kernel, dilation, normalization, activation)
    
    
    def forward(self, x, x_encode):

        x_up = self.up(x)

        if (self.skip_connection) and (self.attenGate):         # concatenate with attention gating
            attenGated_x_encode = self.attenBlock(x_encode=x_encode, gating_signal=x) 
            x = concat_skipConnection(self.dim, x_out_skip=attenGated_x_encode, x_decode=x_up)
            
        elif (self.skip_connection) and not (self.attenGate):   # only concatenation
            x = concat_skipConnection(self.dim, x_out_skip=x_encode, x_decode=x_up)

        x = self.double_conv(x)
        
        return x

    
class UpConv(nn.Module):
    
    def __init__(
            self, dim, num_main_channel, num_skip_channel, num_channel_out, conv_kernel, 
            dilation, normalization, activation, attenGate):
    
        super().__init__()

        self.dim = dim
        ConvTranspose = nn.ConvTranspose2d if dim=='2d' else nn.ConvTranspose3d
        self.up = ConvTranspose(num_main_channel, num_main_channel, kernel_size=2, stride=2)
        
        in_channel = num_main_channel + num_skip_channel
        
        # with skip connections
        if num_skip_channel != 0: 
            self.skip_connection = True
            self.attenGate = attenGate
            mid_channel = num_main_channel
            out_channel = num_channel_out
            
        # no skip connections (-> so no attention gating)
        elif num_skip_channel == 0: 
            self.skip_connection = False
            self.attenGate = False
            mid_channel = num_channel_out
            out_channel = num_channel_out
        
        if self.attenGate: 
            self.attenBlock = AttentionBlock(dim, num_skip_channel, num_main_channel)

        self.double_conv = DoubleConvBlock(
            dim, in_channel, mid_channel, out_channel, conv_kernel, dilation, normalization, activation)

    
    def forward(self, x, x_encode):
        
        x_up = self.up(x)

        if (self.skip_connection) and (self.attenGate):         # concatenate with attention gating
            attenGated_x_encode = self.attenBlock(x_encode=x_encode, gating_signal=x) 
            x = concat_skipConnection(self.dim, x_out_skip=attenGated_x_encode, x_decode=x_up)
            
        elif (self.skip_connection) and not (self.attenGate):   # only concatenation
            x = concat_skipConnection(self.dim, x_out_skip=x_encode, x_decode=x_up)

        x = self.double_conv(x)

        return x
      


#### UNET #####################################################################

class UNet_withAttenGate(nn.Module):
    
    def __init__(self, dim, num_in_channels, features_main, features_skip, conv_kernel_size, 
            dilation, down_mode, up_mode, normalization, activation, attenGate, residual_connection):
        
        super().__init__()
        
        self.dim = dim
        self.depth = len(features_skip)
        self.num_in_channels = num_in_channels
        self.residual_connection = residual_connection
        
        # INCOME, FIRST LAYERS
        self.income = DoubleConvBlock(
            dim, num_in_channels, features_main[0], features_main[0], conv_kernel_size, dilation, normalization, activation)
        
        # DOWN PART
        if (down_mode == 'maxpool') or (down_mode == 'meanpool'):
            self.Downs = nn.ModuleList([
                                DownBlock_Pool(dim, 
                                        features_main[i],
                                        features_main[i+1],
                                        conv_kernel_size,
                                        dilation,
                                        down_mode,
                                        normalization,
                                        activation)
                                for i in range(self.depth)])

        elif (down_mode == 'convStrided'):
            self.Downs = nn.ModuleList([
                                DownBlock_ConvStride2(dim, 
                                        features_main[i],
                                        features_main[i+1],
                                        conv_kernel_size,
                                        dilation, 
                                        normalization, 
                                        activation)
                                for i in range(self.depth)])
        
        # UP PART
        if (up_mode == 'upsample'):
            self.Ups = nn.ModuleList([
                            UpSample(dim, 
                                     features_main[i+1],
                                     features_skip[i],
                                     features_main[i],
                                     conv_kernel_size,
                                     dilation,
                                     normalization, 
                                     activation, 
                                     attenGate)
                            for i in reversed(range(self.depth))])
        
        elif (up_mode == 'upconv'):
            self.Ups = nn.ModuleList([
                            UpConv(dim, 
                                   features_main[i+1],
                                   features_skip[i],
                                   features_main[i],
                                   conv_kernel_size,
                                   dilation, 
                                   normalization,
                                   activation, 
                                   attenGate)
                            for i in reversed(range(self.depth))])
        
        # OUT: Conv1x1 && Add ReLU (enforce non-negativity)
        self.out_Conv1x1 = Conv_1x1(dim, features_main[0], 1)
        self.out_ReLU = getattr(nn, 'ReLU')()
    
    
    def forward(self, x_input):
        
        x = self.income(x_input)
        
        save_skip = []

        for encode_block in self.Downs:
            save_skip.append(x)
            x = encode_block(x)
            
        for decode_block in self.Ups:
            x = decode_block(x, save_skip.pop())
            
        x = self.out_Conv1x1(x)
        x = resizePadding(self.dim, x_to_resize=x, x_reference=x_input)
    
        ## Include Residual Connection: Network predicts y_diff = x_out - x_input
        if self.residual_connection:
            y_diff = x   
            residualChannel = self.num_in_channels // 2   
            if (self.dim == '2d'):
                x_input_oneChannel = x_input[:, residualChannel:residualChannel+1, :, :]
            elif (self.dim == '3d'):
                x_input_oneChannel = x_input[:, residualChannel:residualChannel+1, :, :, :]
            x_out = y_diff + x_input_oneChannel

        else: 
            x_out = x
        
        x_out = self.out_ReLU(x_out)
        
        return x_out

    
#### TEST #####################################################################

if __name__ == '__main__':
    
    #import torchinfo
    
    """ CNN UNET ENCODER-DECODER """

    random2Dtensor = torch.rand(32, 1, 93, 149)

    gener = UNet_withAttenGate(
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
        residual_connection = True)
    
    output = gener(random2Dtensor)
    print(output.shape)
    
    #print(torchinfo.summary(gener, random2Dtensor.shape, depth=4))
