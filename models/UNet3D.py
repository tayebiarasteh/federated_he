"""
Created on March 4, 2022.
UNet3D.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""


import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F


class UNet3D(nn.Module):
    def __init__(self, n_in_channels=4, n_out_classes=3, threelevel=False, firstdim=48, weight_init=True):
        """
        Parameters
        ----------
        firstdim: int
            16
            24

        threelevel: bool
            if we want to have 3-level UNet or 4-level

        weight_init: bool
            if we want to initialize the biases with zero and weights with He initialization
        """
        super(UNet3D, self).__init__()
        self.threelevel = threelevel

        self.input_block = inconv(n_in_channels, firstdim, weight_init)

        # 3-level
        if self.threelevel:
            self.down1 = down_one(firstdim, firstdim * 2, weight_init)
            self.down2 = down(firstdim * 2, firstdim * 2, weight_init)
            self.up2 = up(firstdim * 4, firstdim, weight_init)
            self.up3 = up_one(firstdim * 2, firstdim, weight_init)

        # 4-level
        else:
            self.down1 = down_one(firstdim, firstdim * 2, weight_init)
            self.down2 = down(firstdim * 2, firstdim * 4, weight_init)
            self.down3 = down(firstdim * 4, firstdim * 4, weight_init)
            self.up1 = up(firstdim * 8, firstdim * 2, weight_init)
            self.up2 = up(firstdim * 4, firstdim, weight_init)
            self.up3 = up_one(firstdim * 2, firstdim, weight_init)

        self.output_block = outconv(firstdim, n_out_classes, weight_init)
        

    def forward(self, input_tensor):
        first_output = self.input_block(input_tensor)

        # 3-level
        if self.threelevel:
            down1_output = self.down1(first_output)
            down2_output = self.down2(down1_output)
            up2_output = self.up2(down2_output, down1_output)
            up3_output = self.up3(up2_output, first_output)

        # 4-level
        else:
            down1_output = self.down1(first_output)
            down2_output = self.down2(down1_output)
            down3_output = self.down3(down2_output)
            up1_output = self.up1(down3_output, down2_output)
            up2_output = self.up2(up1_output, down1_output)
            up3_output = self.up3(up2_output, first_output)

        # unpadding
        if input_tensor.shape[-1] != up3_output.shape[-1]:
            diff =  up3_output.shape[-1] - input_tensor.shape[-1]
            up3_output = up3_output[..., :-diff]
        if input_tensor.shape[-2] != up3_output.shape[-2]:
            diff2 =  up3_output.shape[-2] - input_tensor.shape[-2]
            up3_output = up3_output[:, :, :, :-diff2]
        if input_tensor.shape[-3] != up3_output.shape[-3]:
            diff3 =  up3_output.shape[-3] - input_tensor.shape[-3]
            up3_output = up3_output[:, :, :-diff3]

        output_tensor = self.output_block(up3_output)

        return output_tensor


    
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch, weight_init):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1), # (n,c,d,h,w)
            nn.ReLU(inplace=False),
            nn.BatchNorm3d(out_ch),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.BatchNorm3d(out_ch))

        if weight_init:
            for idx in range(len(self.conv)):
                for name, param in self.conv[idx].named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        if isinstance(self.conv[idx], nn.Conv3d) or isinstance(self.conv[idx], nn.Conv2d) or isinstance(
                                self.conv[idx], nn.ConvTranspose2d) or isinstance(self.conv[idx], nn.ConvTranspose3d):
                            nn.init.kaiming_normal_(param, a=1e-2)


    def forward(self, input_tensor):
        output_tensor = self.conv(input_tensor)
        return output_tensor


#Input Block
class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, weight_init):
        super(inconv, self).__init__()
        self.in_double_conv1 = double_conv(in_ch, out_ch, weight_init)

    def forward(self, input_tensor):
        #Apply input_tensor on object from init function and return the output_tensor 
        output_tensor = self.in_double_conv1(input_tensor)
        return output_tensor

#Down 2, 3 Block
class down(nn.Module):
    def __init__(self, in_ch, out_ch, weight_init):
        super(down, self).__init__()
        self.maxpool1 = nn.MaxPool3d(kernel_size = 2)
        self.down_double_conv1 = double_conv(in_ch, out_ch, weight_init)

    def forward(self, input_tensor):
        #Apply input_tensor on object from init function and return the output_tensor
        input_tensor = self.maxpool1(input_tensor)
        output_tensor = self.down_double_conv1(input_tensor)
        return output_tensor


#Down 1 Block
class down_one(nn.Module):
    def __init__(self, in_ch, out_ch, weight_init):
        super(down_one, self).__init__()
        self.maxpool1 = nn.MaxPool3d(kernel_size = 2)
        self.down_double_conv1 = double_conv(in_ch, out_ch, weight_init)

    def forward(self, input_tensor):
        #Apply input_tensor on object from init function and return the output_tensor
        input_tensor = self.maxpool1(input_tensor)
        output_tensor = self.down_double_conv1(input_tensor)
        return output_tensor



#Up 2, 3 Block
class up(nn.Module):
    def __init__(self, in_ch, out_ch, weight_init):
        super(up, self).__init__()
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.up_double_conv1 = double_conv(in_ch, out_ch, weight_init)
        
    def forward(self, input_tensor_1, input_tensor_2):
        input_tensor_1 = self.upsample1(input_tensor_1)

        # zero-padding
        if input_tensor_1.shape[-1] != input_tensor_2.shape[-1]:
            diff =  input_tensor_2.shape[-1] - input_tensor_1.shape[-1]
            input_tensor_1 = F.pad(input_tensor_1, (0, diff), "constant", 0)
        if input_tensor_1.shape[-2] != input_tensor_2.shape[-2]:
            diff2 =  input_tensor_2.shape[-2] - input_tensor_1.shape[-2]
            input_tensor_1 = F.pad(input_tensor_1, (0, 0, 0, diff2), "constant", 0)
        if input_tensor_1.shape[-3] != input_tensor_2.shape[-3]:
            diff3 =  input_tensor_2.shape[-3] - input_tensor_1.shape[-3]
            input_tensor_1 = F.pad(input_tensor_1, (0, 0, 0, 0, 0, diff3), "constant", 0)

        #Concatenation of the  upsampled result and input_tensor_2
        input_tensor = torch.cat((input_tensor_1 , input_tensor_2), 1)
        output_tensor = self.up_double_conv1(input_tensor)
        return output_tensor


#Up 1 Block
class up_one(nn.Module):
    def __init__(self, in_ch, out_ch, weight_init):
        super(up_one, self).__init__()
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.up_double_conv1 = double_conv(in_ch, out_ch, weight_init)

    def forward(self, input_tensor_1, input_tensor_2):
        input_tensor_1 = self.upsample1(input_tensor_1)

        # zero-padding
        if input_tensor_1.shape[-1] != input_tensor_2.shape[-1]:
            diff =  input_tensor_2.shape[-1] - input_tensor_1.shape[-1]
            input_tensor_1 = F.pad(input_tensor_1, (0, diff), "constant", 0)
        if input_tensor_1.shape[-2] != input_tensor_2.shape[-2]:
            diff2 =  input_tensor_2.shape[-2] - input_tensor_1.shape[-2]
            input_tensor_1 = F.pad(input_tensor_1, (0, 0, 0, diff2), "constant", 0)
        if input_tensor_1.shape[-3] != input_tensor_2.shape[-3]:
            diff3 =  input_tensor_2.shape[-3] - input_tensor_1.shape[-3]
            input_tensor_1 = F.pad(input_tensor_1, (0, 0, 0, 0, 0, diff3), "constant", 0)

        #Concatenation of the  upsampled result and input_tensor_2
        input_tensor = torch.cat((input_tensor_1 , input_tensor_2), 1)
        output_tensor = self.up_double_conv1(input_tensor)
        return output_tensor



#Out Block
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, weight_init):
        super(outconv, self).__init__()
        self.conv_out = nn.Conv3d(in_ch, out_ch, kernel_size=1)
        if weight_init:
            for name, param in self.conv_out.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    if isinstance(self.conv_out, nn.Conv3d) or isinstance(self.conv_out, nn.Conv2d) or isinstance(
                            self.conv_out, nn.ConvTranspose2d) or isinstance(self.conv_out, nn.ConvTranspose3d):
                        nn.init.kaiming_normal_(param, a=1e-2)

    def forward(self, input_tensor):
        output_tensor = self.conv_out(input_tensor)
        return output_tensor