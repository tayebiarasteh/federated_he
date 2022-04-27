import time
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import pdb
import nibabel as nib
from skimage.color import gray2rgb
from scipy.ndimage.interpolation import zoom
from math import ceil
import random




class LeNet(nn.Module):
    def __init__(self, channel=3, hideen=768, num_classes=10, firstdim=48):
        super(LeNet, self).__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, firstdim, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(firstdim, firstdim, kernel_size=5, padding=5 // 2, stride=2),
            act(),
            nn.Conv2d(firstdim, firstdim, kernel_size=5, padding=5 // 2, stride=1),
            act(),
        )
        self.fc = nn.Sequential(
            nn.Linear(hideen, num_classes)
        )

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())


def encrypter(input_number, PRECISION_FRACTIONAL=1):
    BASE = 10
    PRECISION_INTEGRAL = 1
    # Q = 293973345475167247070445277780365744413
    # Q = 29397334547516724707
    # Q = 2939733
    Q = 800
    PRECISION = PRECISION_INTEGRAL + PRECISION_FRACTIONAL

    assert (Q > BASE ** PRECISION)

    # encoding
    upscaled = int(input_number * BASE**PRECISION_FRACTIONAL)
    field_element = upscaled % Q

    # encrypting
    first  = random.randrange(Q)
    second = random.randrange(Q)
    third  = (field_element - first - second) % Q
    result = first + second + third

    upscaled = result if result <= Q/2 else result - Q
    result = upscaled / BASE**PRECISION_FRACTIONAL

    return result * 1.0



def attack_main(num_classes=2, channel=1, num_iterations=3000, lr=1.0, downsample_size=28, firstdim=48, HE=False):
    num_dummy = 1
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    tt = transforms.Compose([transforms.ToTensor()])
    image = nib.load('/BraTS20_Training_002_t1ce.nii.gz').get_fdata()  # (h, w, d)
    image = image[:, :, 75]

    hdim = ceil(ceil(downsample_size / 2) / 2)
    wdim = ceil(ceil(downsample_size / 2) / 2)
    # hdim = ceil(ceil(image.shape[0] / 2) / 2)
    # wdim = ceil(ceil(image.shape[1] / 2) / 2)
    hidden = hdim * wdim * firstdim
    net = LeNet(channel=channel, hideen=hidden, num_classes=num_classes, firstdim=firstdim)
    net.apply(weights_init)
    net = net.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    gt_label = torch.Tensor([1]).long().to(device)

    #normalize
    non_zeros = image > 0
    low, high = np.percentile(image[non_zeros], [1, 99])
    image = np.clip(image, low, high)
    min_ = np.min(image)
    max_ = np.max(image)
    scale = max_ - min_
    image = (image - min_) / scale

    # downsample the image
    resize_ratio = np.divide(tuple((downsample_size, downsample_size)), image.shape)
    image = zoom(image, resize_ratio, order=2)
    gt_data = tt(image).float().to(device)
    gt_data = gt_data.unsqueeze(0)

    # compute original gradient
    out = net(gt_data)
    y = criterion(out, gt_label)
    dy_dx = torch.autograd.grad(y, net.parameters())

    if HE:
        dy_dx_new = []
        for index, grd in enumerate(dy_dx):
            grd = grd.detach().cpu()
            output_tensor = torch.zeros_like(grd)

            if grd.ndim == 1:
                for iiii in range(grd.shape[0]):
                    temp = encrypter(grd[iiii])
                    output_tensor[iiii] = temp

            elif grd.ndim == 2:
                for iii in range(grd.shape[0]):
                    for iiii in range(grd.shape[1]):
                        temp = encrypter(grd[iii, iiii])
                        output_tensor[iii, iiii] = temp

            elif grd.ndim == 3:
                for ii in range(grd.shape[0]):
                    for iii in range(grd.shape[1]):
                        for iiii in range(grd.shape[2]):
                            temp = encrypter(grd[ii, iii, iiii])
                            output_tensor[ii, iii, iiii] = temp

            elif grd.ndim == 4:
                for i in range(grd.shape[0]):
                    for ii in range(grd.shape[1]):
                        for iii in range(grd.shape[2]):
                            for iiii in range(grd.shape[3]):
                                temp = encrypter(grd[i, ii, iii, iiii])
                                output_tensor[i, ii, iii, iiii] = temp
            output_tensor = output_tensor.to(device)
            dy_dx_new.append(output_tensor)

        original_dy_dx = list((_.clone() for _ in dy_dx_new))

    else:
        original_dy_dx = list((_.detach().clone() for _ in dy_dx))

    # generate dummy data and label
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    optimizer = torch.optim.LBFGS([dummy_data, ], lr=lr)
    # predict the ground-truth label
    label_pred = torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False)

    losses = []
    mses = []
    train_iters = []
    historynew = []

    for iters in range(num_iterations):

        def closure():
            optimizer.zero_grad()
            pred = net(dummy_data)
            dummy_loss = criterion(pred, label_pred)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)
        current_loss = closure().item()
        train_iters.append(iters)
        losses.append(current_loss)
        mses.append(torch.mean((dummy_data-gt_data)**2).item())

        # if iters % int(num_iterations / 30) == 0:
        if iters % int(1) == 0:
            current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
            print(current_time, iters, 'loss = %.8f, mse = %.8f' %(current_loss, mses[-1]))
            historynew.append([dummy_data[imidx].cpu() for imidx in range(num_dummy)])
            segmentation = nib.Nifti1Image(historynew[iters][0].detach().numpy(), affine=np.eye(4))
            nib.save(segmentation, './results/BraTS/iter' + str(iters) + '.nii.gz')

            # for imidx in range(num_dummy):
            #     segmentation = nib.Nifti1Image(gt_data[imidx].cpu().numpy(), affine=np.eye(4))
            #     nib.save(segmentation, './results/BraTS/original.nii.gz')
            #
            #     for i in range(min(len(historynew), 29)):
            #         pdb.set_trace()
            #         segmentation = nib.Nifti1Image(historynew[i][imidx].detach().numpy(), affine=np.eye(4))
            #         nib.save(segmentation, './results/BraTS/iter' + str(iters) + '.nii.gz')

            if mses[-1] < 0.000001: # converge
                break

    segmentation = nib.Nifti1Image(gt_data[0].cpu().numpy(), affine=np.eye(4))
    nib.save(segmentation, './results/BraTS/original.nii.gz')
    segmentation = nib.Nifti1Image(historynew[-1][0].detach().numpy(), affine=np.eye(4))
    nib.save(segmentation, './results/BraTS/final.nii.gz')
    loss_iDLG = losses
    label_iDLG = label_pred.item()
    mse_iDLG = mses

    print('loss_iDLG:', loss_iDLG[-1])
    print('mse_iDLG:', mse_iDLG[-1])
    print('gt_label:', gt_label.detach().cpu().data.numpy(), 'lab_iDLG:', label_iDLG)
    print('----------------------\n\n')



if __name__ == '__main__':
    attack_main(num_classes=2, channel=1, num_iterations=161, lr=0.5, downsample_size=128, firstdim=48, HE=True)