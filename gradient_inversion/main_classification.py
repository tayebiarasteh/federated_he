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



def attack_main(num_classes=2, channel=1, num_iterations=3000, lr=1.0, downsample_size=28, firstdim=48):
    num_dummy = 1
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    tt = transforms.Compose([transforms.ToTensor()])
    image = image[:, :, 75]
    # image = gray2rgb(image)
    # image = imread('./data/img_train001_t1_slice75.png')

    hdim = ceil(ceil(image.shape[0] / 2) / 2)
    wdim = ceil(ceil(image.shape[1] / 2) / 2)
    hidden = hdim * wdim * firstdim
    net = LeNet(channel=channel, hideen=hidden, num_classes=num_classes, firstdim=firstdim)
    # net = ResNet(n_in_channels=channel, hidden=hidden, n_out_classes=num_classes)
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
    # resize_ratio = np.divide(tuple((downsample_size, downsample_size)), image.shape)
    # image = zoom(image, resize_ratio, order=2)
    gt_data = tt(image).float().to(device)
    gt_data = gt_data.unsqueeze(0)

    # compute original gradient
    out = net(gt_data)
    y = criterion(out, gt_label)
    dy_dx = torch.autograd.grad(y, net.parameters())
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

        if iters % int(num_iterations / 30) == 0:
            current_time = str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))
            print(current_time, iters, 'loss = %.8f, mse = %.8f' %(current_loss, mses[-1]))
            historynew.append([dummy_data[imidx].cpu() for imidx in range(num_dummy)])

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
    attack_main(num_classes=2, channel=1, num_iterations=500, lr=0.5, downsample_size=128, firstdim=48)