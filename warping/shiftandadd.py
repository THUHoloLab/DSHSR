import random
import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision.transforms import GaussianBlur
import numpy as np 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def get_neighbours(coords):

    coords_lr = torch.ceil(coords)
    coords_ul = torch.floor(coords)
    ys_upper, xs_left = torch.split(coords_ul, 1, dim = 1)
    ys_lower, xs_right = torch.split(coords_lr, 1, dim = 1)
    coords_ll = torch.cat((ys_lower, xs_left), axis = 1)
    coords_ur = torch.cat((ys_upper, xs_right), axis = 1)
    
    return coords_ul, coords_ur, coords_ll, coords_lr

Gaussian_Filter = GaussianBlur(11, sigma=1).to(device)
def coords_unroll(coords, sr_ratio = 2):
    """
    coords : tensor(b,2,h,w)
    """
    b,c,h,w = coords.shape
    assert(c == 2)
    coords_ = coords.view(b,c,-1)
    coords_ = sr_ratio*w*coords_[:,0] + coords_[:,1]
    return coords_

def get_coords(h, w):
    """get coords matrix of x

    # Arguments
        h
        w
    
    # Returns
        coords: (h, w, 2)
    """
    coords = torch.empty(2, h, w, dtype = torch.float)
    coords[0,...] = torch.arange(h)[:, None]
    coords[1,...] = torch.arange(w)

    return coords


def shiftAndAdd(samples, flows, sr_ratio, local_rank):
    """
    samples: Tensor(b, h, w) float32
    flows: Tensor(b, 2, h, w) float32
    """
    
    flows_ = torch.empty_like(flows)
    flows_[:,0,...] = flows[:,1,...]
    flows_[:,1,...] = flows[:,0,...]
    
    
    b, h, w = samples.shape
    samples_= samples.reshape(b, -1).type(torch.float32)

    mapping = sr_ratio*(flows_ + get_coords(h,w).cuda(local_rank))
    mappingy, mappingx = torch.split(mapping, 1, dim = 1)
    mappingy = torch.clamp(mappingy, 0, sr_ratio*h-1)
    mappingx = torch.clamp(mappingx, 0, sr_ratio*w-1)

    mapping = torch.cat((mappingy, mappingx), 1)

    coords_ul, coords_ur, coords_ll, coords_lr = get_neighbours(mapping) # all (b, 2, h, w)


    diff = (mapping - coords_ul).type(torch.float32).cuda(local_rank)
    neg_diff = (1.0 - diff).type(torch.float32).cuda(local_rank)
    diff_y, diff_x = torch.split(diff, 1, dim = 1)
    neg_diff_y, neg_diff_x = torch.split(neg_diff, 1, dim = 1)
    diff_x = diff_x.reshape(b,-1)
    diff_y = diff_y.reshape(b,-1)
    neg_diff_x = neg_diff_x.reshape(b,-1)
    neg_diff_y = neg_diff_y.reshape(b,-1)
    
    coords_ul = coords_unroll(coords_ul, sr_ratio).type(torch.long).cuda(local_rank)
    coords_ur = coords_unroll(coords_ur, sr_ratio).type(torch.long).cuda(local_rank)
    coords_ll = coords_unroll(coords_ll, sr_ratio).type(torch.long).cuda(local_rank)
    coords_lr = coords_unroll(coords_lr, sr_ratio).type(torch.long).cuda(local_rank)
    
    dadd = torch.zeros(b, sr_ratio*sr_ratio*h*w).cuda(local_rank)
    dacc = torch.zeros(b, sr_ratio*sr_ratio*h*w).cuda(local_rank)

    dadd = dadd.scatter_add(1, coords_ul, samples_*neg_diff_x*neg_diff_y)
    dacc = dacc.scatter_add(1, coords_ul, neg_diff_x*neg_diff_y)


    dadd = dadd.scatter_add(1, coords_ur, samples_*diff_x*neg_diff_y)
    dacc = dacc.scatter_add(1, coords_ur, diff_x*neg_diff_y)


    dadd = dadd.scatter_add(1, coords_ll, samples_*neg_diff_x*diff_y)
    dacc = dacc.scatter_add(1, coords_ll, neg_diff_x*diff_y)


    dadd = dadd.scatter_add(1, coords_lr, samples_*diff_x*diff_y)
    dacc = dacc.scatter_add(1, coords_lr, diff_x*diff_y)

    return dadd.view(b, h*sr_ratio, sr_ratio*w), dacc.view(b,h*sr_ratio, sr_ratio*w)

def featureAdd(samples, flows, sr_ratio, local_rank):
    """
    samples: Tensor(b, h, w) float32
    flows: Tensor(b, 2, h, w) float32
    """

    flows_ = torch.empty_like(flows)
    flows_[:,0,...] = flows[:,1,...]
    flows_[:,1,...] = flows[:,0,...] #b*n, 2, h, w
    b, h, w = samples.shape

    samples_= samples.view(b, -1).type(torch.float32)

    mapping = sr_ratio*(flows_ + get_coords(h,w).cuda(local_rank))
    mappingy, mappingx = torch.split(mapping, 1, dim = 1)
    mappingy = torch.clamp(mappingy, 0, sr_ratio*h-1)
    mappingx = torch.clamp(mappingx, 0, sr_ratio*w-1)

    mapping = torch.cat((mappingy, mappingx), 1)

    coords_ul, coords_ur, coords_ll, coords_lr = get_neighbours(mapping) # all (b, 2, h, w)


    diff = (mapping - coords_ul).type(torch.float32).cuda(local_rank)
    neg_diff = (1.0 - diff).type(torch.float32).cuda(local_rank)
    diff_y, diff_x = torch.split(diff, 1, dim = 1)
    neg_diff_y, neg_diff_x = torch.split(neg_diff, 1, dim = 1)
    diff_x = diff_x.view(b,-1)
    diff_y = diff_y.view(b,-1)
    neg_diff_x = neg_diff_x.view(b,-1)
    neg_diff_y = neg_diff_y.view(b,-1)

    coords_ul = coords_unroll(coords_ul, sr_ratio).type(torch.long).cuda(local_rank)
    coords_ur = coords_unroll(coords_ur, sr_ratio).type(torch.long).cuda(local_rank)
    coords_ll = coords_unroll(coords_ll, sr_ratio).type(torch.long).cuda(local_rank)
    coords_lr = coords_unroll(coords_lr, sr_ratio).type(torch.long).cuda(local_rank)

    dadd = torch.zeros(b, sr_ratio*sr_ratio*h*w).cuda(local_rank)

    dadd = dadd.scatter_add(1, coords_ul, samples_*neg_diff_x*neg_diff_y)

    dadd = dadd.scatter_add(1, coords_ur, samples_*diff_x*neg_diff_y)

    dadd = dadd.scatter_add(1, coords_ll, samples_*neg_diff_x*diff_y)

    dadd = dadd.scatter_add(1, coords_lr, samples_*diff_x*diff_y)

    return dadd.view(b, h*sr_ratio, sr_ratio*w)

def featureWeight(flows, sr_ratio, local_rank):
    """
    samples: Tensor(b, h, w) float32
    flows: Tensor(b, 2, h, w) float32
    """

    flows_ = torch.empty_like(flows)
    flows_[:,0,...] = flows[:,1,...]
    flows_[:,1,...] = flows[:,0,...] #b*n, 2, h, w

    b, _, h, w = flows.shape

    mapping = sr_ratio*(flows_ + get_coords(h,w).cuda(local_rank))
    mappingy, mappingx = torch.split(mapping, 1, dim = 1)
    mappingy = torch.clamp(mappingy, 0, sr_ratio*h-1)
    mappingx = torch.clamp(mappingx, 0, sr_ratio*w-1)

    mapping = torch.cat((mappingy, mappingx), 1)

    coords_ul, coords_ur, coords_ll, coords_lr = get_neighbours(mapping) # all (b, 2, h, w)


    diff = (mapping - coords_ul).type(torch.float32).cuda(local_rank)
    neg_diff = (1.0 - diff).type(torch.float32).cuda(local_rank)
    diff_y, diff_x = torch.split(diff, 1, dim = 1)
    neg_diff_y, neg_diff_x = torch.split(neg_diff, 1, dim = 1)
    diff_x = diff_x.view(b,-1)
    diff_y = diff_y.view(b,-1)
    neg_diff_x = neg_diff_x.view(b,-1)
    neg_diff_y = neg_diff_y.view(b,-1)

    coords_ul = coords_unroll(coords_ul, sr_ratio).type(torch.long).cuda(local_rank)
    coords_ur = coords_unroll(coords_ur, sr_ratio).type(torch.long).cuda(local_rank)
    coords_ll = coords_unroll(coords_ll, sr_ratio).type(torch.long).cuda(local_rank)
    coords_lr = coords_unroll(coords_lr, sr_ratio).type(torch.long).cuda(local_rank)

    dacc = torch.zeros(b, sr_ratio*sr_ratio*h*w).cuda(local_rank)

    dacc = dacc.scatter_add(1, coords_ul, neg_diff_x*neg_diff_y)

    dacc = dacc.scatter_add(1, coords_ur, diff_x*neg_diff_y)

    dacc = dacc.scatter_add(1, coords_ll, neg_diff_x*diff_y)

    dacc = dacc.scatter_add(1, coords_lr, diff_x*diff_y)

    dacc = torch.where(dacc == 0.0, torch.full_like(dacc, 1.0), dacc)

    return dacc.view(b,h*sr_ratio, sr_ratio*w)


class TVL1(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVL1, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[-2]
        w_x = x.size()[-1]

        count_h = self._tensor_size(x[..., 1:, :])
        count_w = self._tensor_size(x[..., :, 1:])

        h_tv = torch.abs((x[..., 1:, :] - x[..., :h_x - 1, :])).sum()
        w_tv = torch.abs((x[..., :, 1:] - x[..., :, :w_x - 1])).sum()
        # print("h,w:", h_tv, w_tv)
        return self.TVLoss_weight * (h_tv / count_h + w_tv / count_w) / batch_size
        # return self.TVLoss_weight*(h_tv+w_tv)/batch_size

    def _tensor_size(self, t):
        return t.size()[-3] * t.size()[-2] * t.size()[-1]

def base_detail_decomp(samples, gaussian_filter):
    #samplesLR: b, num_im, h, w
    b, num_im, h, w = samples.shape
    base   = gaussian_filter(samples)
    detail = samples - base
    return base, detail #b, num_im, h, w

class WarpedLoss(nn.Module):
    def __init__(self, p=1, interpolation='bilinear'):
        super(WarpedLoss, self).__init__()
        if p == 1:
            self.criterion = nn.L1Loss(reduction='mean')  # change to reduction = 'mean'
        if p == 2:
            self.criterion = nn.MSELoss(reduction='mean')
        self.interpolation = interpolation

    def cubic_interpolation(self, A, B, C, D, x):
        a, b, c, d = A.size()
        x = x.view(a, 1, c, d)  # .repeat(1,3,1,1)
        return B + 0.5 * x * (C - A + x * (2. * A - 5. * B + 4. * C - D + x * (3. * (B - C) + D - A)))

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        if torch.sum(flo * flo) == 0:
            return x
        else:

            B, C, H, W = x.size()

            # mesh grid
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()
            grid = grid.to(device)
            # print(grid.shape)
            vgrid = Variable(grid) + flo.to(device)

            if self.interpolation == 'bilinear':
                # scale grid to [-1,1]
                vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
                vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

                vgrid = vgrid.permute(0, 2, 3, 1)
                output = nn.functional.grid_sample(x, vgrid, align_corners=True)

            if self.interpolation == 'bicubicTorch':
                # scale grid to [-1,1]
                vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
                vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

                vgrid = vgrid.permute(0, 2, 3, 1)
                output = nn.functional.grid_sample(x, vgrid, align_corners=True, mode='bicubic')

                # mask = torch.ones(x.size()).cuda()
                # mask = nn.functional.grid_sample(mask, vgrid,align_corners = True,mode = 'bicubic')

                # mask[mask < 0.9999] = 0
                # mask[mask > 0] = 1
            return output  # , mask

    def forward(self, input, target, flow, c, losstype='L1', masks=None):
        # Warp input on target
        warped = self.warp(target, flow)

        input_ = input[..., c:-c, c:-c]
        warped_ = warped[..., c:-c, c:-c]
        if losstype == 'HighRes-net':
            warped_ = warped_ / torch.sum(warped_, dim=(2, 3), keepdim=True) * torch.sum(input_, dim=(2, 3), keepdim=True)
        if losstype == 'Detail':
            _, warped_ = base_detail_decomp(warped_, Gaussian_Filter)
            _, input_ = base_detail_decomp(input_, Gaussian_Filter)

        if losstype == 'DetailReal':
            _, warped_ = base_detail_decomp(warped_, Gaussian_Filter)
            _, input_ = base_detail_decomp(input_, Gaussian_Filter)

            masks = masks[..., 2:-2, 2:-2]

            warped_ = warped_ * masks[:, :1] * masks[:, 1:]
            input_ = input_ * masks[:, :1] * masks[:, 1:]

        self.loss = self.criterion(input_, warped_)

        return self.loss, warped

class WarpedSpectral(nn.Module):
    def __init__(self, interpolation='bilinear'):
        super(WarpedSpectral, self).__init__()
        self.interpolation = interpolation

    def cubic_interpolation(self, A, B, C, D, x):
        a, b, c, d = A.size()
        x = x.view(a, 1, c, d)  # .repeat(1,3,1,1)
        return B + 0.5 * x * (C - A + x * (2. * A - 5. * B + 4. * C - D + x * (3. * (B - C) + D - A)))

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        if torch.sum(flo * flo) == 0:
            return x
        else:

            B, C, H, W = x.size()

            # mesh grid
            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()
            grid = grid.to(device)
            # print(grid.shape)
            vgrid = Variable(grid) + flo.to(device)

            if self.interpolation == 'bilinear':
                # scale grid to [-1,1]
                vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
                vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

                vgrid = vgrid.permute(0, 2, 3, 1)
                output = nn.functional.grid_sample(x, vgrid, align_corners=True)

            if self.interpolation == 'bicubicTorch':
                # scale grid to [-1,1]
                vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
                vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

                vgrid = vgrid.permute(0, 2, 3, 1)
                output = nn.functional.grid_sample(x, vgrid, align_corners=True, mode='bicubic')

                # mask = torch.ones(x.size()).cuda()
                # mask = nn.functional.grid_sample(mask, vgrid,align_corners = True,mode = 'bicubic')

                # mask[mask < 0.9999] = 0
                # mask[mask > 0] = 1
            return output  # , mask

    def forward(self, input, target, flow, losstype='L1'):
        # Warp input on target
        warped = self.warp(target, flow)
        input_ = input
        warped_ = warped
        if losstype == 'HighRes-net':
            warped_ = warped / torch.sum(warped, dim=(2, 3), keepdim=True) * torch.sum(input, dim=(2, 3), keepdim=True)
        if losstype == 'Detail':
            _, warped_ = base_detail_decomp(warped, Gaussian_Filter)
            _, input_ = base_detail_decomp(input, Gaussian_Filter)

        if losstype == 'DetailReal':
            _, warped_ = base_detail_decomp(warped, Gaussian_Filter)
            _, input_ = base_detail_decomp(input, Gaussian_Filter)

        return warped_, input_

k = np.load("/home/cyt/SSL-HSR-2/warping/blur_kernel.npy").squeeze()
size = k.shape[0]

class BlurLayer(nn.Module):
    def __init__(self):
        super(BlurLayer, self).__init__()
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(size//2),
            nn.Conv2d(1, 1, size, stride=1, padding=0, bias=None, groups=1)
        )

        self.weights_init()
    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))
            f.required_grad = False
