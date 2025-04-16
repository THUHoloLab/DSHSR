import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import argparse
from torchvision.transforms import GaussianBlur
import torch.nn as nn
import datetime
import os
import torch.backends.cudnn as cudnn
from network.Encoder_Decoder import CoarseFuseNet,FNet
from network.MultiSSE import SUmodule, SFmodule, KernelNet, Decorder
import torch.nn.functional as F
import tempfile
from loss.loss import Pyramid,PSNRM
from loss.kernel_loss import SparsityLoss
from synthetic_dataset.HSIdataset import HSIdataset
from synthetic_dataset.synthetic_hsi_train_set import SyntheticHSI
from time import time
from numpy import mean
from cv2 import imwrite
from synthetic_dataset.data_format_utils import torch_to_numpy, numpy_to_torch
from torch.autograd import Variable
from collections import OrderedDict
from warping.shiftandadd import featureAdd, featureWeight,WarpedLoss,TVL1
from network.merging import Weighted
from network.Encoder_Decoder import EncoderNet, Sharpmodule
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= nprocs
    return rt

def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.makedirs(path)
    except OSError:
        pass

def guided_filter(I, p, r, eps):
    return cv2.ximgproc.guidedFilter(I, p, r, eps)

# 例子：假设 I 是引导图像，p 是输入图像
'''I = cv2.imread("/home/cyt/cyt/HSI/chart_and_stuffed_toy_ms_17.png")
p = cv2.imread("/home/cyt/cyt/HSI/chart_and_stuffed_toy_ms_6.png")
result = guided_filter(I, p, r=5, eps=1e-3)
cv2.imwrite("/home/cyt/cyt/GF-result/GF.png", result)'''
def guided_filter_confidence(I, p, r, eps, gvar=0.2):
    GF_img = guided_filter(I, p, r, eps)
    Filter_img = cv2.boxFilter(p, ddepth=-1, ksize=(r,r))
    res = r**2*pow(GF_img-Filter_img,2)
    L = np.exp(-res/gvar)
    L = cv2.boxFilter(L, ddepth=-1, ksize=(r,r))
    return L


Pyr = Pyramid(num_levels=1, pyr_mode='lap')
def base_detail_decomp(samples, local_rank):
    #samplesLR: b, num_im, h, w
    #b, num_im, h, w = samples.shape
    '''
    base   = gaussian_filter(samples)
    detail = samples - base
    '''
    #Pyr = Pyramid(num_levels=1, pyr_mode='lap')
    #detail = laplacian_highpass(samples)
    detail = Pyr(samples, local_rank)
    detail = detail[0]
    base = samples - detail
    return base, detail #b, num_im, h, w

def conv_func(input, kernel, padding='same'):
        b, c, h, w = input.size()
        assert b == 1, "only support b=1!"
        _, _, ksize, ksize = kernel.size()
        if padding == 'same':
            pad = ksize // 2
        elif padding == 'valid':
            pad = 0
        else:
            raise Exception("not support padding flag!")

        conv_result_list = []
        for i in range(c):
            conv_result_list.append(F.conv2d(input[:, i:i + 1, :, :], kernel, bias=None, stride=1, padding=pad))
        conv_result = torch.cat(conv_result_list, dim=1)
        return conv_result

def blur_func(x, kernel):
        b, c, h, w = x.size()
        _, kc, ksize, _ = kernel.size()
        psize = ksize // 2
        assert kc == 1, "only support kc=1!"

        # blur
        x = F.pad(x, (psize, psize, psize, psize), mode='replicate')
        blur_list = []
        for i in range(b):
            blur_list.append(conv_func(x[i:i + 1, :, :, :], kernel[i:i + 1, :, :, :]))
        blur = torch.cat(blur_list, dim=0)
        blur = blur[:, :, psize:-psize, psize:-psize]

        return blur

def BilinearWarping(x, local_rank, flo, ds_factor=8):
    """
    warp and downsample an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    根据im2到im1的光流，采用反向变形
    """
    if torch.sum(flo * flo) == 0:
        return x[..., ::ds_factor, ::ds_factor]
    else:
        B, _, H, W = flo.size()

        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        grid = grid.cuda(local_rank)
        # print(grid.shape)
        vgrid = ds_factor * (Variable(grid) + flo)
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(ds_factor * W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(ds_factor * H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)

        output = nn.functional.grid_sample(x, vgrid, align_corners=True, mode='bicubic', padding_mode='reflection')

        return output
def create_groups(num_spec, reference_index, is_training=True, num_bands=10):
    indices = []
    if reference_index < num_bands // 2:
        indices.extend(range(num_bands + 1))
        del indices[reference_index]
    elif reference_index > (num_spec - num_bands // 2 - 1):
        indices.extend(range(num_spec - num_bands - 1, num_spec))
        del indices[reference_index - num_spec + num_bands]
    else:
        for i in range(1, (num_bands // 2) + 1):
            # 向前选取波段，注意要循环选取
            indices.append((reference_index - i) % num_spec)
            # 向后选取波段，注意要循环选取
            indices.append((reference_index + i) % num_spec)
    if not is_training:
        indices.append(reference_index)

    indices = sorted(indices)
    indices = indices[: num_bands]
    remaining_indices = [i for i in range(num_spec) if i not in indices]
    remaining_indices.sort(key=lambda x: abs(x - reference_index))
    closest_indices = remaining_indices[:10]
    #sorted_indices_by_distance = sorted(indices, key=lambda x: abs(x - reference_index))
    #nearest_indices = sorted_indices_by_distance[:2]
    #nearest_indices = [reference_index, nearest_indices]

    return indices, closest_indices

def find_farthest_indices(C, batch_size, num_indices=10):
    # 生成所有可能的波段索引,每一行都是[1,2,...]
    indices = torch.arange(C).unsqueeze(0).repeat(C, 1)  # 形状为 (C, C)
    # 生成参考索引，每一行都是 reference_index
    reference_indices = torch.arange(C).unsqueeze(1).repeat(1, C)  # 形状为 (C, C)
    # 计算每个参考索引与所有波段的距离
    distances = torch.abs(indices - reference_indices)

    # 选取最远的 num_indices 个索引
    value_indices,_  = torch.topk(distances, num_indices+10, largest=False, dim=1)  # 形状为 (C, num_indices)
    value_indices = value_indices[:,10:]
    # 扩展为三维张量 (batch_size, C, num_indices)
    farthest_indices_tensor = value_indices.unsqueeze(0).repeat(batch_size, 1, 1).cuda()

    return farthest_indices_tensor
def S_function(neighbors, center_pixels):
    A = 0.023*torch.exp(-2*torch.pow(torch.abs(center_pixels-0.5)/0.42, 5.545))
    SCB_tensor=(neighbors - center_pixels)**2/((neighbors - center_pixels)**2+A)
    return SCB_tensor
def SCB_transform(input_tensor):
    b, n, h, w = input_tensor.size()
    input_tensor = input_tensor.view(b*n, 1, h, w)
    # 扩展输入张量，并用0填充
    padding=1
    padded_input = F.pad(input_tensor, (padding, padding, padding, padding), mode='replicate')
    neighbors=F.unfold(padded_input, kernel_size=3, padding=0)#[b*n,9,h*w]
    neighbors = neighbors.transpose(1, 2)#[b*n,h*w,9]
    # 提取邻域中心像素 (即 3x3 窗口的中间像素)
    center_pixels = neighbors[:, :, 4:5]

    # 展开邻域矩阵和中心像素矩阵为相同形状，便于计算
    neighbors = neighbors.view(b, n, h, w, 9)
    center_pixels = center_pixels.view(b, n, h, w, 1)

    # 计算每个邻域像素与中心像素的 S 函数
    diff = S_function(neighbors, center_pixels)
    # 将中间像素 (即窗口中心) 的差异置为零
    diff[:, :, :, :, 4:5] = 0  # 将窗口中心的 S 函数值设置为零 (即位置 (1, 1))

    # 对于每个像素，计算 3x3 窗口内所有像素的 S 函数值总和
    output_tensor = diff.sum(dim=-1)/8  # 按最后一个维度求和，得到每个像素的结果

    return output_tensor

def flowEstimation(samplesLR, local_rank, ME, c, warping, gaussian_filter, losstype):
    """
    Compute the optical flows from the other frames to the reference:
    samplesLR: Tensor b, num_im, h, w
    ME: Motion Estimator
    """

    b, num_im, h, w = samplesLR.shape
    samplesLR = SCB_transform(samplesLR)

    samplesLRblur = gaussian_filter(samplesLR)  # 去噪和混叠，提高运动估计准确性

    samplesLR_0 = samplesLRblur[:, :1, ...]  # b, 1, h, w 消除混叠伪影和噪声

    samplesLR_0 = samplesLR_0.repeat(1, num_im, 1, 1)  # b, num_im, h, w
    samplesLR_0 = samplesLR_0.reshape(-1, h, w)
    samplesLRblur = samplesLRblur.reshape(-1, h, w)  # b*num_im, h, w
    samplesLRblur = samplesLRblur.unsqueeze(1)
    samplesLR_0 = samplesLR_0.unsqueeze(1)
    concat = torch.cat((samplesLRblur, samplesLR_0), dim=1)  # b*(num_im), 4, h, w
    flow = ME(concat.cuda(local_rank))  # b*(num_im), 2, h, w
    flow[::num_im] = 0

    warploss, warped = warping(samplesLRblur, samplesLR_0, flow, c, losstype=losstype)

    return flow.reshape(b, num_im, 2, h, w), warploss, warped

def zoombase(LR_base, flow, local_rank, sr_ratio):
    warping = WarpedLoss(interpolation='bicubicTorch')
    b, num_im, h, w = LR_base.shape
    LR_base = LR_base.view(-1, 1, h, w)
    LR_base = warping.warp(LR_base, -flow.view(-1, 2, h, w))

    SR_base = torch.nn.functional.interpolate(LR_base, size=[sr_ratio * h, sr_ratio * w], mode='bilinear', align_corners=True)
    SR_base = SR_base.view(b, num_im, sr_ratio * h, sr_ratio * w)
    return SR_base

class HSISR(nn.Module):
    def __init__(self, local_rank, num_features1=32, num_features2=16, ksize=13):
        super(HSISR, self).__init__()
        self.encoder = EncoderNet(in_dim=1,conv_dim=num_features1, out_dim=num_features1, num_blocks=2)
        self.CoarseFuse = Weighted(input_dim=num_features1, project_dim=num_features2,local_rank=local_rank)
        '''
        self.conv_1 = nn.Conv2d(in_channels=num_features1, out_channels=num_features1, kernel_size=3, stride=1,
                                padding=1, padding_mode='reflect')
        self.conv_2 = nn.Conv2d(in_channels=num_features1, out_channels=num_features1, kernel_size=3, stride=1,
                                padding=1, padding_mode='reflect')
        '''
        #self.decoder = DecoderNet()
        self.ME = FNet().float()
        self.FineFuse = SFmodule(num_features1)
        self.BlueEstimation = KernelNet(in_c=1, ksize=ksize)
        self.SharpGray = Sharpmodule(in_dim=num_features2)
        self.num_features1 = num_features1
        self.num_features2 = num_features2
        #self.Conv = nn.Conv2d(in_channels=num_features1, out_channels=num_features2, kernel_size=3, stride=1, padding=1,padding_mode='reflect')
        self.local_rank = local_rank

    def forward(self, samplesHSI, boundary, indices,sr_ratio=4):
        batch, num_spec, h, w = samplesHSI.shape
        flow, _, warped = flowEstimation(samplesHSI, self.local_rank, self.ME, boundary,
                                                WarpedLoss(interpolation='bicubicTorch'),
                                                gaussian_filter=GaussianBlur(11, sigma=1), losstype='HighRes-net')

        '''for n in range(num_spec):
            for m in range(n+1, num_spec):
                confidence = confidences[0, n, m]
                confidence = confidence.detach().cpu().numpy().squeeze()
                confidence = (confidence * 255).astype('uint8')
                imwrite(os.path.join("/home/cyt/cyt/CC/", "CC_{:02d}_{:02d}.png".format(n, m)), confidence)'''

        gaussian_filter = GaussianBlur(11, sigma=1).cuda()
        HSIbase, HSIdetail = base_detail_decomp(samplesHSI, self.local_rank)
        HSIbase = zoombase(HSIbase, flow, self.local_rank, sr_ratio)
        HSIbase = self.SharpGray(HSIbase.view(batch*num_spec, 1, sr_ratio*h, sr_ratio*w)).view(batch, num_spec, sr_ratio*h, sr_ratio*w)

        features = self.encoder(HSIdetail.view(-1, 1, h, w)).view(batch, num_spec, self.num_features1, h,
                                                                  w)  # b, num_spec, num_features, h, w
        flowf = flow.contiguous().view(-1, 1, 2, h, w).repeat(1, self.num_features1, 1, 1, 1).view(-1, 2, h, w)

        warp_weight = featureWeight(flowf, sr_ratio=1, local_rank=self.local_rank)
        features_warp = featureAdd(features.contiguous().view(-1, h, w), flowf, sr_ratio=1, local_rank=self.local_rank)
        features_warp = features_warp / warp_weight
        features_warp = features_warp.view(batch, num_spec, self.num_features1, h, w)

        warp_weight = featureWeight(flowf, sr_ratio=sr_ratio, local_rank=self.local_rank)
        features_sr = featureAdd(features.contiguous().view(-1, h, w), flowf, sr_ratio=sr_ratio, local_rank=self.local_rank)
        features_sr = features_sr / warp_weight
        features_sr = features_sr.view(batch, num_spec, self.num_features1, sr_ratio * h, sr_ratio * w)

        weight_fuse = []
        for i in range(num_spec):
            indice = indices[i]
            confidence_features = self.CoarseFuse(features_warp[:,i:i+1],features_warp[:,indice])
            confidence_features = F.interpolate(confidence_features.view(-1,self.num_features1, h, w), size=(sr_ratio * h, sr_ratio * w), mode='bilinear', align_corners=False)
            confidence_features = confidence_features.view(batch, 10, self.num_features1, sr_ratio * h, sr_ratio * w) * features_sr[:, indice]
            confidence_features = confidence_features.sum(dim=1)
            weight_fuse.append(confidence_features)

        weight_fuse = torch.stack(weight_fuse, dim=1) # batch,num_spec, self.num_features1, h, w
        weight_fuse = weight_fuse.permute(0, 2, 1, 3, 4).contiguous()
        weight_fuse = self.FineFuse(weight_fuse) # b, self.num_features/2, num_spec, 2h, 2w
        weight_fuse = weight_fuse.squeeze(1) # b, 1, num_spec, 8h, 8w to b, num_spec, 8h, 8w
        HRHSI = weight_fuse + HSIbase
        HRHSIBlur = HRHSI.detach()
        HRHSIBlur = HRHSIBlur.view(batch*num_spec, 1, sr_ratio * h, sr_ratio * w)
        #HRHSIBlur = HRHSI.view(batch * num_spec, 1, sr_ratio * h, sr_ratio * w)
        est_kernel = self.BlueEstimation(HRHSIBlur) # b, num_spec, 8h, 8w to b*num_spec, 1, 8h, 8w
        HRHSIBlur = blur_func(HRHSIBlur, est_kernel)
        HRHSIBlur = HRHSIBlur.view(batch, num_spec, sr_ratio * h, sr_ratio * w)
        return weight_fuse, HRHSI, HRHSIBlur, est_kernel,flow



def train(local_rank, world_size, args):
    train_bs, val_bs, num_features, ksize, gray_max, sr_ratio = args.train_bs, args.val_bs, args.num_features, args.kernel_size, args.gray_max, args.sr_ratio
    folder_name = 'self-supervised_multi-image_deepSR_time_{}'.format(f"{datetime.datetime.now():%m-%d-%H-%M-%S}")

    ################## load Models
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    cudnn.benchmark = True
    args.local_rank = local_rank
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='tcp://127.0.0.1:4456',
                                         world_size=args.world_size,
                                         rank=local_rank)
    torch.cuda.set_device(local_rank)
    train_bs = int(train_bs / args.world_size)
    val_bs = int(val_bs / args.world_size)
    SSLHSISR = HSISR(local_rank=local_rank, num_features1=num_features, num_features2=int(num_features/2), ksize=ksize).cuda(local_rank)

    checkpoint_path = os.path.join("/home/cyt/", "initial_weights.pt")
    if local_rank == 0:
        torch.save(SSLHSISR.state_dict(), checkpoint_path)
    torch.distributed.barrier()
    SSLHSISR.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cuda', local_rank)))

    checkpoint_path_ME = args.model_ME_path
    checkpoint = torch.load(checkpoint_path_ME, map_location=torch.device('cuda', local_rank))
    state_dictME = checkpoint['state_dict ME']
    new_state_dictME = OrderedDict()
    for k, v in state_dictME.items():
        name = k[len('module.' + 'ME') + 1:]  # remove 'module.' of dataparallel
        new_state_dictME[name] = v
    SSLHSISR.ME.load_state_dict(new_state_dictME)

    for param in SSLHSISR.ME.parameters():
        param.requires_grad = False

    optimizer_DeepSR = torch.optim.AdamW(SSLHSISR.parameters(), lr=args.lr_DeepSR, weight_decay=args.weight_decay)
    schedulerDeepSR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_DeepSR, args.num_epochs, eta_min=1e-6)
    if args.Sync_BN:
        # time-consuming using SyncBatchNorm
        SSLHSISR = torch.nn.SyncBatchNorm.convert_sync_batchnorm(SSLHSISR).cuda(local_rank)
    SSLHSISR = torch.nn.parallel.DistributedDataParallel(SSLHSISR, device_ids=[local_rank], broadcast_buffers=False)

    Dataset_path = args.train_file
    train_gthsi_1 = HSIdataset(root=Dataset_path, split='train_1')
    train_data_set_1 = SyntheticHSI(train_gthsi_1, downfactor=sr_ratio, crop_sz=args.train_patchsize, center_crop=False, gray_max=gray_max, offset_max=args.offset_max)
    train_sampler_1 = torch.utils.data.distributed.DistributedSampler(train_data_set_1)
    nw = min([os.cpu_count(), train_bs if train_bs >= 1 else 0, 8])  # number of workers
    train_loader_1 = torch.utils.data.DataLoader(train_data_set_1,
                                               batch_size=train_bs,
                                               sampler=train_sampler_1,
                                               pin_memory=True,
                                               num_workers=0)

    val_gthsi = HSIdataset(root=Dataset_path, split='val')
    val_data_set = SyntheticHSI(val_gthsi, downfactor=sr_ratio, crop_sz=args.val_patchsize, center_crop=True, gray_max=gray_max, offset_max=args.offset_max)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data_set)
    nw = min([os.cpu_count(), val_bs if val_bs >= 1 else 0, 8])
    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=val_bs,
                                             sampler=val_sampler,
                                             pin_memory=True,
                                             num_workers=0)

    checkpoint_dir = '/TrainHistory/{}'.format(folder_name)
    result_dir_SR = '/result_dir_SR/{}'.format(folder_name)
    result_dir_LR = "/result_dir_LR/{}".format(folder_name)
    result_dir_GT = "/result_dir_GT/{}".format(folder_name)
    safe_mkdir(checkpoint_dir)
    safe_mkdir(result_dir_SR)
    safe_mkdir(result_dir_LR)
    safe_mkdir(result_dir_GT)
    boundary = 5
    psnr_fn = PSNRM(boundary_ignore=boundary, max_value=1.0)
    criterion = nn.L1Loss()
    kernel_loss_sparse = SparsityLoss()
    weight=50.0
    starttime = time()
    ##################
    for epoch in range(args.num_epochs):
        train_sampler_1.set_epoch(epoch)
        TrainLoss = []
        ValLoss = []
        PSNRLoss = []
        '''
        if epoch<50:
            weight = weight_initial - epoch // 10
        else:
            weight = 1
        '''

        if local_rank == 0:
            print('__________________________________________________')
            print('Training epoch {0:3d}'.format(epoch))
        for i, data in enumerate(train_loader_1):
            """
            samplesLR : b, num_spec, h, w
            SR:  b, num_spec, 8*h, 8*w
            """
            samplesLR, samplesLR_blur, flow_gt, gt = data
            optimizer_DeepSR.zero_grad()
            samplesLR = samplesLR.float().cuda(local_rank)
            samplesLR_blur = samplesLR_blur.float().cuda(local_rank)
            gt = gt.float().cuda(local_rank)
            flow_gt = flow_gt.float().cuda(local_rank)
            batch, num_spec, h, w = samplesLR.shape
            remains_indices = []
            indices = []
            for i in range(num_spec):
                indice, remain_indice = create_groups(num_spec, i, True)
                remains_indices.append(remain_indice)
                indices.append(indice)

            samplesLRbase, samplesLRdetail = base_detail_decomp(samplesLR, local_rank) #gaussian_filter
            warp_weight = featureWeight(flow_gt.view(-1, 2, h, w), sr_ratio=1, local_rank=local_rank)
            forw_warp = featureAdd(samplesLR.contiguous().view(-1, h, w), flow_gt.view(-1, 2, h, w), sr_ratio=1,
                                   local_rank=local_rank)
            forw_warp = forw_warp / warp_weight
            forw_warp = forw_warp.view(batch, num_spec, h, w)
            confidences = np.ones((batch, num_spec, num_spec, h, w))
            for b in range(batch):
                for i in range(num_spec):
                    for j in range(num_spec):
                        if j != i:
                            confidence = guided_filter_confidence(forw_warp[b, i, ...].detach().cpu().numpy(),
                                                                  forw_warp[b, j, ...].detach().cpu().numpy(), 5,
                                                                  0.01)
                            confidences[b, i, j] = confidence


            confidences = numpy_to_torch(confidences).cuda()
            for b in range(batch):
                for i in range(num_spec):
                    for j in range(i + 1, num_spec):
                        confidences[b, i, j] = torch.min(confidences[b, i, j], confidences[b, j, i])
                        confidences[b, j, i] = confidences[b, i, j]

            HRHSIdetail, HRHSI, HRHSIBlur, est_kernel, flow = SSLHSISR(samplesLR_blur, boundary, indices,sr_ratio=sr_ratio)
            HRHSI = HRHSI.clamp(0.0, 1.0)
            confidences_warp = BilinearWarping(confidences.contiguous().view(-1, 1, h, w), args.local_rank,
                                               flow.unsqueeze(1).repeat(1, num_spec, 1, 1, 1, 1).view(-1, 2, h, w),
                                               1).view(batch, num_spec, num_spec, h, w)

            HRHSIdown_blur = BilinearWarping(HRHSIBlur.view(batch*num_spec, 1,  sr_ratio * h, sr_ratio * w), local_rank,flow.view(-1, 2, h, w), sr_ratio).view(batch,num_spec, h, w)
            HRHSIdown = BilinearWarping(HRHSI.view(batch*num_spec, 1,  sr_ratio * h, sr_ratio * w), local_rank,flow.view(-1, 2, h, w), sr_ratio).view(batch,num_spec, h, w)
            HRHSIdown_detail = BilinearWarping(HRHSIdetail.unsqueeze(2).repeat(1, 1, num_spec, 1, 1).view(-1, 1, sr_ratio * h, sr_ratio * w), local_rank,
                                        flow.unsqueeze(1).repeat(1, num_spec, 1, 1, 1, 1).view(-1, 2, h, w), sr_ratio).view(batch, num_spec, num_spec, h, w)

            samplesLRdetail = samplesLRdetail.unsqueeze(1).repeat(1, num_spec, 1, 1, 1)
            remains_indices = torch.tensor(remains_indices).unsqueeze(0).repeat(batch, 1, 1).cuda()
            remains_indices = remains_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, h, w)
            HRHSIdown_detail = torch.gather(HRHSIdown_detail, 2, remains_indices)
            samplesLRdetail = torch.gather(samplesLRdetail, 2, remains_indices)
            confidences_warp = torch.gather(confidences_warp, 2, remains_indices)

            trainloss_detail = criterion(confidences_warp[..., boundary:-boundary, boundary:-boundary]*HRHSIdown_detail[..., boundary:-boundary, boundary:-boundary], confidences_warp[..., boundary:-boundary, boundary:-boundary]*samplesLRdetail[..., boundary:-boundary, boundary:-boundary])
            trainloss_base_blur = criterion(HRHSIdown_blur[..., boundary:-boundary, boundary:-boundary], samplesLR_blur[..., boundary:-boundary, boundary:-boundary])
            trainloss_base = criterion(HRHSIdown[..., boundary:-boundary, boundary:-boundary], samplesLR[..., boundary:-boundary, boundary:-boundary])
            kloss_sparse = kernel_loss_sparse(est_kernel)

            trainloss =weight*trainloss_detail + trainloss_base_blur+trainloss_base+0.04*kloss_sparse
            trainloss.backward()
            optimizer_DeepSR.step()
            reduce_trainloss = reduce_mean(trainloss, args.world_size)
            TrainLoss.append(reduce_trainloss.data.item())

        if local_rank == 0:
            print('Train')
            print('{:.5f}'.format(mean(TrainLoss)))
            sys.stdout.flush()

        SSLHSISR.eval()
        with torch.no_grad():
            for k, data in enumerate(val_loader):
                samplesLR, samplesLR_blur, flow_gt, gt = data
                optimizer_DeepSR.zero_grad()
                samplesLR = samplesLR.float().cuda(local_rank)
                samplesLR_blur = samplesLR_blur.float().cuda(local_rank)
                gt = gt.float().cuda(local_rank)
                flow_gt = flow_gt.float().cuda(local_rank)
                batch, num_spec, h, w = samplesLR.shape
                remains_indices = []
                indices = []
                for i in range(num_spec):
                    indice, remain_indice = create_groups(num_spec, i, True)
                    remains_indices.append(remain_indice)
                    indices.append(indice)

                samplesLRbase, samplesLRdetail = base_detail_decomp(samplesLR, local_rank)  # gaussian_filter
                warp_weight = featureWeight(flow_gt.view(-1, 2, h, w), sr_ratio=1, local_rank=local_rank)
                forw_warp = featureAdd(samplesLR.contiguous().view(-1, h, w), flow_gt.view(-1, 2, h, w), sr_ratio=1,
                                       local_rank=local_rank)
                forw_warp = forw_warp / warp_weight
                forw_warp = forw_warp.view(batch, num_spec, h, w)
                confidences = np.ones((batch, num_spec, num_spec, h, w))
                for b in range(batch):
                    for i in range(num_spec):
                        for j in range(num_spec):
                            if j != i:
                                confidence = guided_filter_confidence(forw_warp[b, i, ...].detach().cpu().numpy(),
                                                                      forw_warp[b, j, ...].detach().cpu().numpy(), 5,
                                                                      0.01)
                                confidences[b, i, j] = confidence

                confidences = numpy_to_torch(confidences).cuda()
                for b in range(batch):
                    for i in range(num_spec):
                        for j in range(i + 1, num_spec):
                            confidences[b, i, j] = torch.min(confidences[b, i, j], confidences[b, j, i])
                            confidences[b, j, i] = confidences[b, i, j]

                HRHSIdetail, HRHSI, HRHSIBlur, est_kernel, flow = SSLHSISR(samplesLR_blur, boundary, indices,
                                                                           sr_ratio=sr_ratio)
                HRHSI = HRHSI.clamp(0.0, 1.0)
                confidences_warp = BilinearWarping(confidences.contiguous().view(-1, 1, h, w), args.local_rank,
                                                   flow.unsqueeze(1).repeat(1, num_spec, 1, 1, 1, 1).view(-1, 2, h, w),
                                                   1).view(batch, num_spec, num_spec, h, w)

                HRHSIdown_blur = BilinearWarping(HRHSIBlur.view(batch * num_spec, 1, sr_ratio * h, sr_ratio * w),
                                                 local_rank, flow.view(-1, 2, h, w), sr_ratio).view(batch, num_spec, h,
                                                                                                    w)
                HRHSIdown = BilinearWarping(HRHSI.view(batch * num_spec, 1, sr_ratio * h, sr_ratio * w), local_rank,
                                            flow.view(-1, 2, h, w), sr_ratio).view(batch, num_spec, h, w)
                HRHSIdown_detail = BilinearWarping(
                    HRHSIdetail.unsqueeze(2).repeat(1, 1, num_spec, 1, 1).view(-1, 1, sr_ratio * h, sr_ratio * w),
                    local_rank,
                    flow.unsqueeze(1).repeat(1, num_spec, 1, 1, 1, 1).view(-1, 2, h, w), sr_ratio).view(batch, num_spec,
                                                                                                        num_spec, h, w)

                samplesLRdetail = samplesLRdetail.unsqueeze(1).repeat(1, num_spec, 1, 1, 1)
                remains_indices = torch.tensor(remains_indices).unsqueeze(0).repeat(batch, 1, 1).cuda()
                remains_indices = remains_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, h, w)
                HRHSIdown_detail = torch.gather(HRHSIdown_detail, 2, remains_indices)
                samplesLRdetail = torch.gather(samplesLRdetail, 2, remains_indices)
                confidences_warp = torch.gather(confidences_warp, 2, remains_indices)

                valloss_detail = criterion(
                    confidences_warp[..., boundary:-boundary, boundary:-boundary] * HRHSIdown_detail[...,
                                                                                    boundary:-boundary,
                                                                                    boundary:-boundary],
                    confidences_warp[..., boundary:-boundary, boundary:-boundary] * samplesLRdetail[...,
                                                                                    boundary:-boundary,
                                                                                    boundary:-boundary])
                valloss_base_blur = criterion(HRHSIdown_blur[..., boundary:-boundary, boundary:-boundary],
                                                samplesLR_blur[..., boundary:-boundary, boundary:-boundary])
                valloss_base = criterion(HRHSIdown[..., boundary:-boundary, boundary:-boundary],
                                           samplesLR[..., boundary:-boundary, boundary:-boundary])
                kloss_sparse = kernel_loss_sparse(est_kernel)

                valloss = weight*valloss_detail + valloss_base_blur + valloss_base +0.04 * kloss_sparse
                reduce_valloss = reduce_mean(valloss, args.world_size)
                ValLoss.append(reduce_valloss.data.item())

                gt = gt.float().cuda(local_rank)
                PSNR = psnr_fn(HRHSI, gt)
                reduce_PSNR = reduce_mean(PSNR, args.world_size)
                PSNRLoss.append(reduce_PSNR.data.item())

            torch.cuda.synchronize(torch.device('cuda', local_rank))
            if epoch % 5 == 0:
                if local_rank == 0:
                    #warped = warped[0, ...]
                    HRHSI = HRHSI[0, ...]
                    HRHSIdetail = HRHSIdetail[0, ...]
                    #samplesLRdetail = samplesLRdetail[0, ...]
                    b, num_spec, h, w = samplesLR_blur.shape
                    samplesLR_blur = samplesLR_blur[0, ...]
                    gt = gt[0, ...]
                    for n in range(num_spec):
                        LR = samplesLR_blur[n, ...]
                        LR = LR.detach().cpu().numpy().squeeze()
                        LR = (LR * 255).astype('uint8')
                        imwrite(os.path.join(result_dir_LR, "LR_{:03d}_{:02d}.png".format(epoch, n)), LR)

                        HR = HRHSI[n, ...]
                        HR = HR.detach().cpu().numpy().squeeze()
                        HR = (HR * 255).astype('uint8')
                        imwrite(os.path.join(result_dir_SR, "HR_{:03d}_{:02d}.png".format(epoch, n)), HR)
                        gt_img = gt[n, ...]
                        gt_img = gt_img.detach().cpu().numpy().squeeze()
                        gt_img = (gt_img * 255).astype('uint8')
                        imwrite(os.path.join(result_dir_GT, "GT_{:03d}_{:02d}.png".format(epoch, n)), gt_img)




        if local_rank == 0:
            print('Val')
            print('{:.5f}'.format(mean(ValLoss)))
            print('PSNR')
            print('{:.5f}'.format(mean(PSNRLoss)))
            sys.stdout.flush()

        schedulerDeepSR.step()
        if epoch % 5 == 0:
            if local_rank == 0:
                print('#### Saving Models ... ####')
                print('#### Saving Models ... ####')
                state = {'epoch': epoch + 1,
                         'state_dict DeepSR': SSLHSISR.state_dict(),
                         'optimizerDeepSR': optimizer_DeepSR.state_dict(),
                         'schedulerDeepSR': schedulerDeepSR.state_dict()}
                torch.save(state, os.path.join(checkpoint_dir, 'checkpoint_{}.tar'.format(epoch)))


    if local_rank == 0:
        print('Execution time = {:.0f}s'.format(time() - starttime))
    if local_rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    torch.distributed.destroy_process_group()
    return


def main(args):
    """
        Given a configuration, trains Encoder, Decoder and fnet for Multi-Frame Super Resolution (MFSR), and saves best model.
        Args:
            config: dict, configuration file
    """
    torch.cuda.empty_cache()
    mp.spawn(train, nprocs=args.world_size, args=(args.world_size, args))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-bst", "--train_bs", help="Batch size of train loader", type=int, default=4)
    parser.add_argument("-bsv", "--val_bs", help="Batch size of val loader", type=int, default=4)
    parser.add_argument("-nf", "--num_features", help="Num of features for encoder", type=int, default=32)
    parser.add_argument("-ls", "--lr_DeepSR", help="Learning rate of DeepSR", type=float, default=1e-4)
    parser.add_argument("-lm", "--lr_ME", help="Learning rate of DeepSR", type=float, default=1e-5)
    parser.add_argument("-ks", "--kernel_size", help="Size of blur kernel", type=int, default=13)
    parser.add_argument("-ne", "--num_epochs", help="Num_epochs", type=int, default=100)
    parser.add_argument('-wd', '--weight-decay', help='weight decay (default: 1e-6)', type=float, default=1e-6)
    parser.add_argument('-tps', '--train_patchsize', help="the size of crop for training", default=192)
    parser.add_argument('-vps', '--val_patchsize', help="the size of crop for val", default=192)
    parser.add_argument("-gm", "--gray_max", help="the max of gray", type=float, default=65535.0)
    parser.add_argument("-om", "--offset_max", help="the max of offset", type=int, default=16)
    parser.add_argument("-cme", "--model_ME_path", help="the checkpoint of ME module", type=str, default="/checkpoint_ME.tar")
    parser.add_argument("-tf", "--train_file", help="the file of train data", type=str, default="/HSI-data/")
    parser.add_argument('-wz', '--world_size', default=2, type=int, help='number of distributed processes')
    parser.add_argument('-sbn', '--Sync_BN', help="Synchronization of BN across multi-gpus", default=True)
    parser.add_argument("-srr", "--sr_ratio", help="Super-resolution factor", type=int, default=4)
    args = parser.parse_args()
    main(args)



