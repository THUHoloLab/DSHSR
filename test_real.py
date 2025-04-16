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
from synthetic_dataset.HSIdataset import HSIdataset, HSISRDataset
from synthetic_dataset.synthetic_hsi_train_set import SyntheticHSI
from time import time
from numpy import mean
from cv2 import imwrite
from synthetic_dataset.data_format_utils import torch_to_numpy, numpy_to_torch
from torch.autograd import Variable
from warping.shiftandadd import featureAdd, featureWeight,WarpedLoss,TVL1
from network.merging import Weighted
from network.Encoder_Decoder import EncoderNet, Sharpmodule
import sys
from real_dataset.burstsr_dataset import BurstSRDataset


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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

        gaussian_filter = GaussianBlur(11, sigma=1).cuda()
        HSIbase, HSIdetail = base_detail_decomp(samplesHSI, self.local_rank)
        HSIbase = zoombase(HSIbase, flow, self.local_rank, sr_ratio)
        HSIbase = self.SharpGray(HSIbase.view(batch*num_spec, 1, sr_ratio*h, sr_ratio*w)).view(batch, num_spec, sr_ratio*h, sr_ratio*w)
        #HSIbase = torch.cat((HSIbase,HSIbase[:,6:9]), dim=1)

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
            confidence_features = confidence_features.view(batch, 9, self.num_features1, sr_ratio * h, sr_ratio * w) * features_sr[:, indice]
            confidence_features = confidence_features.sum(dim=1)
            weight_fuse.append(confidence_features)

        weight_fuse = torch.stack(weight_fuse, dim=1) # batch,num_spec, self.num_features1, h, w
        #weight_fuse = torch.cat((weight_fuse,weight_fuse[:,6:9]), dim=1)
        weight_fuse = weight_fuse.permute(0, 2, 1, 3, 4).contiguous()
        weight_fuse = self.FineFuse(weight_fuse) # b, self.num_features/2, num_spec, 2h, 2w
        weight_fuse = weight_fuse.squeeze(1) # b, 1, num_spec, 8h, 8w to b, num_spec, 8h, 8w
        HRHSI = weight_fuse + HSIbase
        return weight_fuse, HRHSI


def test(local_rank, world_size, args):
    train_bs, val_bs, num_features, ksize, gray_max, sr_ratio = args.train_bs, args.val_bs, args.num_features, args.kernel_size, args.gray_max, args.sr_ratio

    ################## load Models
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    cudnn.benchmark = True
    args.local_rank = local_rank
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='tcp://127.0.0.1:1456',
                                         world_size=args.world_size,
                                         rank=local_rank)
    torch.cuda.set_device(local_rank)
    train_bs = int(train_bs / args.world_size)
    val_bs = int(val_bs / args.world_size)
    SSLHSISR = HSISR(local_rank=local_rank, num_features1=num_features, num_features2=int(num_features/2), ksize=ksize).cuda(local_rank)
    torch.distributed.barrier()

    if args.Sync_BN:
        # time-consuming using SyncBatchNorm
        SSLHSISR = torch.nn.SyncBatchNorm.convert_sync_batchnorm(SSLHSISR).cuda(local_rank)
    SSLHSISR = torch.nn.parallel.DistributedDataParallel(SSLHSISR, device_ids=[local_rank], broadcast_buffers=False)
    checkpoint_path_SR = args.model_path
    checkpoint = torch.load(checkpoint_path_SR, map_location=torch.device('cuda', local_rank))
    SSLHSISR.load_state_dict(checkpoint['state_dict DeepSR'])

    Dataset_path = args.test_file
    val_data_set = BurstSRDataset(root=Dataset_path, split='test', burst_size=9, crop_sz=args.test_size,
                                  center_crop=True, random_flip=False)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data_set)
    nw = min([os.cpu_count(), val_bs if val_bs > 1 else 0, 8])
    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=val_bs,
                                             sampler=val_sampler,
                                             pin_memory=True,
                                             num_workers=nw)

    result_dir_SR = args.save_SR
    result_dir_LR = args.save_LR

    boundary = 4
    starttime = time()
    ##################
    SSLHSISR.eval()
    with torch.no_grad():
        for k, data in enumerate(val_loader):
            samplesLR_blur, numbers = data
            samplesLR_blur = samplesLR_blur.float().cuda(local_rank)
            samplesLR_blur = samplesLR_blur / args.gray_max
            batch, num_spec, h, w = samplesLR_blur.shape
            indices = []
            for i in range(num_spec):
                indice, remain_indice = create_groups(num_spec, i, True)
                indices.append(indice)

            HRHSIdetail, HRHSI = SSLHSISR(samplesLR_blur, boundary, indices,sr_ratio=sr_ratio)
            HRHSI = HRHSI.clamp(0.0, 1.0)

            if local_rank == 0:
                safe_mkdir(os.path.join(result_dir_SR, numbers[0]))
                safe_mkdir(os.path.join(result_dir_LR, numbers[0]))
            if local_rank == 0:
                HRHSI = HRHSI[0, ...]
                samplesLR_blur = samplesLR_blur[0, ...]
                num_spec, h, w = samplesLR_blur.shape
                for n in range(num_spec):
                    HR = HRHSI[n, ...]
                    HR = HR.detach().cpu().numpy().squeeze()
                    HR = (HR * 255).astype('uint8')
                    imwrite(os.path.join(result_dir_SR, numbers[0], "HR_{:02d}.png".format(n)), HR)
                    LR = samplesLR_blur[n, ...]
                    LR = LR.detach().cpu().numpy().squeeze()
                    LR = (LR * 255).astype('uint8')
                    imwrite(os.path.join(result_dir_LR, numbers[0], "LR_{:02d}.png".format(n)), LR)

    torch.cuda.synchronize(torch.device('cuda', local_rank))
    if local_rank == 0:
        print('Execution time = {:.2f}s'.format(time() - starttime))

    torch.distributed.destroy_process_group()
    return


def main(args):
    """
        Given a configuration, trains Encoder, Decoder and fnet for Multi-Frame Super Resolution (MFSR), and saves best model.
        Args:
            config: dict, configuration file
    """
    torch.cuda.empty_cache()
    mp.spawn(test, nprocs=args.world_size, args=(args.world_size, args))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-bsv", "--val_bs", help="Batch size of val loader", type=int, default=1)
    parser.add_argument("-nf", "--num_features", help="Num of features for encoder", type=int, default=32)
    parser.add_argument("-ls", "--lr_DeepSR", help="Learning rate of DeepSR", type=float, default=1e-4)
    parser.add_argument("-ks", "--kernel_size", help="Size of blur kernel", type=int, default=13)
    parser.add_argument('-ts', '--test_size', help="the size of crop for test", default=448)
    parser.add_argument("-gm", "--gray_max", help="the max of gray", type=float, default=65535.0)
    parser.add_argument("-ssr", "--save_SR", help="save SR image", type=str, default='/SR_file')
    parser.add_argument("-slr", "--save_LR", help="save LR image", type=str, default='/LR_file')
    parser.add_argument("-cm", "--model_path", help="checkpoint of SR model", type=str, default='/checkpoint_SR.tar')
    parser.add_argument("-tf", "--test_file", help="file of test data", type=str, default='/test_file')
    parser.add_argument('-wz', '--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('-sbn', '--Sync_BN', help="Synchronization of BN across multi-gpus", default=True)
    parser.add_argument("-srr", "--sr_ratio", help="Super-resolution factor", type=int, default=4)
    args = parser.parse_args()
    main(args)



