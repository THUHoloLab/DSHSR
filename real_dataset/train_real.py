""" Python script to train option J """
import datetime
import random
import numpy as np
# import matplotlib.pyplot as plt
import os
import argparse
import tempfile
from tqdm import tqdm
from time import time

from numpy import mean
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F

from MEnet import FNet
from DAST import EncoderNet, DecoderNet
from shiftandadd import shiftAndAdd, featureAdd, featureWeight
from HRPAF import PCAFNet, AGFNet
from warpingOperator import WarpedLoss, TVL1, base_detail_decomp, BlurLayer
import os
from torch.autograd import Variable
from synthetic_dataset.zurich_raw2rgb_dataset import ZurichRAW2RGB
from synthetic_dataset.synthetic_burst_train_set import SyntheticBurst
from pwcnet.pwcnet import PWCNet

from torchvision.transforms import GaussianBlur
import torch.multiprocessing as mp
# from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import Process
import cv2
from seed_everything import seed_everything
from pytorch_wavelets import DWTForward
from real_dataset.burstsr_dataset import BurstSRDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "3,4,5"


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
    args.dist_backend = 'nccl'  # 通信后端，nvidia GPU推荐使用NCCL
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()


def safe_mkdir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.makedirs(path)
    except OSError:
        pass

def flowEstimation(samplesLR, local_rank, ME, warping, gaussian_filter):
    """
    Compute the optical flows from the other frames to the reference:
    samplesLR: Tensor b, num_im, h, w
    ME: Motion Estimator
    """

    b, num_im, h, w = samplesLR.shape

    samplesLRblur = gaussian_filter(samplesLR)


    samplesLR_0 = samplesLRblur[:,:1,...] #b, 1, h, w


    samplesLR_0 = samplesLR_0.repeat(1, num_im, 1,1)  #b, num_im, h, w
    samplesLR_0 = samplesLR_0.reshape(-1, h, w)
    samplesLRblur = samplesLRblur.reshape(-1, h, w)  #b*num_im, h, w
    concat = torch.cat((samplesLRblur.unsqueeze(1), samplesLR_0.unsqueeze(1)), dim=1) #b*(num_im), 2, h, w
    flow = ME(concat.cuda(local_rank)) #b*(num_im), 2, h, w

    flow[::num_im] = 0

    warploss, _ = warping(samplesLRblur.unsqueeze(1),samplesLR_0.unsqueeze(1), flow, losstype = 'Detail')

    return flow.reshape(b, num_im, 2, h, w), warploss


class DeepSR(nn.Module):
    def __init__(self, local_rank, num_im=9, Encoder_num_features=64, PCAF_num_features=64, sr_ratio=3, img_size=192):
        super(DeepSR, self).__init__()
        self.encoder = EncoderNet(inp_channels=1,
                                  out_channels=Encoder_num_features,
                                  num_blocks=[1, 2, 2],
                                  heads=[1, 2, 2],
                                  ffn_expansion_factor=1.5,  # 门控模块里的膨胀系数，为了增加通道数量
                                  bias=False,
                                  LayerNorm_type='WithBias')
        self.PCAF = PCAFNet(num_im, PCAF_num_features)
        self.AGF = AGFNet(PCAF_num_features, 4)
        self.decoder = DecoderNet(inp_channels=PCAF_num_features,
                                  out_channels=1,
                                  num_blocks=[2, 2, 2],
                                  heads=[2, 2, 2],
                                  img_size=img_size,
                                  bias=False)
        # self.alignment_net = PWCNet(load_pretrained=True, weights_path=checkpoint_path)
        self.alignment_net = FNet().float()
        self.sr_ratio = sr_ratio
        self.local_rank = local_rank
        self.Encoder_num_features = Encoder_num_features

    def forward(self, samplesLR):

        # checkpoint = torch.load(self.checkpoint_path)  # 将光流网络预训练权重载入
        # FNet.load_state_dict(checkpoint['state_dictFnet'])  # Load the pretrained Fnet
        # pwcnet = PWCNet(load_pretrained=True,weights_path='{}/pwcnet-network-default.pth'.format(checkpoint_path))
        flow, warploss = flowEstimation(samplesLR, self.local_rank, self.alignment_net, WarpedLoss(interpolation = 'bicubicTorch'), gaussian_filter=GaussianBlur(11, sigma=1))  # 待补充
        b, num_im, h, w = samplesLR.shape
        random_shifts = torch.randint(low=0, high=self.sr_ratio, size=(b, 1, 2, 1, 1)) / self.sr_ratio  # Grid shifting
        flow = flow - random_shifts.cuda(self.local_rank)

        samplesLR = samplesLR.view(-1, 1, h, w)  # b*(num_im),1, h, w
        features = self.encoder(samplesLR).view(-1, h, w)  # b * (num_im), num_features, h, w
        # b * num_im *num_features, h, w
        dacc = featureWeight(flow.view(-1, 2, h, w), sr_ratio=self.sr_ratio,
                             local_rank=self.local_rank)  # b * num_im, 2*h, 2*w
        flowf = flow.contiguous().view(-1, 1, 2, h, w).repeat(1, self.Encoder_num_features, 1, 1, 1).view(-1, 2, h,
                                                                                                          w)  # b * num_im* num_features, 2, h, w
        dadd = featureAdd(features, flowf, sr_ratio=self.sr_ratio,
                          local_rank=self.local_rank)  # b * num_im * num_features, 2h, 2w
        dadd = dadd.view(b, num_im, self.Encoder_num_features, self.sr_ratio * h, self.sr_ratio * w)
        dacc = dacc.view(b, num_im, 1, self.sr_ratio * h, self.sr_ratio * w).repeat(1, 1, self.Encoder_num_features, 1, 1)
        dadd = torch.div(dadd, dacc)  # 待确定
        # dadd = torch.where(torch.isnan(dadd), torch.full_like(dadd, 0), dadd)
        '''if self.local_rank == 1:
            for i in range(num_im):
                cv2.imwrite('{}/{}.png'.format("/home/WXS/CYT/SPMC/", str(i)),
                            dadd[1, i, 10, ...].cpu().detach().numpy().astype(np.uint8))'''

        ''' SR = self.PCAF(dadd)
        SR = self.AGF(SR)  # b, num_features, sr_ration*h, sr_ratio*w
        SR = self.decoder(SR)  # b, 1, sr_ration*h, sr_ratio*w
        SR = torch.empty(b, self.Encoder_num_features, self.sr_ratio * h, self.sr_ratio * w)
        # print(next(self.PCAF.parameters()).device)
        for i in range(0, b):
            cc_feat = dadd[i]
            cc_feat = self.PCAF(cc_feat)
            print(cc_feat.device)
            cc_feat = self.AGF(cc_feat)  # 1, num_features, sr_ration*h, sr_ratio*w
            cc_feat = cc_feat.squeeze(0)  # num_features, sr_ration*h, sr_ratio*w
            SR[i] = cc_feat'''
        SR = self.AGF(self.PCAF(dadd)).squeeze(1)  # AGF后尺寸为 b, 1, PCAF_num_features, self.sr_ratio * h, self.sr_ratio * w
        SR = self.decoder(SR)  # b, 1, sr_ration*h, sr_ratio*w

        return SR, flow, warploss


def BicubicWarping(x, local_rank, flo, ds_factor=3):
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


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= nprocs
    return rt


def train(local_rank, world_size, args):
    seed_everything(0)  # Fix the seed for reproducibility
    criterion = nn.L1Loss()
    TVLoss = TVL1(TVLoss_weight=1)
    # Load the parameters
    train_bs, val_bs, lr_DeepSR, factor_DeepSR, patience_DeepSR, num_epochs, num_im, sr_ratio, train_patchsize, val_patchsize, warp_weight, TVflow_weight, high_weight = \
        args.train_bs, args.val_bs, args.lr_DeepSR, args.factor_DeepSR, args.patience_DeepSR, args.num_epochs, args.num_im, args.sr_ratio, args.train_patchsize, args.val_patchsize, args.warp_weight, args.TVflow_weight, args.high_weight
    Ed_num_features, PCAF_num_features, num_blocks = args.Ed_num_features, args.PCAF_num_features, args.num_blocks

    sr_ratio = args.sr_ratio
    folder_name = 'self-supervised_multi-image_deepSR_time_{}'.format(
        f"{datetime.datetime.now():%m-%d-%H-%M-%S}")

    ################## load Models
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    # 初始化各进程环境 start
    cudnn.benchmark = True
    args.local_rank = local_rank
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='tcp://127.0.0.1:9456',
                                         world_size=args.world_size,
                                         rank=local_rank)
    torch.cuda.set_device(local_rank)
    # lr_DeepSR *= args.world_size  # 学习率要根据并行GPU的数量进行倍增
    train_bs = int(train_bs / args.world_size)
    val_bs = int(val_bs / args.world_size)
    # 实例化模型
    deepSuperresolve = DeepSR(local_rank=local_rank, num_im=num_im,
                              Encoder_num_features=Ed_num_features, PCAF_num_features=PCAF_num_features, sr_ratio=sr_ratio, img_size=train_patchsize).cuda(local_rank)

    checkpoint_path_SR = "/home/WXS/CYT/TrainHistory/self-supervised_multi-image_deepSR_time_12-22-15-21-49/checkpoint_100.pth.tar"
    checkpoint = torch.load(checkpoint_path_SR, map_location=torch.device('cuda', local_rank))
    deepSuperresolve.load_state_dict(checkpoint['state_dict DeepSR'])
    optimizer_DeepSR = torch.optim.Adam(deepSuperresolve.parameters(), lr=lr_DeepSR)
    #optimizer_DeepSR = torch.optim.AdamW(deepSuperresolve.parameters(), lr=lr_DeepSR, weight_decay=args.weight_decay)
    schedulerDeepSR = torch.optim.lr_scheduler.StepLR(optimizer_DeepSR, step_size=patience_DeepSR, gamma=factor_DeepSR)

    if args.Sync_BN:
        # 使用SyncBatchNorm后训练会更耗时
        deepSuperresolve = torch.nn.SyncBatchNorm.convert_sync_batchnorm(deepSuperresolve).cuda(local_rank)

    deepSuperresolve = torch.nn.parallel.DistributedDataParallel(deepSuperresolve, device_ids=[local_rank],
                                                                 broadcast_buffers=False)  # find_unused_parameters=True
    '''checkpoint_path_SR = '/home/WXS/CYT/TrainHistory/self-supervised_multi-image_deepSR_time_09-30-16-03-56/checkpoint_60.pth.tar'
    checkpoint = torch.load(checkpoint_path_SR, map_location=torch.device('cuda', local_rank))
    deepSuperresolve.load_state_dict(checkpoint['state_dict DeepSR'])
    optimizer_DeepSR.load_state_dict(checkpoint['optimizerDeepSR'])'''

    ################## load datas
    # print(next(deepSuperresolve.parameters()).device)
    # print(torch.device('cuda', local_rank))

    Dataset_path = '/home/WXS/CYT/'  # Make sure to preprocess the downloaded data first
    train_zurich_raw2rgb = ZurichRAW2RGB(root=Dataset_path, split='train')
    train_data_set = BurstSRDataset(root=Dataset_path, split='train', burst_size=num_im, crop_sz=train_patchsize, random_flip=True)
    # a = train_data_set[1000]
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)  # 给每个rank对应的进程分配训练的样本索引
    nw = min([os.cpu_count(), train_bs if train_bs >= 1 else 0, 8])  # number of workers
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=train_bs,
                                               sampler=train_sampler,
                                               pin_memory=True,
                                               num_workers=nw)
    val_data_set = BurstSRDataset(root=Dataset_path, split='val', burst_size=num_im, crop_sz=train_patchsize, random_flip=False)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data_set)
    nw = min([os.cpu_count(), val_bs if val_bs >= 1 else 0, 8])
    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=val_bs,
                                             sampler=val_sampler,
                                             pin_memory=True,
                                             num_workers=nw)

    checkpoint_dir = '/home/WXS/CYT/TrainHistory/{}'.format(folder_name)
    #logs_dir = '/home/WXS/CYT/TrainLogs/{}'.format(folder_name)
    result_dir_SR = '/home/WXS/CYT/result_dir_SR/{}'.format(folder_name)
    result_dir_high_low = '/home/WXS/CYT/result_dir_high_low/{}'.format(folder_name)
    if local_rank == 0:
        safe_mkdir(checkpoint_dir)
        #safe_mkdir(logs_dir)
        safe_mkdir(result_dir_SR)
        safe_mkdir(result_dir_high_low)
    # writer = SummaryWriter(logs_dir)
    # print(next(deepSuperresolve.parameters()).device)
    #gaussian_filter = GaussianBlur(11, sigma=1).cuda(local_rank)
    DWT2 = DWTForward(J=1, wave='haar', mode='reflect').cuda(local_rank)
    ##################
    starttime = time()
    ##################
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)  # 打乱样本
        TrainLoss = []
        ValLoss = []

        if local_rank == 0:  # 在第一进程中执行打印
            print('__________________________________________________')
            print('Training epoch {0:3d}'.format(epoch))

        for i, data in enumerate(train_loader):
            """
            samplesLR : b, num_im, h, w
            flow: b, num_im, 2, h, w
            SR:  b, 1, 3*h, 3*w

            """
            samplesLR, gt = data
            optimizer_DeepSR.zero_grad()

            b, num_im, h, w = samplesLR.shape
            samplesLR = samplesLR.float().cuda(local_rank)
            # gt = gt.float().cuda(local_rank)
            # print(samplesLR.device)
            # with torch.autograd.set_detect_anomaly(True):
            SR, flow, warploss = deepSuperresolve(samplesLR)
            c = 5  # 取决于图像间的最大偏移量，在计算损失函数时去除无重叠区域
            tvloss = TVLoss(flow[..., c:-c, c:-c])  # 对齐模块中TV损失

            SR_ds = BicubicWarping(SR.unsqueeze(1).repeat(1, num_im, 1, 1, 1).view(-1, 1, sr_ratio * h, sr_ratio * w), local_rank,
                                   flow.view(-1, 2, h, w),
                                   sr_ratio)  # To align the SR with the reference frame before downsampling 改成所有图像序列
            samplesLR = samplesLR.view(b * num_im, 1, h, w)
            samplesLR_LL, samplesLR_Hc = DWT2(samplesLR[..., c:-c, c:-c])
            samplesLR_LH, samplesLR_HL, samplesLR_HH = samplesLR_Hc[0][:, :, 0, :, :], samplesLR_Hc[0][:, :, 1, :, :], samplesLR_Hc[0][:, :, 2, :, :]
            '''cv2.imwrite("/home/WXS/CYT/临时/小波高频/LH.png", samplesLR_LH[0, ...].detach().cpu().numpy().squeeze())
            cv2.imwrite("/home/WXS/CYT/临时/小波高频/HL.png", samplesLR_HL[0, ...].detach().cpu().numpy().squeeze())
            cv2.imwrite("/home/WXS/CYT/临时/小波高频/HH.png", samplesLR_HH[0, ...].detach().cpu().numpy().squeeze())'''
            SR_ds_LL, SR_ds_Hc = DWT2(SR_ds[..., c:-c, c:-c])
            SR_ds_LH, SR_ds_HL, SR_ds_HH = SR_ds_Hc[0][:, :, 0, :, :], SR_ds_Hc[0][:, :, 1, :, :], SR_ds_Hc[0][:, :, 2, :, :]
            samplesLR_high = torch.cat((samplesLR_LH, samplesLR_HL, samplesLR_HH), 1)
            SR_ds_high = torch.cat((SR_ds_LH, SR_ds_HL, SR_ds_HH), 1)

            '''if local_rank == 0:
                            cv2.imwrite(os.path.join(result_dir_high_low, "SR_ds_low.png"),
                                        SR_ds_LL[0, 0, ...].detach().cpu().numpy().squeeze())
                            cv2.imwrite(os.path.join(result_dir_high_low, "SR_ds_high0.png"),
                                        SR_ds_high[0, 0, ...].detach().cpu().numpy().squeeze())
                            cv2.imwrite(os.path.join(result_dir_high_low, "SR_ds_high1.png"),
                                        SR_ds_high[0, 1, ...].detach().cpu().numpy().squeeze())
                            cv2.imwrite(os.path.join(result_dir_high_low, "SR_ds_high2.png"),
                                        SR_ds_high[0, 2, ...].detach().cpu().numpy().squeeze())
                            cv2.imwrite(os.path.join(result_dir_high_low, "samplesLR_low.png"),
                                        samplesLR_LL[0, 0, ...].detach().cpu().numpy().squeeze())
                            cv2.imwrite(os.path.join(result_dir_high_low, "samplesLR_high0.png"),
                                        samplesLR_high[0, 0, ...].detach().cpu().numpy().squeeze())
                            cv2.imwrite(os.path.join(result_dir_high_low, "samplesLR_high1.png"),
                                        samplesLR_high[0, 1, ...].detach().cpu().numpy().squeeze())
                            cv2.imwrite(os.path.join(result_dir_high_low, "samplesLR_high2.png"),
                                        samplesLR_high[0, 2, ...].detach().cpu().numpy().squeeze())'''

            N2Nloss = criterion(samplesLR[::num_im, ...], SR_ds[::num_im, ...])

            highloss = criterion(samplesLR_high, SR_ds_high)
            trainloss = N2Nloss + high_weight*highloss + warp_weight*warploss+ TVflow_weight* tvloss

            trainloss.backward()  # 每個GPU的輸入數據損失函數的反向傳播的梯度均值，因此其梯度大小是基於train_bs個樣本計算

            optimizer_DeepSR.step()

            reduce_trainloss = reduce_mean(trainloss,
                                           args.world_size)  # 這裡的損失函數是train_bs*args.world_size個樣本的args.world_size均值，是為了展示
            TrainLoss.append(reduce_trainloss.data.item())
        # 等待所有进程计算完毕
        torch.cuda.synchronize(torch.device('cuda', local_rank))

        if epoch < 300:
            if local_rank == 0:
                print('Train')
                print('{:.5f}'.format(mean(TrainLoss)))

        deepSuperresolve.eval()
        with torch.no_grad():
            for k, data in enumerate(val_loader):
                samplesLR, gt = data
                b, num_im, h, w = samplesLR.shape
                samplesLR = samplesLR.float().cuda(local_rank)
                # gt = gt.float().cuda(local_rank)
                SR, flow, warploss = deepSuperresolve(samplesLR)
                c = 5  # 取决于图像间的最大偏移量，在计算损失函数时去除无重叠区域
                tvloss = TVLoss(flow[..., c:-c, c:-c])  # 对齐模块中TV损失

                SR_ds = BicubicWarping(SR.unsqueeze(1).repeat(1, num_im, 1, 1, 1).view(-1, 1, sr_ratio * h, sr_ratio * w), local_rank,
                                       flow.view(-1, 2, h, w),
                                       sr_ratio)  # To align the SR with the reference frame before downsampling 改成所有图像序列
                samplesLR = samplesLR.view(b * num_im, 1, h, w)
                samplesLR_LL, samplesLR_Hc = DWT2(samplesLR[..., c:-c, c:-c])
                samplesLR_LH, samplesLR_HL, samplesLR_HH = samplesLR_Hc[0][:, :, 0, :, :], samplesLR_Hc[0][:, :, 1, :,:], samplesLR_Hc[0][:, :, 2,:, :]
                SR_ds_LL, SR_ds_Hc = DWT2(SR_ds[..., c:-c, c:-c])
                SR_ds_LH, SR_ds_HL, SR_ds_HH = SR_ds_Hc[0][:, :, 0, :, :], SR_ds_Hc[0][:, :, 1, :, :], SR_ds_Hc[0][:, :,2, :, :]
                samplesLR_high = torch.cat((samplesLR_LH, samplesLR_HL, samplesLR_HH), 1)
                SR_ds_high = torch.cat((SR_ds_LH, SR_ds_HL, SR_ds_HH), 1)

                N2Nloss = criterion(samplesLR[::num_im, ...], SR_ds[::num_im, ...])
                highloss = criterion(samplesLR_high, SR_ds_high)
                valloss = N2Nloss + high_weight*highloss + warp_weight*warploss+ TVflow_weight* tvloss

                reduce_valloss = reduce_mean(valloss, args.world_size)
                ValLoss.append(reduce_valloss.data.item())

            torch.cuda.synchronize(torch.device('cuda', local_rank))
            if local_rank == 0:
                SR = SR[0, 0, ...] #[]才是索引
                SR = SR.detach().cpu().numpy().squeeze() #.astype('int8')
                cv2.imwrite(os.path.join(result_dir_SR, "SR_{:03d}.png".format(epoch)), SR)
                '''samplesLR = samplesLR[0, 0, ...]
                samplesLR = samplesLR.detach().cpu().numpy().squeeze()
                cv2.imwrite(os.path.join(result_dir_LR, "LR_{:03d}.png".format(epoch)), samplesLR)'''
                #np.save(os.path.join(result_dir_SR, "SR_{:03d}.png".format(epoch)), SR)

        if epoch < 300:
            if local_rank == 0:
                print('Val')
                print('{:.5f}'.format(mean(ValLoss)))

        # schedulerFnet.step()
        schedulerDeepSR.step()

        if epoch % 10 == 0:
            if local_rank == 0:
                print('#### Saving Models ... ####')
                print('#### Saving Models ... ####')
                state = {'epoch': epoch + 1, 'state_dict DeepSR': deepSuperresolve.state_dict(),
                         'optimizerDeepSR': optimizer_DeepSR.state_dict()}
                torch.save(state, os.path.join(checkpoint_dir, 'checkpoint_{}.pth.tar'.format(epoch)))
                '''writer.add_image('SR Image', SR[0, 0, 0, ...], epoch, dataformats='HW')
                writer.add_scalar("train_loss", mean(TrainLoss), epoch)
                writer.add_scalar("val_loss", mean(ValLoss), epoch)'''
    if local_rank == 0:
        print('Execution time = {:.0f}s'.format(time() - starttime))
    if local_rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    torch.distributed.destroy_process_group()
    # writer.close()
    return


def zoombase(LR_base, flow, device, warping):
    b, num_im, h, w = LR_base.shape

    LR_base = LR_base.view(-1, 1, h, w)
    LR_base = warping.warp(LR_base, -flow.view(-1, 2, h, w))
    LR_base = LR_base.view(b, num_im, h, w)
    LR_base = torch.mean(LR_base, 1, keepdim=True)

    SR_base = torch.nn.functional.interpolate(LR_base, size=[2 * h - 1, 2 * w - 1], mode='bilinear', align_corners=True)
    SR_base = torch.cat((SR_base, torch.zeros(b, 1, 1, 2 * w - 1).to(device)), dim=2)
    SR_base = torch.cat((SR_base, torch.zeros(b, 1, 2 * h, 1).to(device)), dim=3)
    return SR_base


def zoombase_weighted(LR_base, expotime, flow, device, warping):
    b, num_im, h, w = LR_base.shape

    LR_base = LR_base.view(-1, 1, h, w)
    LR_base = warping.warp(LR_base, -flow.view(-1, 2, h, w))
    LR_base = LR_base.view(b, num_im, h, w)
    LR_base = torch.mean(LR_base * expotime, 1, keepdim=True) / torch.mean(expotime, 1, keepdim=True)

    SR_base = torch.nn.functional.interpolate(LR_base, size=[2 * h - 1, 2 * w - 1], mode='bilinear', align_corners=True)
    SR_base = torch.cat((SR_base, torch.zeros(b, 1, 1, 2 * w - 1).to(device)), dim=2)
    SR_base = torch.cat((SR_base, torch.zeros(b, 1, 2 * h, 1).to(device)), dim=3)
    return SR_base


def check(args):
    feature_mode = args.feature_mode
    print(feature_mode)
    print(len(feature_mode))


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
    parser.add_argument("-on", "--option_name", help="Option id", default='J_selfSR')
    parser.add_argument("-bst", "--train_bs", help="Batch size of train loader", type=int,
                        default=6)  # 每个进程的batch size,n个gup就有4*n个样本 15
    parser.add_argument("-bsv", "--val_bs", help="Batch size of val loader", type=int, default=6) #5
    # parser.add_argument("-lrf", "--lr_fnet", help="Learning rate of fnet", type=float, default=1e-5)
    # parser.add_argument("-lre", "--lr_encoder", help="Learning rate of Encoder", type=float, default=1e-4)
    ##parser.add_argument("-lrd", "--lr_decoder", help="Learning rate of Decoder", type=float, default=1e-4)
    # parser.add_argument("-lrp", "--lr_PCAF", help="Learning rate of PCAF", type=float, default=1e-4)  # 待确定
    parser.add_argument("-lr", "--lr_DeepSR", help="Learning rate of DeepSR", type=float, default=1e-4)  # 待确定
    # parser.add_argument("-ff", "--factor_fnet", help="Learning rate decay factor of fnet", type=float, default=0.3)
    # parser.add_argument("-fe", "--factor_encoder", help="Learning rate decay factor of Encoder", type=float,default = 0.3)
    # parser.add_argument("-fd", "--factor_decoder", help="Learning rate decay factor of Decoder", type=float,default = 0.3)
    # parser.add_argument("-fp", "--factor_PCAF", help="Learning rate decay factor of PBFF", type=float,default = 0.3)  # 待确定
    parser.add_argument("-fa", "--factor_DeepSR", help="Learning rate decay factor of DeepSR", type=float,
                        default=0.5)  # 待确定
    # parser.add_argument("-pf", "--patience_fnet", help="Step size for learning rate of fnet", type=int, default=300)
    # parser.add_argument("-pe", "--patience_encoder", help="Step size for learning rate of Encoder", type=int,default=400)
    # parser.add_argument("-pd", "--patience_decoder", help="Step size for learning rate of Decoder", type=int,default=400)
    # parser.add_argument("-pp", "--patience_pbff", help="Step size for learning rate of PBFF", type=int,default=400)  # 待确定
    parser.add_argument("-pa", "--patience_DeepSR", help="Step size for learning rate of DeepSR", type=int,
                        default=50)  # 待确定
    parser.add_argument("-ne", "--num_epochs", help="Num_epochs", type=int, default=200)
    parser.add_argument("-ednf", "--Ed_num_features", help="Num of features for encoder", type=int, default=64)
    parser.add_argument("-pbffnf", "--PBFF_num_features", help="Num of features for PBFF", type=int, default=64)
    parser.add_argument("-nb", "--num_blocks", help="Number of residual blocks in encoder", type=int, default=4)
    # parser.add_argument("-ww", "--warp_weight", help="Weight for the warping loss", type=float, default=3)
    # parser.add_argument("-tvw", "--TVflow_weight", help="Weight for the TV flow loss", type=float, default=0.01)
    # parser.add_argument("-s", "--sigma", help="Std for SR filtering", type=float, default=1)
    parser.add_argument("-srr", "--sr_ratio", help="Super-resolution factor", type=int, default=3)
    parser.add_argument('-num', '--num_im', nargs='+', help="Number of image for camera array", default=9)
    parser.add_argument('-sbn', '--Sync_BN', help="Synchronization of BN across multi-gpus", default=True)
    parser.add_argument('-tps', '--train_patchsize', help="the size of crop for training", default=168)
    parser.add_argument('-vps', '--val_patchsize', help="the size of crop for val", default=168)
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('-wz', '--world_size', default=3, type=int, help='number of distributed processes')
    parser.add_argument("-ww", "--warp_weight", help="Weight for the warping loss", type=float, default=3)
    parser.add_argument("-tvw", "--TVflow_weight", help="Weight for the TV flow loss", type=float, default=0.01)
    parser.add_argument("-hw", "--high_weight", help="Weight for the high-frequency loss", type=float, default=1.0)
    # parser.add_argument('-wd', '--weight-decay', help='weight decay (default: 1e-6)', type=float, default=1e-6)

    args = parser.parse_args()

    main(args)