import torch.nn as nn
import math
import torch.nn.functional as F
from torchvision.models.vgg import vgg19
import utils.util as util
import torch


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:6]).eval() #35
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()

    def forward(self, high_resolution, fake_high_resolution):
        perception_loss = self.l1_loss(self.loss_network(high_resolution), self.loss_network(fake_high_resolution))
        return perception_loss

class Pyramid(nn.Module):
    """Pyramid Loss"""
    def __init__(self, num_levels=3, pyr_mode='gau'):
        super(Pyramid, self).__init__()
        self.num_levels = num_levels
        self.pyr_mode = pyr_mode
        assert self.pyr_mode == 'gau' or self.pyr_mode == 'lap'

    def forward(self, x, local_rank):
        B, C, H, W = x.shape
        gauss_kernel = util.gauss_kernel(size=5, device=local_rank, channels=C)
        if self.pyr_mode == 'gau':
            pyr_x = util.gau_pyramid(img=x, kernel=gauss_kernel, max_levels=self.num_levels)
        else:
            pyr_x = util.lap_pyramid(img=x, kernel=gauss_kernel, max_levels=self.num_levels)
        return pyr_x

def SmoothFlowLoss(images, flows,
    smoothness_edge_weighting, smoothness_edge_constant, weights):
    losses = {}
    for key in weights:
        losses[key] = torch.Tensor([0.0])
    if 'smooth1' in weights:
        smooth_loss_1st = first_order_smoothness_loss(
            image=images,
            flow=flows,
            smoothness_edge_weighting=smoothness_edge_weighting, smoothness_edge_constant=smoothness_edge_constant)
        losses['smooth1'] = weights['smooth1'] * torch.sum(smooth_loss_1st.squeeze(), dim=0, keepdim=True)/ images.shape[0] # 配对的图像对
    if 'smooth2' in weights:
        # Compute second-order smoohtness term loss.
        smooth_loss_2nd = second_order_smoothness_loss(
            image=images,
            flow=flows,
            smoothness_edge_weighting=smoothness_edge_weighting, smoothness_edge_constant=smoothness_edge_constant)
        losses['smooth2'] = weights['smooth2'] * torch.sum(smooth_loss_2nd.squeeze(), dim=0, keepdim=True) / images.shape[0]

    return losses



def first_order_smoothness_loss(
    image, flow,
    smoothness_edge_weighting, smoothness_edge_constant):
  """Computes a first-order smoothness loss.

  Args:
    image: Image used for the edge-aware weighting [batch, height, width, 2].
    flow: Flow field for with to compute the smoothness loss [batch, height,
      width, 2].
    edge_weighting_fn: Function used for the edge-aware weighting.

  Returns:
    Average first-order smoothness loss.
  """
  img_gx, img_gy = image_grads(image)
  weights_x = edge_weighting_fn(img_gx, smoothness_edge_weighting, smoothness_edge_constant)
  weights_y = edge_weighting_fn(img_gy, smoothness_edge_weighting, smoothness_edge_constant)

  # Compute second derivatives of the predicted smoothness.
  flow_gx, flow_gy = image_grads(flow)
  flow_gx = torch.mean(torch.mean(torch.mean(weights_x * robust_l1(flow_gx), dim=-1, keepdim=True), dim=-2, keepdim=True), dim=-3, keepdim=True)
  flow_gy = torch.mean(torch.mean(torch.mean(weights_y * robust_l1(flow_gy), dim=-1, keepdim=True), dim=-2, keepdim=True), dim=-3, keepdim=True)

  # Compute weighted smoothness
  return (flow_gx+ flow_gy) / 2.


def second_order_smoothness_loss(
    image, flow,
    smoothness_edge_weighting, smoothness_edge_constant):
  """Computes a second-order smoothness loss.

  Computes a second-order smoothness loss (only considering the non-mixed
  partial derivatives).

  Args:
    image: Image used for the edge-aware weighting [batch, height, width, 2].
    flow: Flow field for with to compute the smoothness loss [batch, height,
      width, 2].
    edge_weighting_fn: Function used for the edge-aware weighting.

  Returns:
    Average second-order smoothness loss.
  """
  img_gx, img_gy = image_grads(image, stride=2)
  weights_xx = edge_weighting_fn(img_gx, smoothness_edge_weighting, smoothness_edge_constant)
  weights_yy = edge_weighting_fn(img_gy, smoothness_edge_weighting, smoothness_edge_constant)

  # Compute second derivatives of the predicted smoothness.
  flow_gx, flow_gy = image_grads(flow)
  flow_gxx, _ = image_grads(flow_gx)
  _, flow_gyy = image_grads(flow_gy)

  flow_gxx = torch.mean(torch.mean(torch.mean(weights_xx * robust_l1(flow_gxx), dim=-1, keepdim=True), dim=-2, keepdim=True), dim=-2, keepdim=True)
  flow_gyy = torch.mean(torch.mean(torch.mean(weights_yy * robust_l1(flow_gyy), dim=-1, keepdim=True), dim=-2, keepdim=True), dim=-2, keepdim=True)

  # Compute weighted smoothness
  return (flow_gxx + flow_gyy) / 2.


def image_grads(image_batch, stride=1):
  image_batch_gh = image_batch[..., stride:, :] - image_batch[..., :-stride, :]
  image_batch_gw = image_batch[..., stride:] - image_batch[..., :-stride]
  return image_batch_gw, image_batch_gh


def edge_weighting_fn(x, smoothness_edge_weighting, smoothness_edge_constant):
    if smoothness_edge_weighting == 'gaussian':
        return torch.exp(-torch.mean((smoothness_edge_constant * x) ** 2, dim=-3, keepdim=True))
    elif smoothness_edge_weighting == 'exponential':
        return torch.exp(-torch.mean(abs(smoothness_edge_constant * x), dim=-3, keepdim=True))
    else:
        raise ValueError('Only gaussian or exponential edge weighting '
                         'implemented.')

def robust_l1(x):
  """Robust L1 metric."""
  return (x**2 + 0.001**2)**0.5

class L2(nn.Module):
    def __init__(self, boundary_ignore=None):
        super().__init__()
        self.boundary_ignore = boundary_ignore

    def forward(self, pred, gt, valid=None):
        if self.boundary_ignore is not None:
            pred = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            if valid is not None:
                valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        pred_m = pred
        gt_m = gt

        if valid is None:
            mse = F.mse_loss(pred_m, gt_m)
        else:
            mse = F.mse_loss(pred_m, gt_m, reduction='none')

            eps = 1e-12
            elem_ratio = mse.numel() / valid.numel()
            mse = (mse * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)

        return mse + 1e-6


class PSNRM(nn.Module):
    def __init__(self, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.l2 = L2(boundary_ignore=boundary_ignore)
        self.max_value = max_value

    def psnr(self, pred, gt, valid=None):
        mse = self.l2(pred, gt, valid=valid)

        psnr = 20 * math.log10(self.max_value) - 10.0 * mse.log10()

        return psnr

    def forward(self, pred, gt, valid=None):
        assert pred.dim() == 4 and pred.shape == gt.shape
        if valid is None:
            psnr_all = [self.psnr(p.unsqueeze(0), g.unsqueeze(0)) for p, g in
                        zip(pred, gt)]
        else:
            psnr_all = [self.psnr(p.unsqueeze(0), g.unsqueeze(0), v.unsqueeze(0)) for p, g, v in zip(pred, gt, valid)]
        psnr = sum(psnr_all) / len(psnr_all)
        return psnr
