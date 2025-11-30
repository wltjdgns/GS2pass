#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
try:
    from diff_gaussian_rasterization._C import fusedssim, fusedssim_backward
except:
    pass

C1 = 0.01 ** 2
C2 = 0.03 ** 2

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2):
        ssim_map = fusedssim(C1, C2, img1, img2)
        ctx.save_for_backward(img1.detach(), img2)
        ctx.C1 = C1
        ctx.C2 = C2
        return ssim_map

    @staticmethod
    def backward(ctx, opt_grad):
        img1, img2 = ctx.saved_tensors
        C1, C2 = ctx.C1, ctx.C2
        grad = fusedssim_backward(C1, C2, img1, img2, opt_grad)
        return None, None, grad, None

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def fast_ssim(img1, img2):
    ssim_map = FusedSSIMMap.apply(C1, C2, img1, img2)
    return ssim_map.mean()

def compute_pseudo_normal(depthmap, scale_factor=15.0):
    """
    Compute pseudo normal from depth using central difference
    
    Args:
        depthmap: (H, W, 1) - depth map in view/camera space
        scale_factor: gradient scale (default 20.0)
    
    Returns:
        pseudo_normal: (H, W, 3) - normalized normal map in view space
    """
    H, W = depthmap.shape[:2]
    depth_2d = depthmap.squeeze(-1)  # (H, W)
    
    # Pad depth map to handle boundaries
    depth_padded = F.pad(depth_2d.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
    depth_padded = depth_padded.squeeze(0).squeeze(0)  # (H+2, W+2)
    
    # Central difference for gradients
    # dz/dx: (right - left) / 2
    dz_dx = (depth_padded[1:-1, 2:] - depth_padded[1:-1, :-2]) / 2.0  # (H, W)
    # dz/dy: (bottom - top) / 2
    dz_dy = (depth_padded[2:, 1:-1] - depth_padded[:-2, 1:-1]) / 2.0  # (H, W)
    
    # Apply scale factor
    dz_dx = dz_dx * scale_factor
    dz_dy = dz_dy * scale_factor
    
    # Construct normal: cross product of tangent vectors
    # Tangent X: (1, 0, dz/dx)
    # Tangent Y: (0, 1, dz/dy)
    # Normal = Tangent_X × Tangent_Y = (-dz/dx, -dz/dy, 1)
    
    normal_x = -dz_dx
    normal_y = -dz_dy
    normal_z = -torch.ones_like(dz_dx)
    
    # Stack and normalize
    pseudo_normal = torch.stack([normal_x, normal_y, normal_z], dim=-1)  # (H, W, 3)
    pseudo_normal = F.normalize(pseudo_normal, p=2, dim=-1)
    
    return pseudo_normal

def normal_loss(rendered_normal, pseudo_normal, reduction='mean'):
    """
    Normal consistency loss (개선된 버전)
    
    Args:
        rendered_normal: (H, W, 3) rendered normal
        pseudo_normal: (H, W, 3) pseudo normal from depth
        reduction: 'mean' or 'sum'
    
    Returns:
        loss: scalar
    """
    # ✅ 1. Cosine similarity loss (기존)
    cos_sim = torch.sum(rendered_normal * pseudo_normal, dim=-1)  # (H, W)
    loss_cos = 1.0 - cos_sim.mean()  # [0, 2]
    
    # ✅ 2. L1 loss 추가 (gradient 강화)
    loss_l1 = torch.abs(rendered_normal - pseudo_normal).mean()
    
    # ✅ 3. Combined loss
    loss = loss_cos + 0.5 * loss_l1  # Weighted sum
    
    return loss


def depth_uncertainty_loss(rendered_depth):
    """
    Relightable3DGaussian Eq.5: Depth uncertainty loss
    Minimize variance of depth distribution to encourage sharp surfaces
    L_u = E[D^2] - E[D]^2 = Variance(D)
    Args:
        rendered_depth: (H, W, 1) or (H*W,) per-pixel depth from blending
    Returns:
        loss: scalar
    """
    # Flatten depth map
    depth_flat = rendered_depth.reshape(-1)
    
    mean_depth = depth_flat.mean()
    mean_depth_sq = (depth_flat ** 2).mean()
    
    variance = mean_depth_sq - (mean_depth ** 2)
    
    return variance

# utils/loss_utils.py에 추가
def smoothness_loss_with_edge_aware(rendered_map, gt_image, epsilon=1e-3):
    """
    Edge-aware smoothness loss (Eq. 12 from paper)
    
    Args:
        rendered_map: (H, W) or (1, H, W) - rendered roughness/normal/albedo
        gt_image: (3, H, W) - ground truth RGB image
        epsilon: small value to avoid division by zero
    
    Returns:
        loss: scalar
    """
    # Ensure correct dimensions
    if rendered_map.dim() == 3:
        rendered_map = rendered_map.squeeze(0)  # (H, W)
    
    if gt_image.dim() == 3:
        # Convert RGB to grayscale for gradient computation
        gt_gray = 0.299 * gt_image[0] + 0.587 * gt_image[1] + 0.114 * gt_image[2]  # (H, W)
    else:
        gt_gray = gt_image
    
    # Compute gradients of rendered map (∇R)
    grad_r_x = rendered_map[:, 1:] - rendered_map[:, :-1]  # (H, W-1)
    grad_r_y = rendered_map[1:, :] - rendered_map[:-1, :]  # (H-1, W)
    
    # Compute gradients of GT image (∇C_gt)
    grad_gt_x = gt_gray[:, 1:] - gt_gray[:, :-1]  # (H, W-1)
    grad_gt_y = gt_gray[1:, :] - gt_gray[:-1, :]  # (H-1, W)
    
    # Gradient magnitudes
    grad_r_x_mag = torch.abs(grad_r_x)
    grad_r_y_mag = torch.abs(grad_r_y)
    
    grad_gt_x_mag = torch.abs(grad_gt_x)
    grad_gt_y_mag = torch.abs(grad_gt_y)
    
    # Edge-aware weights: exp(-||∇C_gt||)
    # 경계에서는 weight가 작아지고(smoothness 약함), 평탄한 곳에서는 weight가 커짐(smoothness 강함)
    weight_x = torch.exp(-grad_gt_x_mag[:, :-1])  # Match dimensions
    weight_y = torch.exp(-grad_gt_y_mag[:-1, :])
    
    # Weighted smoothness loss
    loss_x = (grad_r_x_mag[:, :-1] * weight_x).mean()
    loss_y = (grad_r_y_mag[:-1, :] * weight_y).mean()
    
    return loss_x + loss_y


def smoothness_loss_normal(rendered_normal, gt_image, epsilon=1e-3):
    """
    Edge-aware smoothness for normal map (3 channels)
    
    Args:
        rendered_normal: (3, H, W) - rendered normal map
        gt_image: (3, H, W) - ground truth RGB
    """
    # Apply to each channel
    loss = 0.0
    for c in range(3):
        loss += smoothness_loss_with_edge_aware(rendered_normal[c], gt_image, epsilon)
    return loss / 3.0

