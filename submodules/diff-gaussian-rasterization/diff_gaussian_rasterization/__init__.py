from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
    depths,
    normals,
    basecolors,      # === 추가 ===
    roughness,       # === 추가 ===
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        depths,
        normals,
        basecolors,      # === 추가 ===
        roughness)       # === 추가 ===


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means3D, means2D, sh, colors_precomp, opacities, scales, rotations, 
                cov3Ds_precomp, raster_settings, depths, normals, basecolors, roughness):
        
        # Homography setup
        homo = bool(getattr(raster_settings, "homo_grid", False))
        Hmat = getattr(raster_settings, "Hmat", None)
        if Hmat is None:
            Hmat = torch.eye(3, device=means3D.device, dtype=torch.float32)
        else:
            if not isinstance(Hmat, torch.Tensor):
                Hmat = torch.tensor(Hmat, dtype=torch.float32, device=means3D.device)
            else:
                Hmat = Hmat.to(device=means3D.device, dtype=torch.float32)
            Hmat = Hmat.reshape(3, 3)

        # Restructure arguments for C++ lib
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.antialiasing,
            raster_settings.debug,
            homo,
            Hmat,
            depths,
            normals,
            basecolors,      # === 추가 ===
            roughness        # === 추가 ===
        )

        # Invoke C++/CUDA rasterizer
        num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, \
            invdepths, rendered_depth, rendered_normal, rendered_basecolor, rendered_roughness = \
            _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, 
                             radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer, 
                             depths, normals, basecolors, roughness)

        return color, radii, invdepths, rendered_normal, rendered_depth, \
               rendered_basecolor, rendered_roughness  # === 추가 ===

    @staticmethod
    def backward(ctx, grad_out_color, _, grad_out_depth, grad_out_rendered_normal, 
                grad_out_rendered_depth, grad_out_rendered_basecolor, grad_out_rendered_roughness):
        
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, \
            geomBuffer, binningBuffer, imgBuffer, depths_in, normals_in, basecolors_in, roughness_in = \
            ctx.saved_tensors

        empty_tensor = torch.empty(0, device=means3D.device, dtype=torch.float32)

        args = (
            raster_settings.bg,                   # 0
            means3D,                              # 1
            radii,                                # 2
            colors_precomp,                       # 3
            opacities,                            # 4
            scales,                               # 5
            rotations,                            # 6
            raster_settings.scale_modifier,       # 7
            cov3Ds_precomp,                       # 8
            raster_settings.viewmatrix,           # 9
            raster_settings.projmatrix,           # 10
            raster_settings.tanfovx,              # 11
            raster_settings.tanfovy,              # 12
            grad_out_color,                       # 13
            grad_out_depth,                       # 14
            grad_out_rendered_depth if grad_out_rendered_depth is not None else empty_tensor,  # 15
            grad_out_rendered_normal if grad_out_rendered_normal is not None else empty_tensor, # 16
            sh,                                   # 17 ← 추가!
            raster_settings.sh_degree,            # 18
            raster_settings.campos,               # 19
            geomBuffer,                           # 20
            num_rendered,                         # 21
            binningBuffer,                        # 22
            imgBuffer,                            # 23
            depths_in,                            # 24
            normals_in,                           # 25
            basecolors_in,                        # 26
            roughness_in,                         # 27
            grad_out_rendered_basecolor if grad_out_rendered_basecolor is not None else empty_tensor,  # 28
            grad_out_rendered_roughness if grad_out_rendered_roughness is not None else empty_tensor,  # 29
            raster_settings.antialiasing,         # 30
            raster_settings.debug                 # 31
        )

        grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, \
            grad_sh, grad_scales, grad_rotations, grad_depths, grad_normals, \
            grad_basecolors, grad_roughness = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,           # raster_settings
            grad_depths,
            grad_normals,
            grad_basecolors,
            grad_roughness,
        )

        return grads


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    antialiasing : bool
    homo_grid: bool
    Hmat: torch.Tensor


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
        return visible

    def forward(self, means3D, means2D, opacities, shs=None, colors_precomp=None, 
                scales=None, rotations=None, cov3D_precomp=None, 
                depths=None, normals=None, basecolors=None, roughness=None):  # === 추가 ===
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide exactly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or \
           ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])
        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke rasterization routine
        return rasterize_gaussians(
            means3D, means2D, shs, colors_precomp, opacities, scales, rotations, 
            cov3D_precomp, raster_settings, depths, normals, basecolors, roughness)
