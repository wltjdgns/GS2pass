import torch
import numpy as np
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from .render_2pass import render_2pass
from scene.envmap import EnvLight

__all__ = ['render', 'EnvLight']
_h_dbg_printed = False

def _fit_normalize_H_to_viewport(H_cuda, width, height, mode="contain"):
    """
    H_cuda: (3,3) CUDA float32 tensor
    width,height: viewport (int)
    mode: "none" | "contain" | "cover" | "stretch"
    Return: H_norm = N @ H (torch.cuda.FloatTensor 3x3)
    """
    if mode == "none" or mode is False or mode is None:
        return H_cuda

    # 1) 원본 4 코너
    corners = torch.tensor([[0., 0., 1.],
                            [float(width), 0., 1.],
                            [float(width), float(height), 1.],
                            [0., float(height), 1.]], device=H_cuda.device, dtype=H_cuda.dtype)

    # 2) 워프
    q = (H_cuda @ corners.T).T
    w = q[:, 2].clamp(min=1e-8)
    x = q[:, 0] / w
    y = q[:, 1] / w

    minx, _ = torch.min(x, dim=0)
    maxx, _ = torch.max(x, dim=0)
    miny, _ = torch.min(y, dim=0)
    maxy, _ = torch.max(y, dim=0)

    width_p  = (maxx - minx).clamp(min=1e-8)
    height_p = (maxy - miny).clamp(min=1e-8)

    if mode == "stretch":
        sx = float(width)  / float(width_p)
        sy = float(height) / float(height_p)
        tx = -sx * float(minx)
        ty = -sy * float(miny)
        N = torch.tensor([[sx, 0., tx],
                          [0., sy, ty],
                          [0., 0., 1. ]], device=H_cuda.device, dtype=H_cuda.dtype)
    else:
        s_contain = float(min(float(width)/float(width_p), float(height)/float(height_p)))
        s_cover   = float(max(float(width)/float(width_p), float(height)/float(height_p)))
        s = s_contain if mode == "contain" else s_cover
        tx = 0.5 * (float(width)  - s * float(width_p))  - s * float(minx)
        ty = 0.5 * (float(height) - s * float(height_p)) - s * float(miny)
        N = torch.tensor([[s, 0., tx],
                          [0., s, ty],
                          [0., 0., 1. ]], device=H_cuda.device, dtype=H_cuda.dtype)

    return N @ H_cuda

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, 
           scaling_modifier = 1.0, separate_sh = False, override_color = None, use_trained_exp=False):
    """
    Render the scene with Depth/Normal/Basecolor/Roughness support.
    """

    # Homography setup
    def _to_cuda_H(x):
        if x is None:
            return torch.eye(3, device="cuda", dtype=torch.float32)
        if isinstance(x, torch.Tensor):
            return x.to(device="cuda", dtype=torch.float32).reshape(3,3)
        return torch.tensor(x, dtype=torch.float32, device="cuda").reshape(3,3)

    use_homo = bool(getattr(pipe, "homo_grid", False))
    H_src = None
    if use_homo:
        if hasattr(viewpoint_camera, "Hmat") and viewpoint_camera.Hmat is not None:
            H_src = viewpoint_camera.Hmat
        elif getattr(pipe, "Hmat", None) is not None:
            H_src = pipe.Hmat

    H_cuda = _to_cuda_H(H_src)
    H_cuda = _fit_normalize_H_to_viewport(
        H_cuda,
        int(viewpoint_camera.image_width),
        int(viewpoint_camera.image_height),
        mode=getattr(pipe, "fit_mode", "contain")
    )
 
    # Gradient tracking
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, 
                                         requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing,
        homo_grid=use_homo,
        Hmat=H_cuda
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # Covariance
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # Color
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            if separate_sh:
                dc, shs = pc.get_features_dc, pc.get_features_rest
            else:
                shs = pc.get_features
    else:
        colors_precomp = override_color

    means3D = pc.get_xyz  # (N,3)
    ones = torch.ones((means3D.shape[0], 1), device=means3D.device, dtype=means3D.dtype)
    means3D_hom = torch.cat([means3D, ones], dim=1)  # (N,4)
    viewmatrix = viewpoint_camera.world_view_transform  # (4,4)
    means_view = (viewmatrix @ means3D_hom.T).T  # (N,4)
    depths_in = means_view[:, 2:3]

    if pc.normal.numel() > 0:
        # world space normal
        normals_world = pc.normal.to(device="cuda", dtype=torch.float32)  # (N,3)
        # 카메라 view rotation 행렬 (3x3)
        R = viewpoint_camera.world_view_transform[:3, :3]
        # view space 변환
        normals_view = (R @ normals_world.T).T
        normals_in = normals_view
    else:
        normals_in = torch.empty((0, 3), device="cuda", dtype=torch.float32)


    # === 수정: Basecolor/Roughness ===
    if hasattr(pc, 'get_basecolor') and pc.get_basecolor.numel() > 0:
        basecolors_in = pc.get_basecolor.to(device="cuda", dtype=torch.float32)
    else:
        basecolors_in = torch.empty((0, 3), device="cuda", dtype=torch.float32)

    if hasattr(pc, 'get_roughness') and pc.get_roughness.numel() > 0:
        roughness_in = pc.get_roughness.to(device="cuda", dtype=torch.float32)
        if roughness_in.dim() == 1:
            roughness_in = roughness_in.unsqueeze(-1)  # (N,) -> (N,1)
    else:
        roughness_in = torch.empty((0, 1), device="cuda", dtype=torch.float32)

    # Rasterize with basecolor/roughness
    if separate_sh:
        rendered_image, radii, depth_image, rendered_normal, rendered_depth, \
            rendered_basecolor, rendered_roughness = rasterizer(
            means3D = means3D,
            means2D = means2D,
            dc = dc,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            depths = depths_in,
            normals = normals_in,
            basecolors = basecolors_in,      # === 추가 ===
            roughness = roughness_in)        # === 추가 ===
    else:
        rendered_image, radii, depth_image, rendered_normal, rendered_depth, \
            rendered_basecolor, rendered_roughness = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp,
            depths = depths_in,
            normals = normals_in,
            basecolors = basecolors_in,      # === 추가 ===
            roughness = roughness_in)        # === 추가 ===

    # Apply exposure
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3, None, None]

    rendered_image = rendered_image.clamp(0, 1)
    
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image,
        "normal": rendered_normal,
        "depth_per_gaussian": rendered_depth,
        "albedomap": rendered_basecolor,     # === 추가: (3, H, W) ===
        "roughnessmap": rendered_roughness   # === 추가: (1, H, W) ===
    }
    
    return out
