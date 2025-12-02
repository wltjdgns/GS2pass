import torch
import torch.nn.functional as F
import sys
import os

from utils.planar_utils import (
    detect_planar_from_rendered_normal,
    compute_virtual_camera_simple
)


def render_2pass(
    viewpoint_camera,
    gaussians,
    pipeline,
    background,
    render_func,
    envlight=None,
    lambda_weight=0.3,
    enable_2pass=True,
    detect_planar_interval=100,
    iteration=0
):
    """2-pass rendering with rasterization-based reflection"""
    
    # ===== Pass 1: Base Rendering (Diffuse/SH) =====
    render_pkg_1 = render_func(
        viewpoint_camera,
        gaussians,
        pipeline,
        background
    )
    
    if not enable_2pass:
        return render_pkg_1
    
    # Initialize planar cache
    if not hasattr(render_2pass, 'planar_cache'):
        render_2pass.planar_cache = {
            'indices': None,
            'normal': None,
            'center': None,
            'last_detect': -1
        }
    
    cache = render_2pass.planar_cache
    
    # Detect planar surfaces periodically
    if iteration - cache['last_detect'] >= detect_planar_interval:
        planar_indices, plane_normal, plane_center = detect_planar_from_rendered_normal(
            viewpoint_camera,
            gaussians,
            pipeline,
            background,
            render_func=render_func,
            normal_threshold=0.98,
            min_pixel_count=1000
        )
        
        cache['indices'] = planar_indices
        cache['normal'] = plane_normal
        cache['center'] = plane_center
        cache['last_detect'] = iteration
        
        
        if planar_indices is None:
            return render_pkg_1
    
    planar_indices = cache['indices']
    plane_normal = cache['normal']
    plane_center = cache['center']
    
    if planar_indices is None or plane_normal is None:
        return render_pkg_1
    
    # ===== Pass 2: Virtual Camera Rendering (Specular Reflection) =====
    virtual_camera = compute_virtual_camera_simple(
        viewpoint_camera,
        plane_normal,
        plane_center
    )
    
    render_pkg_2 = render_func(
        virtual_camera,
        gaussians,
        pipeline,
        background
    )
    
    # ===== Extract BRDF Parameters =====
    albedomap = render_pkg_1.get('albedomap', None)
    roughnessmap = render_pkg_1.get('roughnessmap', None)
    normalmap = render_pkg_1.get('normal', None)
    
    # ===== Simple BRDF Weighting =====
    img_pass1 = render_pkg_1['render']
    img_pass2 = render_pkg_2['render']
    
    if roughnessmap is not None:
        roughness_factor = roughnessmap
        specular_weight = (1.0 - roughness_factor).clamp(0, 1)
        img_final = img_pass1 * (1 - specular_weight * lambda_weight) + \
                    img_pass2 * (specular_weight * lambda_weight)
    else:
        img_final = (1 - lambda_weight) * img_pass1 + lambda_weight * img_pass2
    
    # ===== Return Package (수정) =====
    render_pkg = {
        'render': img_final,
        'pass1': img_pass1,
        'pass2': img_pass2,
        'viewspace_points': render_pkg_1['viewspace_points'],
        'visibility_filter': render_pkg_1['visibility_filter'],
        'radii': render_pkg_1['radii'],
        'depth': render_pkg_1.get('depth', None),
        'normal': normalmap,
        'depth_per_gaussian': render_pkg_1.get('depth_per_gaussian', None),
        'albedomap': albedomap,
        'roughnessmap': roughnessmap,
        'lambda': lambda_weight,
        'planar_indices': planar_indices,
        'plane_normal': plane_normal,      # ← 추가!
        'plane_center': plane_center        # ← 추가!
    }
    
    return render_pkg