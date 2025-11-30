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
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


# ============================================================
# ✅ ReCap Normal 관련 함수 추가 (파일 끝에)
# ============================================================

def get_model_view_matrix_for_normal(world_view_transform, homo=True):
    """
    Normal 변환용 ModelView 행렬 생성
    
    Args:
        world_view_transform: (4, 4) world-to-view 변환 행렬
        homo: homogeneous 좌표 사용 여부
    Returns:
        (4, 4) or (3, 3) 변환 행렬
    """
    if homo:
        # Homogeneous 좌표계: (4, 4) 행렬의 translation 제거
        new_world_view = torch.eye(4, device=world_view_transform.device, dtype=world_view_transform.dtype)
        new_world_view[:3, :3] = world_view_transform[:3, :3]  # Rotation만 사용
        return new_world_view
    else:
        # (3, 3) Rotation 행렬만 반환
        return world_view_transform[:3, :3]


def prepare_normal(normal, camera):
    """
    Normal을 카메라 좌표계로 변환
    
    Args:
        normal: (N, 3) Gaussian normal vectors in world space
        camera: Camera object with world_view_transform
    Returns:
        (N, 3) Normal vectors in camera space
    """
    # World-to-camera 변환 행렬 (translation 제외)
    old_world_view = camera.world_view_transform.T  # (4, 4)
    new_world_view = get_model_view_matrix_for_normal(old_world_view, homo=True).cuda()
    
    # Homogeneous 좌표로 변환: (N, 3) → (N, 4)
    hom_normal = torch.cat([normal, torch.ones(normal.shape[0], 1).cuda()], dim=-1)  # (N, 4)
    
    # 변환 적용
    normal_transformed = torch.matmul(hom_normal, new_world_view.T)  # (N, 4)
    normal = normal_transformed[:, :3]  # (N, 3)
    
    # Normalize
    normal = torch.nn.functional.normalize(normal, p=2, dim=1)
    
    return normal
