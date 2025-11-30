# utils/planar_utils.py
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import cv2

from scipy.ndimage import label  # Connected components
from utils.loss_utils import compute_pseudo_normal  # pseudo normal from depth


def detect_planar_from_rendered_normal(
    viewpoint_camera,
    gaussians,
    pipeline,
    background,
    render_func,
    edge_threshold=0.05,
    min_pixel_count=500,
):
    """
    Depth + pseudo normal + rendered normal 기반 planar 영역 검출.

    흐름:
      1) render_func로 depth, normal 렌더링
      2) depth로 pseudo normal 생성
      3) pseudo normal에 Sobel edge → edge map
      4) edge로 닫힌 영역들(connected components) 중, 픽셀 수가 min_pixel_count 이상인
         가장 큰 영역을 planar 후보로 선택
      5) 선택된 영역에서 rendered normal 평균 → plane_normal
      6) plane_normal과 per-Gaussian normal의 cosine similarity로 planar_indices 필터링
      7) planar_indices의 3D 위치 평균 → plane_center

    Args:
        viewpoint_camera: Camera 객체
        gaussians: GaussianModel (gaussians.normal, gaussians.get_xyz 사용)
        pipeline: PipelineParams
        background: (3,) torch.Tensor, background color
        render_func: callable(viewpoint_camera, gaussians, pipeline, background) -> dict
                     dict 안에 "depth"(1,H,W), "normal"(3,H,W) 키를 포함해야 함
        edge_threshold: Sobel edge magnitude threshold (0~1 스케일 가정 후 0~255로 곱해 사용)
        min_pixel_count: planar 후보 최소 픽셀 수

    Returns:
        planar_indices: torch.LongTensor, shape (K,)
            - planar로 판정된 Gaussian들의 인덱스
        plane_normal: torch.FloatTensor, shape (3,)
            - 선택된 planar 영역의 dominant normal
            - rendered normal 평균 (fallback 시 pseudo normal 평균)
        plane_center: torch.FloatTensor, shape (3,)
            - planar_indices에 해당하는 Gaussian 위치의 평균 (world space)
        실패 시: (None, None, None)
    """

    # ===== Step 1: Render depth / normal =====
    with torch.no_grad():
        render_pkg = render_func(
            viewpoint_camera,
            gaussians,
            pipeline,
            background,
        )

    depth_map = render_pkg.get("depth", None)   # (1,H,W) or (H,W)
    normal_map = render_pkg.get("normal", None) # (3,H,W)

    if depth_map is None:
        print("⚠️ Depth map not available")
        return None, None, None

    if depth_map.dim() == 3:
        depth_map = depth_map.squeeze(0)  # (H,W)
    H, W = depth_map.shape

    # ===== Step 2: pseudo normal from depth =====
    # depth_for_pseudo: (H,W,1)
    depth_for_pseudo = depth_map.unsqueeze(-1)
    # compute_pseudo_normal: (H,W,1) -> (H,W,3), [-1,1] normalized
    pseudo_normal = compute_pseudo_normal(depth_for_pseudo, scale_factor=20.0)

    # ===== Step 3: Edge detection on pseudo normal =====
    # pseudo normal [-1,1] → [0,255] RGB → gray
    pseudo_normal_np = pseudo_normal.cpu().numpy()
    pseudo_vis = (pseudo_normal_np * 0.5 + 0.5) * 255.0  # (H,W,3), 0~255
    pseudo_vis = pseudo_vis.astype(np.uint8)
    gray = cv2.cvtColor(pseudo_vis, cv2.COLOR_RGB2GRAY)  # (H,W), uint8

    # Sobel edge
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)  # float32

    # threshold를 0~1로 받았다고 보고 0~255 스케일로 변환
    thr = edge_threshold * 255.0
    _, edge_mask = cv2.threshold(edge_mag, thr, 1.0, cv2.THRESH_BINARY)

    edge_mask = edge_mask.astype(np.uint8)          # 0 or 1
    closed_regions = cv2.bitwise_not(edge_mask)     # 엣지 바깥 영역(후보 영역)

    # ===== Step 4: Connected components로 planar 후보 찾기 =====
    num_labels, labels = cv2.connectedComponents(closed_regions)

    if num_labels <= 1:
        print("⚠️ No planar regions found (connected components)")
        return None, None, None

    best_region = None
    best_area = 0

    for lbl in range(1, num_labels):
        region_mask = (labels == lbl)
        area = int(region_mask.sum())
        if area < min_pixel_count:
            continue
        if area > best_area:
            best_area = area
            best_region = region_mask

    if best_region is None:
        print(f"⚠️ No region larger than {min_pixel_count} pixels")
        return None, None, None

    # torch mask로 변환
    best_region_mask = torch.from_numpy(best_region).to(pseudo_normal.device)  # (H,W), bool

    # ===== Step 5: plane_normal 계산 (rendered normal 평균) =====
    if normal_map is not None:
        # normal_map: (3,H,W) -> (H,W,3)
        normal_hw3 = normal_map.permute(1, 2, 0)
        region_normals = normal_hw3[best_region_mask]         # (N,3)
    else:
        print("⚠️ Normal map not available, fallback to pseudo normal for plane_normal")
        region_normals = pseudo_normal[best_region_mask]      # (N,3)

    if region_normals.numel() == 0:
        print("⚠️ Empty region_normals")
        return None, None, None

    region_normals = F.normalize(region_normals, dim=-1)
    plane_normal = F.normalize(region_normals.mean(dim=0), dim=0)  # (3,)

    pointwise_normals = gaussians.get_minaxis(viewpoint_camera.camera_center)
    pointwise_norm = F.normalize(pointwise_normals, dim=1)

    # cosine similarity
    gaussian_similarity = torch.mv(pointwise_norm, plane_normal)   # (N,)
    planar_gaussian_mask = gaussian_similarity > 0.90              # threshold 완화 가능
    planar_indices = torch.where(planar_gaussian_mask)[0]

    if planar_indices.numel() < 100:
        print(f"⚠️ Not enough planar Gaussians: {planar_indices.numel()}")
        return None, None, None

    # ===== Step 7: plane_center (world space) =====
    planar_positions = gaussians.get_xyz[planar_indices]  # (K,3)
    plane_center = planar_positions.mean(dim=0)           # (3,)

    return planar_indices, plane_normal, plane_center


def compute_plane_equation(plane_normal, plane_center):
    """
    평면 방정식 n^T x + d = 0 의 d 계산

    Args:
        plane_normal: (3,) 단위 법선
        plane_center: (3,) 평면 상의 한 점

    Returns:
        plane_offset: float (스칼라 텐서) - d 값
    """
    plane_offset = -torch.dot(plane_normal, plane_center)
    return plane_offset


def compute_virtual_camera(camera, plane_normal, plane_offset):
    """
    주어진 평면에 대해 카메라를 반사시킨 virtual camera 생성.

    Args:
        camera: Camera 객체 (scene.cameras.Camera)
        plane_normal: (3,) 단위 법선 (world space)
        plane_offset: float, 평면 방정식 n^T x + d = 0 의 d

    Returns:
        virtual_cam: Camera 객체
    """
    from scene.cameras import Camera

    # 카메라 중심 (world space)
    camera_center = camera.camera_center  # (3,)

    # 평면에 대한 signed distance
    distance = torch.dot(plane_normal, camera_center) + plane_offset
    t_virtual = camera_center - 2 * distance * plane_normal  # 반사된 위치

    # 회전 반사 행렬 R_mirror
    R_mirror = torch.eye(3, device=camera_center.device) - 2 * torch.outer(
        plane_normal, plane_normal
    )

    # world_to_cam 회전
    R_world_to_cam = camera.world_view_transform[:3, :3].t()

    # virtual camera의 world 방향 회전
    R_virtual_to_world = R_mirror @ R_world_to_cam

    # 이미지를 PIL로 변환 (Camera 생성자 입력 형식 맞추기 위함)
    original_tensor = camera.original_image  # (3,H,W)
    image_pil = TF.to_pil_image(original_tensor.cpu())

    virtual_cam = Camera(
        resolution=(camera.image_width, camera.image_height),
        colmap_id=-1,
        R=R_virtual_to_world.detach().cpu().numpy(),
        T=(-R_virtual_to_world.t() @ t_virtual).detach().cpu().numpy(),
        FoVx=camera.FoVx,
        FoVy=camera.FoVy,
        depth_params=None,
        image=image_pil,
        invdepthmap=None,
        image_name="virtual",
        uid=-1,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
        train_test_exp=False,
        is_test_dataset=False,
        is_test_view=False,
    )
    return virtual_cam


def compute_virtual_camera_simple(camera, plane_normal, plane_center):
    """
    plane_center로부터 plane_offset을 계산한 뒤 virtual camera를 생성하는 헬퍼.

    Args:
        camera: Camera 객체
        plane_normal: (3,) 단위 법선
        plane_center: (3,) 평면 중심

    Returns:
        virtual_cam: Camera 객체
    """
    plane_offset = compute_plane_equation(plane_normal, plane_center)
    return compute_virtual_camera(camera, plane_normal, plane_offset)
