import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np

from scipy.ndimage import label  # Connected components (optional)


def detect_planar_from_rendered_normal(
    viewpoint_camera,
    gaussians,
    pipeline,
    background,
    render_func,
    normal_threshold=0.98,
    min_pixel_count=1000
):
    """
    Rendered normal map에서 planar 영역 감지 후 해당 Gaussians 추출
    
    Args:
        viewpoint_camera: Camera object
        gaussians: GaussianModel
        pipeline: Pipeline parameters
        background: Background tensor
        render_func: render() 함수 (from gaussian_renderer)
        normal_threshold: Normal similarity threshold [0.95~0.99]
        min_pixel_count: 최소 픽셀 개수
    
    Returns:
        planar_indices: Tensor - Planar Gaussians의 indices
        plane_normal: (3,) - 평면 법선
        plane_center: (3,) - 평면 중심
    """
    
    # ===== Step 1: Render Normal Map =====
    with torch.no_grad():  # Planar detection은 gradient 불필요
        render_pkg = render_func(
            viewpoint_camera, 
            gaussians, 
            pipeline, 
            background
        )
    
    normal_map = render_pkg.get('normal', None)
    
    if normal_map is None:
        print("⚠️ Normal map not available")
        return None, None, None
    
    # (3, H, W) -> (H, W, 3)
    normal_map = normal_map.permute(1, 2, 0)
    H, W = normal_map.shape[:2]
    
    # ===== Step 2: Find Dominant Normal Direction =====
    # Flatten to (H*W, 3)
    normals_flat = normal_map.reshape(-1, 3)
    normals_norm = F.normalize(normals_flat, dim=1)
    
    # Valid pixels (non-zero normals)
    valid_mask = (normals_flat.norm(dim=1) > 0.1)
    valid_normals = normals_norm[valid_mask]
    
    if valid_normals.shape[0] < min_pixel_count:
        print("⚠️ Not enough valid pixels for planar detection")
        return None, None, None
    
    # Dominant normal (평균 방향)
    dominant_normal = valid_normals.mean(dim=0)
    dominant_normal = F.normalize(dominant_normal, dim=0)
    
    # ===== Step 3: Segment Planar Region =====
    # Pixels with similar normal to dominant
    similarity = torch.mv(normals_norm, dominant_normal)
    planar_pixel_mask = (similarity > normal_threshold) & valid_mask
    planar_pixel_mask = planar_pixel_mask.reshape(H, W)
    
    planar_pixel_count = planar_pixel_mask.sum().item()
    
    if planar_pixel_count < min_pixel_count:
        print(f"⚠️ Planar region too small: {planar_pixel_count} pixels")
        return None, None, None
    
    
    # ===== Step 4: Back-project to Gaussians =====
    # Approximation: Gaussians with similar pointwise normal
    pointwise_normals = gaussians.normal  # (N, 3)
    pointwise_norm = F.normalize(pointwise_normals, dim=1)
    
    # Similarity to dominant normal
    gaussian_similarity = torch.mv(pointwise_norm, dominant_normal)
    
    # Threshold
    planar_gaussian_mask = (gaussian_similarity > 0.95)  # Slightly lower threshold
    planar_indices = torch.where(planar_gaussian_mask)[0]
    
    if len(planar_indices) < 100:
        print(f"⚠️ Not enough planar Gaussians: {len(planar_indices)}")
        return None, None, None
    
    # ===== Step 5: Compute Plane Parameters =====
    planar_positions = gaussians.get_xyz[planar_indices]
    planar_normals_pointwise = pointwise_normals[planar_indices]
    
    plane_center = planar_positions.mean(dim=0)
    plane_normal = F.normalize(planar_normals_pointwise.mean(dim=0), dim=0)
    
    
    return planar_indices, plane_normal, plane_center

def compute_plane_equation(plane_normal, plane_center):
    """
    평면 방정식 계산: n^T x + d = 0
    
    Args:
        plane_normal: (3,) - 단위 법선 벡터
        plane_center: (3,) - 평면 위의 한 점
    
    Returns:
        plane_offset: float - plane equation의 d
    """
    # d = -n^T * center
    plane_offset = -torch.dot(plane_normal, plane_center)
    return plane_offset


def compute_virtual_camera(camera, plane_normal, plane_offset):
    """
    Mirror camera across the plane to create virtual camera
    """
    from scene.cameras import Camera
    
    # Camera center in world space
    camera_center = camera.camera_center  # (3,)
    
    # Mirror camera center across plane
    distance = torch.dot(plane_normal, camera_center) + plane_offset
    t_virtual = camera_center - 2 * distance * plane_normal
    
    # Mirror rotation
    R_mirror = torch.eye(3, device="cuda") - 2 * torch.outer(plane_normal, plane_normal)
    
    # Camera rotation (world to camera)
    R_world_to_cam = camera.world_view_transform[:3, :3].t()
    
    # Virtual camera rotation
    R_virtual_to_world = R_mirror @ R_world_to_cam
    
    # Tensor를 PIL Image로 변환
    original_tensor = camera.original_image  # 3, H, W
    image_pil = TF.to_pil_image(original_tensor.cpu())

    virtual_cam = Camera(
        resolution=(camera.image_width, camera.image_height),
        colmap_id=-1,
        R=R_virtual_to_world.detach().cpu().numpy(),
        T=(-R_virtual_to_world.t() @ t_virtual).detach().cpu().numpy(),
        FoVx=camera.FoVx,
        FoVy=camera.FoVy,
        depth_params=None,
        image=image_pil,  #  PIL Image 전달
        invdepthmap=None,
        image_name="virtual",
        uid=-1,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
        data_device="cuda",
        train_test_exp=False,
        is_test_dataset=False,
        is_test_view=False
    )
    return virtual_cam



def compute_virtual_camera_simple(camera, plane_normal, plane_center):
    """
    간단한 버전: plane_center로부터 직접 계산
    
    Args:
        camera: Camera object
        plane_normal: (3,) - 정규화된 법선
        plane_center: (3,) - 평면 중심점
    
    Returns:
        virtual_camera: Camera object
    """
    plane_offset = compute_plane_equation(plane_normal, plane_center)
    return compute_virtual_camera(camera, plane_normal, plane_offset)
