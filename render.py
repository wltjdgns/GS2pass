
import torch
import numpy as np
import cv2
from scene import Scene
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from gaussian_renderer.render_2pass import render_2pass  #  추가
from scene.envmap import EnvLight  #  추가
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.loss_utils import compute_pseudo_normal

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False



def render_set(model_path, name, iteration, views, gaussians, pipeline, background, 
               train_test_exp, separate_sh, envlight=None, enable_2pass=False):
    planar_cache_per_view = {}
    planar_cache_path = os.path.join(model_path, f"planar_cache_{iteration}.pth")
    
    if os.path.exists(planar_cache_path):
        planar_cache_per_view = torch.load(planar_cache_path, map_location='cpu')
        print(f"Loaded planar cache: {len(planar_cache_per_view)} views from {planar_cache_path}")
    else:
        print(f"Planar cache not found: {planar_cache_path}")


    save_root = getattr(pipeline, "output_dir", None)
    if not save_root:
        save_root = os.path.join(model_path, name, f"ours_{iteration}")
    
    render_path = os.path.join(save_root, name)
    gts_path = os.path.join(save_root, f"{name}_gt")
    depth_path = os.path.join(save_root, f"{name}_depth")
    normal_path = os.path.join(save_root, f"{name}_normal")
    albedo_path = os.path.join(save_root, f"{name}_albedo")
    roughness_path = os.path.join(save_root, f"{name}_roughness")
    pseudo_path = os.path.join(save_root, f"{name}_pseudonormal")
    
    #  추가: Pass 1, Pass 2 저장 경로
    if enable_2pass:
        pass1_path = os.path.join(save_root, f"{name}_pass1")
        pass2_path = os.path.join(save_root, f"{name}_pass2")
        makedirs(pass1_path, exist_ok=True)
        makedirs(pass2_path, exist_ok=True)

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(normal_path, exist_ok=True)
    # makedirs(albedo_path, exist_ok=True)
    # makedirs(roughness_path, exist_ok=True)
    makedirs(pseudo_path, exist_ok=True)

    # === 추가: Planar visualization path ===
    planar_path = os.path.join(save_root, f"{name}_planar")
    makedirs(planar_path, exist_ok=True)
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        # [ADD] per-view H를 viewpoint_cam에 직접 세팅
        if getattr(pipeline, "homo_grid", False):
            if hasattr(pipeline, 'Hmat') and pipeline.Hmat is not None:
                view.Hmat = pipeline.Hmat
        
        #  2-pass rendering 유지 (Pass1, Pass2 이미지 필요)
        if enable_2pass and envlight is not None:
            render_pkg = render_2pass(
                view, gaussians, pipeline, background,
                render_func=render,
                envlight=envlight,
                lambda_weight=0.3,
                enable_2pass=True,
                detect_planar_interval=1,
                iteration=idx
            )
        else:
            # 기존 방식
            render_pkg = render(view, gaussians, pipeline, background,
                use_trained_exp=train_test_exp,
                separate_sh=separate_sh)
        
        rendering = render_pkg["render"]  # Final (blended)
        
        #  Pass 1, Pass 2 추출
        pass1_img = render_pkg.get("pass1", None)
        pass2_img = render_pkg.get("pass2", None)
        
        # Depth, Normal, Albedo, Roughness 추출
        depth_rendered = render_pkg.get("depth", None)
        normal_rendered = render_pkg.get("normal", None)

        gt = view.original_image[0:3, :, :]
        
        #  핵심: Cache에서 planar 정보 가져오기 (render2pass 결과 대신 사용)
        view_key = view.image_name
        planar_data = planar_cache_per_view.get(view_key, None)
        
        # render_pkg에서 가져온 planar_indices 대신 cache 사용
        if planar_data is not None:
            planar_indices = planar_data['indices'].cuda()
        else:
            planar_indices = render_pkg.get("planar_indices", None)  # Fallback
            # train_test_exp 처리
        if train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]
            if depth_rendered is not None:
                depth_rendered = depth_rendered[..., depth_rendered.shape[-1] // 2:]
            if normal_rendered is not None:
                normal_rendered = normal_rendered[..., normal_rendered.shape[-1] // 2:]
            #  Pass 1, Pass 2도 처리
            if pass1_img is not None:
                pass1_img = pass1_img[..., pass1_img.shape[-1] // 2:]
            if pass2_img is not None:
                pass2_img = pass2_img[..., pass2_img.shape[-1] // 2:]

        # Final 렌더링 이미지 저장
        torchvision.utils.save_image(rendering, 
            os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        
        # GT 이미지 저장
        torchvision.utils.save_image(gt, 
            os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        
        #  Pass 1, Pass 2 저장
        if enable_2pass:
            if pass1_img is not None:
                torchvision.utils.save_image(pass1_img, 
                    os.path.join(pass1_path, '{0:05d}'.format(idx) + ".png"))
            if pass2_img is not None:
                torchvision.utils.save_image(pass2_img, 
                    os.path.join(pass2_path, '{0:05d}'.format(idx) + ".png"))
        
        # Depth 저장 (정규화: 0-1)
        if depth_rendered is not None:
            depth_norm = depth_rendered.squeeze(0)
            depth_min = depth_norm.min()
            depth_max = depth_norm.max()
            if depth_max > depth_min:
                depth_norm = (depth_norm - depth_min) / (depth_max - depth_min)
            else:
                depth_norm = torch.zeros_like(depth_norm)
            
            torchvision.utils.save_image(depth_norm.unsqueeze(0), 
                os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
            # === Pseudo Normal 계산 및 저장 ===
            depth_for_pseudo = depth_norm.unsqueeze(-1)  # (H, W, 1)
            pseudo_normal = compute_pseudo_normal(depth_for_pseudo, scale_factor = 15.0)  # (H, W, 3)

            # 시각화: [-1,1] -> [0,1]

            pseudo_normal_visual = (pseudo_normal + 1.0) / 2.0
            pseudo_normal_visual = torch.clamp(pseudo_normal_visual, 0, 1)
            pseudo_normal_visual = pseudo_normal_visual.permute(2, 0, 1)  # (3, H, W)

            torchvision.utils.save_image(pseudo_normal_visual, 
                os.path.join(pseudo_path, '{0:05d}'.format(idx) + ".png"))
        else:
            # ← depth가 None일 경우 경고 출력
            print(f"Warning: depth_rendered is None for frame {idx}")
        
        # Normal 저장 (RGB로 변환: [-1,1] -> [0,1])
        if normal_rendered is not None:
            normal_visual = (normal_rendered + 1.0) / 2.0
            normal_visual = torch.clamp(normal_visual, 0, 1)
            torchvision.utils.save_image(normal_visual, 
                os.path.join(normal_path, '{0:05d}'.format(idx) + ".png"))
            
        
        # albedo_rendered = render_pkg.get("albedomap", None)
        # roughness_rendered = render_pkg.get("roughnessmap", None)

        
        # # Albedo 저장
        # if albedo_rendered is not None:
        #     torchvision.utils.save_image(albedo_rendered, 
        #         os.path.join(albedo_path, '{0:05d}'.format(idx) + ".png"))
        
        # # Roughness 저장
        # if roughness_rendered is not None:
        #     roughness_visual = roughness_rendered.squeeze(0).unsqueeze(0)  # (1,H,W)
        #     torchvision.utils.save_image(roughness_visual, 
        #         os.path.join(roughness_path, '{0:05d}'.format(idx) + ".png"))
            
        # === Planar visualization (Pass1 이미지 위에 표시) ===
        if planar_data is not None:
            planar_indices = planar_data['indices'].cuda()
            plane_normal = planar_data.get('plane_normal', None)
            plane_center = planar_data.get('plane_center', None)
        else:
            planar_indices = render_pkg.get("planar_indices", None)
            plane_normal = render_pkg.get("plane_normal", None)
            plane_center = render_pkg.get("plane_center", None)

        if planar_indices is not None and len(planar_indices) > 0:
                       
            # Pass1 렌더링 이미지를 numpy로 변환
            pass1_image = rendering.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
            pass1_image = (pass1_image * 255).clip(0, 255).astype(np.uint8)
            pass1_bgr = cv2.cvtColor(pass1_image, cv2.COLOR_RGB2BGR)
            
            # Planar Gaussian 3D 위치
            xyz = gaussians.get_xyz[planar_indices]  # (N, 3)
            
            # Homogeneous coordinates
            ones = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
            xyz_h = torch.cat([xyz, ones], dim=1)  # (N, 4)
            
            # World to screen projection
            proj = view.full_proj_transform  # (4, 4)
            xyz_screen = xyz_h @ proj.T  # (N, 4)
            xyz_screen = xyz_screen / (xyz_screen[:, 3:4] + 1e-8)
            
            # NDC to pixel coordinates
            width, height = int(view.image_width), int(view.image_height)
            x_pix = ((xyz_screen[:, 0] + 1.0) * 0.5 * width).cpu().numpy()
            y_pix = ((xyz_screen[:, 1] + 1.0) * 0.5 * height).cpu().numpy()
            
            # Valid points
            valid_mask = (x_pix >= 0) & (x_pix < width) & (y_pix >= 0) & (y_pix < height)
            
            # Convex hull로 planar 영역 표시
            if valid_mask.sum() > 3:
                points = np.column_stack([x_pix[valid_mask], y_pix[valid_mask]]).astype(np.int32)
                hull = cv2.convexHull(points)
                
                # 반투명 노란색 오버레이
                overlay = pass1_bgr.copy()
                cv2.fillPoly(overlay, [hull], (0, 255, 255))  # Yellow
                pass1_bgr = cv2.addWeighted(pass1_bgr, 0.7, overlay, 0.3, 0)
                
                # 빨간 테두리
                cv2.polylines(pass1_bgr, [hull], True, (0, 0, 255), 3)  # Red outline
            
            # Dominant normal 화살표 표시
            if plane_normal is not None and plane_center is not None:
                # Plane center를 화면 좌표로 변환
                center_h = torch.cat([plane_center, torch.ones(1, device=plane_center.device)])
                center_screen = center_h @ proj.T
                center_screen = center_screen / (center_screen[3] + 1e-8)
                
                cx = int((center_screen[0] + 1.0) * 0.5 * width)
                cy = int((center_screen[1] + 1.0) * 0.5 * height)
                
                # Normal 방향 화살표
                normal_world = plane_normal.cpu().numpy()
                arrow_length = 150  # 픽셀 단위
                
                # Normal을 2D로 투영 (x, y 사용)
                nx, ny = normal_world[0], normal_world[1]
                norm = np.sqrt(nx**2 + ny**2) + 1e-8
                nx, ny = nx / norm, ny / norm
                
                ex = int(cx + nx * arrow_length)
                ey = int(cy + ny * arrow_length)
                
                # 두꺼운 마젠타 화살표
                cv2.arrowedLine(pass1_bgr, (cx, cy), (ex, ey), 
                            (255, 0, 255), 4, tipLength=0.25)  # Magenta
                
                # Center point 표시
                cv2.circle(pass1_bgr, (cx, cy), 6, (0, 255, 0), -1)  # Green dot
                
                # Normal 방향 텍스트 표시
                normal_text = f"N:({normal_world[0]:.2f},{normal_world[1]:.2f},{normal_world[2]:.2f})"
                cv2.putText(pass1_bgr, normal_text, (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # 저장
            cv2.imwrite(os.path.join(planar_path, f'{idx:05d}_planar.png'), pass1_bgr)


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, 
                skip_train : bool, skip_test : bool, separate_sh: bool, 
                enable_2pass: bool = False):  #  envmap_path 파라미터 제거
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        #  Environment map 하드코딩
        envlight = None
        if enable_2pass:
            envmap_path = "./env_map/envmap12.exr"  # 하드코딩
            if os.path.exists(envmap_path):
                print(f" Loading environment map: {envmap_path}")
                envlight = EnvLight(path=envmap_path, scale=1.0)
            else:
                print(f"Environment map not found: {envmap_path}, using default")
                envlight = EnvLight(path=None, scale=1.0)

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, 
                       scene.getTrainCameras(), gaussians, pipeline, background, 
                       dataset.train_test_exp, separate_sh, envlight, enable_2pass)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, 
                       scene.getTestCameras(), gaussians, pipeline, background, 
                       dataset.train_test_exp, separate_sh, envlight, enable_2pass)



if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true", default=False)
    parser.add_argument("--skip_test", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--homo-grid", action="store_true", default=False)
    parser.add_argument("--H", nargs=9, type=float, default=None,
                        help="Single 3x3 homography (row-major) given as 9 floats. Required if --homo-grid is enabled.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="렌더 이미지를 저장할 디렉터리. 지정하지 않으면 모델 폴더 하위에 자동 생성.")
    parser.add_argument("--enable-2pass", action="store_true", default=False,
                        help="Enable 2-pass rendering with environment map")
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    #  조건부 검증 추가
    if args.homo_grid and args.H is None:
        parser.error("--homo-grid requires --H to be specified")
    
    # Initialize system state (RNG)
    safe_state(args.quiet)
    pp = pipeline.extract(args)
    pp.fit_mode = "contain"
    pp.homo_grid = getattr(pp, "homo_grid", False) or args.homo_grid
    
    #  homo_grid가 활성화되었을 때만 H 설정
    if args.homo_grid:
        if args.H is not None:
            H_np = np.array(args.H, dtype=np.float32).reshape(3, 3)
            pp.Hmat = torch.from_numpy(H_np).to(device="cuda", dtype=torch.float32)
        else:
            # Identity matrix로 fallback (또는 에러)
            pp.Hmat = torch.eye(3, device="cuda", dtype=torch.float32)
            print("--homo-grid enabled but --H not provided, using identity matrix")
    else:
        pp.Hmat = None  # homo-grid 비활성화 시 None

    pp.output_dir = args.output_dir
    
    render_sets(model.extract(args), args.iteration, pp, args.skip_train, args.skip_test, 
                SPARSE_ADAM_AVAILABLE, args.enable_2pass)