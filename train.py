
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import torch
import torch.nn.functional as F

from random import randint
import numpy as np
#  수정 1: Import 추가
from utils.loss_utils import (
    l1_loss, ssim, 
    compute_pseudo_normal, normal_loss, depth_uncertainty_loss,
    smoothness_loss_with_edge_aware, smoothness_loss_normal  # 추가
)
from gaussian_renderer import render, render_2pass, EnvLight
from gaussian_renderer import network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


# === [ADD] GT 워프 유틸 (Π → Π′, grid_sample 역매핑) ===
def warp_image_with_H_torch(img_BCHW, H_3x3, mode="bilinear"):
    """
    img_BCHW: (B,C,H,W)
    H_3x3: (B,3,3) homography (Π -> Π′). grid_sample은 역매핑 필요하므로 H^{-1} 사용.
    """
    B, C, H, W = img_BCHW.shape
    device = img_BCHW.device
    dtype  = img_BCHW.dtype

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device, dtype=dtype),
        torch.linspace(-1, 1, W, device=device, dtype=dtype),
        indexing="ij"
    )
    x = (xx + 1) * (W - 1) / 2.0
    y = (yy + 1) * (H - 1) / 2.0
    ones = torch.ones_like(x)
    grid = torch.stack([x, y, ones], dim=-1).view(1, H, W, 3).repeat(B,1,1,1)

    Hinv = torch.linalg.inv(H_3x3)
    src = grid @ Hinv.transpose(1,2)
    w = src[...,2].clamp(min=1e-8)
    xs = src[...,0] / w
    ys = src[...,1] / w
    xs_n = xs / (W - 1) * 2 - 1
    ys_n = ys / (H - 1) * 2 - 1
    samp_grid = torch.stack([xs_n, ys_n], dim=-1)

    return torch.nn.functional.grid_sample(
        img_BCHW, samp_grid, mode=mode, align_corners=True, padding_mode="border"
    )


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    #  추가: Planar cache 저장용 딕셔너리
    planar_cache_per_view = {}

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)

    # ===== Multiple Env Maps 로딩 =====
    envlights = []
    ENVMAP_DIR = "./env_map"
    
    if os.path.exists(ENVMAP_DIR):
        print(f"Loading environment maps from: {ENVMAP_DIR}")
        exr_files = sorted([f for f in os.listdir(ENVMAP_DIR) if f.endswith('.exr')])
        
        for exr_file in exr_files:
            exr_path = os.path.join(ENVMAP_DIR, exr_file)
            envlights.append(EnvLight(path=exr_path, scale=1.0))
            print(f"   Loaded: {exr_file}")
        
        print(f" Total {len(envlights)} environment maps loaded")
    
    if len(envlights) == 0:
        print("⚠️ No .exr files found, using uniform lighting")
        envlights.append(EnvLight(path=None, scale=0.5))

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # [ADD] 전역 homo_grid/Hmat을 pipeline에 주입
    pipe.homo_grid = bool(getattr(pipe, "homo_grid", False) or args.homo_grid)

    if args.H is not None:
        H_np = np.array(args.H, dtype=np.float32).reshape(3, 3)
        pipe.Hmat = torch.tensor(H_np, dtype=torch.float32, device="cuda")
    else:
        pipe.Hmat = None

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)
    
    #  수정 2: Stage 정보 출력
    print(f"[Config] Total iterations: {opt.iterations}")
    print(f"[Config] Stage 1: 0 ~ {opt.stage1_iter} iter (Base training)")
    print(f"[Config] Stage 2: {opt.stage1_iter} ~ {opt.iterations} iter (BRDF + 2-pass)")
    print(f"[Config] depth_loss_mode: {opt.depth_loss_mode}")
    
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        #  수정 3: 2-Pass Rendering 조건 변경
        enable_2pass = iteration > opt.stage1_iter  # Stage 2부터 2-pass
    
        if enable_2pass and len(envlights) > 0:
            current_envlight = envlights[randint(0, len(envlights) - 1)]
            
            render_pkg = render_2pass(
                viewpoint_cam, gaussians, pipe, bg,
                render_func = render,
                envlight=current_envlight,
                lambda_weight=opt.lambda_reflection,
                enable_2pass=True,
                iteration=iteration
            )

            #  추가: Planar detection 결과 저장
            planar_indices = render_pkg.get("planar_indices", None)
            plane_normal = render_pkg.get("plane_normal", None)
            plane_center = render_pkg.get("plane_center", None)
            
            # View별로 저장 (마지막 detection 결과만 유지)
            if planar_indices is not None and len(planar_indices) > 0:
                view_key = viewpoint_cam.image_name  # "00001.png" 같은 이름
                planar_cache_per_view[view_key] = {
                    'indices': planar_indices.cpu(),  # GPU -> CPU
                    'normal': plane_normal.cpu() if plane_normal is not None else None,
                    'center': plane_center.cpu() if plane_center is not None else None
                }

                
        else:
            # Standard rendering (Stage 1)
            render_pkg = render(
                viewpoint_cam, 
                gaussians, 
                pipe, 
                bg,
                use_trained_exp=dataset.train_test_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE
            )

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        if args.homo_grid and args.warp_gt:
            from gaussian_renderer import _fit_normalize_H_to_viewport
            
            assert pipe.Hmat is not None, "pipe.Hmat is None (check --H input)"
            H_b = pipe.Hmat.clone().unsqueeze(0).to("cuda")

            H_b[0] = _fit_normalize_H_to_viewport(
                H_b[0],
                int(viewpoint_cam.image_width),
                int(viewpoint_cam.image_height),
                mode="contain"
            )

            gt_image_b = gt_image.unsqueeze(0) if gt_image.dim() == 3 else gt_image
            with torch.no_grad():
                gt_image_prime = warp_image_with_H_torch(gt_image_b, H_b, mode="bilinear")
            gt_image = gt_image_prime.squeeze(0)

        #  수정 4: Loss 계산 - 개별 loss 추적
        # ===== RGB Loss (Both Stages) =====
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = opt.lambda_l1 * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        
        # 개별 loss 추적을 위한 dict
        loss_dict = {
            'L_n': 0.0,
            'L_s_n': 0.0,
            'L_s_b': 0.0,
            'L_s_r': 0.0
        }
        

        if iteration <= opt.stage1_iter:
            depth_rendered = render_pkg.get("depth", None)
            normal_rendered = render_pkg.get("normal", None)
            
            if depth_rendered is not None and normal_rendered is not None:
                # pseudo normal 계산 (view space 깊이 사용)
                depth_for_pseudo = depth_rendered.permute(1, 2, 0)  # (H, W, 1)
                pseudo_normal = compute_pseudo_normal(depth_for_pseudo, scale_factor=20.0)
                
                # view space로 맞추기 위해 정규화
                normal_for_loss = F.normalize(normal_rendered.permute(1, 2, 0), dim=-1)  # (H, W, 3)
                pseudo_normal = F.normalize(pseudo_normal, dim=-1)
                
                # normal consistency loss 계산
                l_normal = normal_loss(normal_for_loss, pseudo_normal)
                loss += opt.lambda_normal * l_normal
                loss_dict['L_n'] = l_normal.item()

                # 기존 edge-aware normal smoothness loss 유지
                loss_smooth_n = smoothness_loss_normal(normal_rendered, gt_image)
                loss += opt.lambda_normal_smooth * loss_smooth_n
                loss_dict['L_s_n'] = loss_smooth_n.item()

        
        else:
            # ===== Stage 2: BRDF decomposition with smoothness =====
            # L_{s,b}: Albedo smoothness (edge-aware)
            if "albedomap" in render_pkg:
                albedomap = render_pkg["albedomap"]
                loss_smooth_b = smoothness_loss_normal(albedomap, gt_image)
                loss += opt.lambda_albedo_smooth * loss_smooth_b
                loss_dict['L_s_b'] = loss_smooth_b.item()
            
            # L_{s,r}: Roughness smoothness (edge-aware)
            if "roughnessmap" in render_pkg:
                roughnessmap = render_pkg["roughnessmap"]
                loss_smooth_r = smoothness_loss_with_edge_aware(roughnessmap, gt_image)
                loss += opt.lambda_roughness_smooth * loss_smooth_r
                loss_dict['L_s_r'] = loss_smooth_r.item()

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # EMA 추적 변수 초기화
            if iteration == 1:
                ema_loss_dict = {k: 0.0 for k in loss_dict.keys()}
            
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            
            # 각 loss별 EMA 업데이트
            for key in loss_dict.keys():
                if key not in ema_loss_dict:
                    ema_loss_dict[key] = 0.0
                ema_loss_dict[key] = 0.4 * loss_dict[key] + 0.6 * ema_loss_dict[key]

            #  수정 5: Progress bar - 개별 loss 표시
            if iteration % 10 == 0:
                stage_name = "S1" if iteration <= opt.stage1_iter else "S2"
                
                if iteration <= opt.stage1_iter:
                    # Stage 1: L_n, L_u, L_s_n 표시
                    progress_bar.set_postfix({
                        "Stage": stage_name,
                        "Loss": f"{ema_loss_for_log:.5f}",
                        "L_n": f"{ema_loss_dict['L_n']:.5f}",
                        "Ls_n": f"{ema_loss_dict['L_s_n']:.5f}"
                    })
                else:
                    # Stage 2: L_s_b, L_s_r 표시
                    progress_bar.set_postfix({
                        "Stage": stage_name,
                        "Loss": f"{ema_loss_for_log:.5f}",
                        "Ls_b": f"{ema_loss_dict['L_s_b']:.5f}",
                        "Ls_r": f"{ema_loss_dict['L_s_r']:.5f}"
                    })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            #  수정: Checkpoint 저장 시 planar cache도 포함
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint + Planar Cache".format(iteration))
                torch.save({
                    'gaussians': gaussians.capture(),
                    'iteration': iteration,
                    'planar_cache': planar_cache_per_view  # Planar cache 추가
                }, scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
            #  추가: PLY 저장 시점에 planar cache도 별도 저장
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians (PLY) + Planar Cache".format(iteration))
                scene.save(iteration)
                
                # Planar cache 별도 저장
                if len(planar_cache_per_view) > 0:
                    planar_cache_path = os.path.join(scene.model_path, 
                                                    f"planar_cache_{iteration}.pth")
                    torch.save(planar_cache_per_view, planar_cache_path)
                    print(f"   Saved planar cache: {len(planar_cache_per_view)} views")
                    

def prepare_output_and_logger(args):    
    """
    --model-path: 직접 경로 지정 (최우선)
    --output-dir: 저장 폴더 지정 (랜덤 ID 없이 바로 사용)
    둘 다 없으면: ./output/ 사용
    """
    if args.model_path and args.model_path != "":
        # --model-path 명시적 지정: 그대로 사용
        final_path = args.model_path
    else:
        # --output-dir 사용 (랜덤 ID 없음)
        base_dir = getattr(args, 'output_dir', './output/')
        if not base_dir or base_dir == "":
            base_dir = './output/'
        final_path = base_dir  # ← 랜덤 ID 제거!
    
    args.model_path = final_path
    print(f"[Output] Training results will be saved to: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer



def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=True)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--homo-grid", action="store_true", default=False)
    parser.add_argument("--H", nargs=9, type=float, default=None)
    parser.add_argument("--warp-gt", action="store_true", default=True)
    
    # ===== output-dir 추가 =====
    parser.add_argument("--output-dir", type=str, default=None, dest="output_dir",
                        help="Directory for saving outputs (overrides --model-path)")
    
    args = parser.parse_args(sys.argv[1:])
    
    # ===== 우선순위 처리 =====
    if hasattr(args, 'output_dir') and args.output_dir:
        args.model_path = args.output_dir
        print(f"[Mode] Using --output-dir: {args.model_path}")
    elif not args.model_path or args.model_path.strip() == "":
        args.model_path = "./output/"
        print(f"[Mode] Using default: {args.model_path}")
    else:
        print(f"[Mode] Using --model-path: {args.model_path}")
    # =========================
    
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    pp.fit_mode = "contain"
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, 
             args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    print("\nTraining complete.")
