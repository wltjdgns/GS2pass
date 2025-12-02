from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        # [ADD] homography-warped grid 옵션
        self.homo_grid = False
        self.Hmat = None
        self.Hmat_file = ""
        super().__init__(parser, "Pipeline Parameters")

    def _init(self, parser: ArgumentParser):
        group = parser.add_argument_group("Pipeline Parameters")
        group.add_argument("--convert_SHs_python", action="store_true")
        group.add_argument("--compute_cov3D_python", action="store_true")
        group.add_argument("--debug", action="store_true")
        group.add_argument("--antialiasing", action="store_true")
        # [ADD] 인자 바인딩
        group.add_argument("--homo-grid", action="store_true", dest="homo_grid", default=False,
                           help="Warp rasterization domain to Π′ by homography H")
        group.add_argument("--H", type=float, nargs=9, default=None,
                           help="Row-major 3x3 homography for Π→Π′")
        group.add_argument("--H_file", type=str, default="",
                           help="Path to per-frame H (npz/json/pt). If given, overrides --H.")
        group.add_argument("--normal-mode", choices=["off","gaussian"], default="gaussian",
                           help="off: 기본, gaussian: 가우시안 단위 world-space normal 파라미터 최적화")
        group.add_argument('--depth_loss_mode', choices=['supervised','unsupervised'], default='unsupervised', 
                          help='Depth supervision mode: supervised/unsupervised')
        
        return group


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        # ===== Basic Training Parameters =====
        self.iterations = 30_000  #  유지
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        
        # ===== RGB Loss Weights =====
        self.lambda_l1 = 0.8           #  추가: L1 loss weight
        self.lambda_dssim = 0.2        #  유지: SSIM loss weight
        
        # ===== Stage Configuration =====
        self.stage1_iter = 15000       #  추가: Stage 1 끝 / Stage 2 시작
        self.enable_2pass_iter = 15000 #  변경: 15000 → 10000
        self.lambda_reflection = 0.3   #  유지: Reflection blending weight
        
        # ===== Stage 1 Loss Weights (0~10k) =====
        self.lambda_normal = 0.3        #  변경: 0.005 → 0.01 (L_n: Normal consistency)
        self.lambda_normal_smooth = 0.01 #  추가: L_{s,n} (Normal smoothness)
        
        # ===== Stage 2 Loss Weights (10k~30k) =====
        self.lambda_albedo_smooth = 0.01    #  추가: L_{s,b} (Albedo smoothness)
        self.lambda_roughness_smooth = 0.01 #  유지: L_{s,r} (Roughness smoothness)
        
        # ===== Densification Parameters =====
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        
        # ===== Misc Parameters =====
        self.random_background = False
        self.optimizer_type = "default"
        self.depth_loss_mode = 'unsupervised'
        self.normal_lr = 0.001
        self.depth_lr = 0.001
        self.depth_l1_weight_init = 0.0001
        self.depth_l1_weight_final = 0.001
        
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)