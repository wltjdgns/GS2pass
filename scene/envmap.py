# scene/envmap.py
import torch
import torch.nn.functional as F
import numpy as np


class EnvLight(torch.nn.Module):
    """
    Environment map for lighting (HDR .exr format)
    """
    def __init__(self, path=None, scale=1.0, resolution=512):
        super().__init__()
        self.device = "cuda"
        self.scale = scale
        self.resolution = resolution
        
        # OpenGL to standard coordinate conversion
        self.to_opengl = torch.tensor([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ], dtype=torch.float32, device="cuda")
        
        if path is not None:
            self.envmap = self.load_envmap(path, scale, self.device)
        else:
            # Default: uniform white light
            self.envmap = torch.ones(3, resolution, resolution * 2, 
                                     dtype=torch.float32, device="cuda") * 0.5
        
        self.transform = None
    
    @staticmethod
    def load_envmap(path, scale, device):
        """
        Load HDR environment map from .exr file
        """
        import cv2  # ✅ OpenCV 사용
        
        # Load with OpenCV (supports .exr)
        hdr_image = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        
        if hdr_image is None:
            print(f"⚠️ Failed to load env map: {path}, using uniform light")
            hdr_image = np.ones((512, 1024, 3), dtype=np.float32) * 0.5
        else:
            # OpenCV loads as BGR, convert to RGB
            hdr_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB)
            print(f"✅ Loaded env map: {path}, shape: {hdr_image.shape}")
        
        # Convert to torch tensor
        envmap_torch = torch.from_numpy(hdr_image).to(device).float()
        envmap_torch = envmap_torch.permute(2, 0, 1)  # (3, H, W)
        
        # Apply scale
        envmap_torch = envmap_torch * scale
        
        return envmap_torch
    

    def directlight(self, dirs, transform=None):
        original_shape = dirs.shape
        dirs = dirs.reshape(-1, 3)  # Flatten to (N*num_samples, 3)
        
        # Apply transformation
        if transform is not None:
            dirs = dirs @ transform.T
        elif self.transform is not None:
            dirs = dirs @ self.transform.T
        
        # Convert directions to spherical coordinates
        # theta: azimuth [-π, π], phi: elevation [0, π]
        phi = torch.acos(dirs[:, 2].clamp(-1 + 1e-6, 1 - 1e-6))  # [0, π]
        theta = torch.atan2(dirs[:, 1], dirs[:, 0])  # [-π, π]
        
        # Convert to UV coordinates [0, 1]
        u = (theta / (2 * np.pi) + 0.5)  # [0, 1]
        v = phi / np.pi  # [0, 1]
        
        # Convert to grid_sample coordinates [-1, 1]
        query_x = u * 2 - 1
        query_y = v * 2 - 1
        grid = torch.stack([query_x, query_y], dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)
        
        # Sample from environment map
        envmap = self.envmap.unsqueeze(0)  # (1, 3, H, W)
        light_rgbs = F.grid_sample(
            envmap, 
            grid, 
            mode='bilinear', 
            padding_mode='border', 
            align_corners=True
        )  # Output: (1, 3, 1, N)
        
        # Process output dimensions
        light_rgbs = light_rgbs.squeeze(0).squeeze(1)  # (3, N) - squeeze batch and height
        light_rgbs = light_rgbs.permute(1, 0)  # (N, 3)
        
        # Reshape back to original
        light_rgbs = light_rgbs.reshape(*original_shape[:-1], 3)
        
        return light_rgbs.clamp(min=0)

    
    def get_env(self):
        """Return the full environment map"""
        return self.envmap.permute(1, 2, 0)  # (H, W, 3)
