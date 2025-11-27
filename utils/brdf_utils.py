# utils/brdf_utils.py
import torch
import torch.nn.functional as F
import numpy as np


def GGX_specular(normal, view_dir, light_dir, roughness, f0_value=0.04):
    """
    Cook-Torrance GGX BRDF for specular reflection
    
    Args:
        normal: (N, 3) - surface normals
        view_dir: (N, 3) - view directions (camera to point)
        light_dir: (N, num_samples, 3) - light directions
        roughness: (N, 1) - roughness [0, 1]
        f0_value: float or (N, 3) - base reflectivity
    
    Returns:
        specular: (N, num_samples, 3) - specular BRDF values
    """
    # Normalize inputs
    N = F.normalize(normal, dim=-1)  # (N, 3)
    V = F.normalize(view_dir, dim=-1)  # (N, 3)
    L = F.normalize(light_dir, dim=-1)  # (N, num_samples, 3)
    
    # Half vector
    H = F.normalize(L + V[:, None, :], dim=-1)  # (N, num_samples, 3)
    
    # Dot products (clamp to avoid NaN)
    NoV = torch.sum(V * N, dim=-1, keepdim=True).clamp(1e-6, 1)  # (N, 1)
    NoL = torch.sum(N[:, None] * L, dim=-1, keepdim=True).clamp(1e-6, 1)  # (N, num_samples, 1)
    NoH = torch.sum(N[:, None] * H, dim=-1, keepdim=True).clamp(1e-6, 1)  # (N, num_samples, 1)
    VoH = torch.sum(V[:, None] * H, dim=-1, keepdim=True).clamp(1e-6, 1)  # (N, num_samples, 1)
    
    # Alpha (roughness^2)
    alpha = roughness * roughness  # (N, 1)
    alpha2 = alpha * alpha
    
    # === D: GGX Normal Distribution ===
    denom = (NoH * NoH * (alpha2[:, None, :] - 1) + 1)
    D = alpha2[:, None, :] / (np.pi * denom * denom + 1e-7)  # (N, num_samples, 1)
    
    # === F: Fresnel (Schlick approximation) ===
    # Optimized Schlick from r3dg
    FMi = (-5.55473 * VoH - 6.98316) * VoH
    F_value = f0_value + (1 - f0_value) * torch.pow(2.0, FMi)  # (N, num_samples, 1 or 3)
    
    # === G: Geometry (Smith GGX) ===
    k = alpha / 2  # (N, 1)
    G1_V = NoV / (NoV * (1 - k) + k + 1e-7)  # (N, 1)
    G1_L = NoL / (NoL * (1 - k[:, None, :]) + k[:, None, :] + 1e-7)  # (N, num_samples, 1)
    G = G1_V[:, None, :] * G1_L  # (N, num_samples, 1)
    
    # === Cook-Torrance BRDF ===
    numerator = D * F_value * G
    denominator = 4  * NoV[:, None, :] * NoL + 1e-7
    specular = numerator / denominator  # (N, num_samples, 1 or 3)
    
    return specular.clamp(min=0)


def compute_reflection_direction(normal, view_dir):
    """
    Compute perfect reflection direction
    
    Args:
        normal: (N, 3) - surface normals
        view_dir: (N, 3) - view directions (camera to point)
    
    Returns:
        reflect_dir: (N, 3) - reflection directions
    """
    normal = F.normalize(normal, dim=-1)
    view_dir = F.normalize(view_dir, dim=-1)
    
    # Reflection: R = V - 2(V·N)N
    dot_prod = torch.sum(view_dir * normal, dim=-1, keepdim=True)
    reflect_dir = view_dir - 2 * dot_prod * normal
    
    return F.normalize(reflect_dir, dim=-1)


def sample_specular_lobe(reflect_dir, roughness, num_samples=16):
    """
    Sample directions around reflection direction based on roughness
    (Simplified importance sampling for GGX)
    
    Args:
        reflect_dir: (N, 3) - perfect reflection directions
        roughness: (N, 1) - roughness [0, 1]
        num_samples: int - number of samples
    
    Returns:
        sample_dirs: (N, num_samples, 3) - sampled directions
    """
    N = reflect_dir.shape[0]
    device = reflect_dir.device
    
    if roughness.mean() < 0.01:  # Mirror-like
        # Just return perfect reflection
        return reflect_dir.unsqueeze(1).expand(-1, num_samples, -1)
    
    # Random spherical coordinates
    u1 = torch.rand(N, num_samples, device=device)
    u2 = torch.rand(N, num_samples, device=device)
    
    # GGX importance sampling
    alpha = roughness * roughness
    cos_theta = torch.sqrt((1 - u1) / (1 + (alpha * alpha - 1) * u1 + 1e-7))
    sin_theta = torch.sqrt(1 - cos_theta * cos_theta + 1e-7)
    phi = 2 * np.pi * u2
    
    # Local coordinates (around reflection direction)
    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    z = cos_theta
    
    # Build tangent frame
    up = torch.where(
        torch.abs(reflect_dir[:, 2:3]) < 0.999,
        torch.tensor([0., 0., 1.], device=device).expand(N, 3),
        torch.tensor([1., 0., 0.], device=device).expand(N, 3)
    )
    tangent = F.normalize(torch.cross(up, reflect_dir, dim=-1), dim=-1)
    bitangent = F.normalize(torch.cross(reflect_dir, tangent, dim=-1), dim=-1)
    
    # Transform to world space
    sample_dirs = (
        x[:, :, None] * tangent[:, None, :] +
        y[:, :, None] * bitangent[:, None, :] +
        z[:, :, None] * reflect_dir[:, None, :]
    )
    
    return F.normalize(sample_dirs, dim=-1)


def schlick_fresnel(cos_theta, f0_value=0.04):
    """
    Schlick approximation for Fresnel term
    
    Args:
        cos_theta: (N,) or (N, 1) - cosine of angle
        f0_value: float - base reflectivity
    
    Returns:
        F: (N, 1) - Fresnel term
    """
    return f0_value + (1 - f0_value) * torch.pow(1 - cos_theta.clamp(0, 1), 5)
