#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

// =========================================
// FORWARD PASS - 11개 요소 반환 ✅
// =========================================
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
           torch::Tensor, torch::Tensor>  // 11개 ✅
RasterizeGaussiansCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx, 
    const float tan_fovy,
    const int image_height,
    const int image_width,
    const torch::Tensor& sh,
    const int degree,
    const torch::Tensor& campos,
    const bool prefiltered,
    const bool antialiasing,
    const bool debug,
    const bool homo_grid,
    const torch::Tensor& Hmat,
    const torch::Tensor& depths,
    const torch::Tensor& normals,
    // ★ 추가: Basecolor/Roughness
    const torch::Tensor& basecolors,
    const torch::Tensor& roughness);

// =========================================
// BACKWARD PASS - 12개 요소 반환 ✅
// =========================================
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>  // 12개 ✅
RasterizeGaussiansBackwardCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& radii,
    const torch::Tensor& colors,
    const torch::Tensor& opacities,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const torch::Tensor& dL_dout_color,
    const torch::Tensor& dL_dout_invdepth,
    const torch::Tensor& dL_dout_depth,
    const torch::Tensor& dL_dout_normal,
    const torch::Tensor& sh,
    const int degree,
    const torch::Tensor& campos,
    const torch::Tensor& geomBuffer,
    const int R,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer,
    const torch::Tensor& depths,
    const torch::Tensor& normals,
    // ★ 추가: Basecolor/Roughness 입력
    const torch::Tensor& basecolors,
    const torch::Tensor& roughness,
    // ★ 추가: Basecolor/Roughness loss gradients
    const torch::Tensor& dL_dout_basecolor,
    const torch::Tensor& dL_dout_roughness,
    const bool antialiasing,
    const bool debug);
        
torch::Tensor markVisible(
        torch::Tensor& means3D,
        torch::Tensor& viewmatrix,
        torch::Tensor& projmatrix);
