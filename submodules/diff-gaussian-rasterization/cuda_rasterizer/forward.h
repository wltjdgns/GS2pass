#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
    // Perform initial steps for each Gaussian prior to rasterization.
    void preprocess(int P, int D, int M,
        const float* orig_points,
        const glm::vec3* scales,
        const float scale_modifier,
        const glm::vec4* rotations,
        const float* opacities,
        const float* shs,
        bool* clamped,
        const float* cov3D_precomp,
        const float* colors_precomp,
        const float* viewmatrix,
        const float* projmatrix,
        const glm::vec3* cam_pos,
        const int W, int H,
        const float focal_x, float focal_y,
        const float tan_fovx, float tan_fovy,
        int* radii,
        float2* points_xy_image,
        float* depths,
        float* cov3Ds,
        float* colors,
        float4* conic_opacity,
        const dim3 grid,
        uint32_t* tiles_touched,
        bool prefiltered,
        bool antialiasing,
        bool homo_grid,
        const float* Hmat);
        
    // Main rasterization method with depth/normal/basecolor/roughness
    void render(
        const dim3 grid, dim3 block,
        const uint2* ranges,
        const uint32_t* point_list,
        int W, int H,
        const float2* points_xy_image,
        const float* features,
        const float4* conic_opacity,
        float* final_T,
        uint32_t* n_contrib,
        const float* bg_color,
        float* out_color,
        float* depths,
        float* depth,
        
        // Depth/Normal parameters
        const float* normals,           // (N×3)
        float* out_depth,               // (H×W)
        float* out_normal,              // (H×W×3)
        
        // === 추가: Basecolor/Roughness parameters ===
        const float* basecolors,        // (N×3) 각 Gaussian의 basecolor
        const float* roughness,         // (N×1) 각 Gaussian의 roughness
        float* out_basecolor,           // (H×W×3) 렌더링된 basecolor
        float* out_roughness);          // (H×W) 렌더링된 roughness
}

#endif
