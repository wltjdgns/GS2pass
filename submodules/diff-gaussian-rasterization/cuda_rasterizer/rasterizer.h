#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
    class Rasterizer
    {
    public:

        static void markVisible(
            int P,
            float* means3D,
            float* viewmatrix,
            float* projmatrix,
            bool* present);

        // Forward 함수: depth/normal/basecolor/roughness 출력
        static int forward(
            std::function<char* (size_t)> geometryBuffer,
            std::function<char* (size_t)> binningBuffer,
            std::function<char* (size_t)> imageBuffer,
            const int P, int D, int M,
            const float* background,
            const int width, int height,
            const float* means3D,
            const float* shs,
            const float* colors_precomp,
            const float* opacities,
            const float* scales,
            const float scale_modifier,
            const float* rotations,
            const float* cov3D_precomp,
            const float* viewmatrix,
            const float* projmatrix,
            const float* cam_pos,
            const float tan_fovx, float tan_fovy,
            const bool prefiltered,
            float* out_color,
            float* depth,
            bool antialiasing,
            int* radii = nullptr,
            bool debug = false,
            bool homo_grid = false,
            const float* Hmat = nullptr,
            
            // Depth/Normal parameters
            const float* depths = nullptr,      // (N×1)
            const float* normals = nullptr,     // (N×3)
            float* out_depth = nullptr,         // (H×W)
            float* out_normal = nullptr,        // (H×W×3)
            
            // === 추가: Basecolor/Roughness parameters ===
            const float* basecolors = nullptr,  // (N×3)
            const float* roughness = nullptr,   // (N×1)
            float* out_basecolor = nullptr,     // (H×W×3)
            float* out_roughness = nullptr);    // (H×W)

        // Backward 함수: depth/normal/basecolor/roughness gradient
        static void backward(
            const int P, int D, int M, int R,
            const float* background,
            const int width, int height,
            const float* means3D,
            const float* shs,
            const float* colors_precomp,
            const float* opacities,
            const float* scales,
            const float scale_modifier,
            const float* rotations,
            const float* cov3D_precomp,
            const float* viewmatrix,
            const float* projmatrix,
            const float* campos,
            const float tan_fovx, float tan_fovy,
            const int* radii,
            char* geom_buffer,
            char* binning_buffer,
            char* image_buffer,
            const float* dL_dpix,
            const float* dL_invdepths,
            const float* dL_depths,
            const float* dL_normals,
            const float* dL_basecolors,
            const float* dL_roughness,
            float* dL_dmean2D,
            float* dL_dconic,
            float* dL_dopacity,
            float* dL_dcolor,
            float* dL_dinvdepth,
            float* dL_ddepths,             // ★ 언더스코어 1개
            float* dL_dnormals,            // ★ 언더스코어 1개
            float* dL_dbasecolors,         // ★ 언더스코어 1개
            float* dL_droughness,          // ★ 언더스코어 1개
            float* dL_dmean3D,
            float* dL_dcov3D,
            float* dL_dsh,
            float* dL_dscale,
            float* dL_drot,
            const float* depths_in = nullptr,
            const float* normals_in = nullptr,
            const float* basecolors_in = nullptr,
            const float* roughness_in = nullptr,
            bool antialiasing = true,
            bool debug = false);
    };
};

#endif
