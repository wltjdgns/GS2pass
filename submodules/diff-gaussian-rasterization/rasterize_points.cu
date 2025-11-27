#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
        return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

// ★ 수정: 11개 요소로 변경
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
           torch::Tensor, torch::Tensor>  // 11개
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
    const torch::Tensor& basecolors,
    const torch::Tensor& roughness)
{
    // Validation code (동일)
    TORCH_CHECK(Hmat.ndimension()==2 && Hmat.size(0)==3 && Hmat.size(1)==3,
                "H must be (3,3)");
    TORCH_CHECK(Hmat.is_cuda(), "H must be CUDA tensor");
    TORCH_CHECK(Hmat.dtype()==torch::kFloat32, "H must be float32");

    TORCH_CHECK(depths.ndimension()==2 && depths.size(1)==1,
                "depths must have dimensions (num_points, 1)");
    TORCH_CHECK(depths.is_cuda(), "depths must be CUDA tensor");
    TORCH_CHECK(depths.dtype()==torch::kFloat32, "depths must be float32");
    
    TORCH_CHECK(normals.ndimension()==2 && normals.size(1)==3,
                "normals must have dimensions (num_points, 3)");
    TORCH_CHECK(normals.is_cuda(), "normals must be CUDA tensor");
    TORCH_CHECK(normals.dtype()==torch::kFloat32, "normals must be float32");

    TORCH_CHECK(basecolors.ndimension()==2 && basecolors.size(1)==3,
                "basecolors must have dimensions (num_points, 3)");
    TORCH_CHECK(basecolors.is_cuda(), "basecolors must be CUDA tensor");
    TORCH_CHECK(basecolors.dtype()==torch::kFloat32, "basecolors must be float32");

    TORCH_CHECK(roughness.ndimension()==2 && roughness.size(1)==1,
                "roughness must have dimensions (num_points, 1)");
    TORCH_CHECK(roughness.is_cuda(), "roughness must be CUDA tensor");
    TORCH_CHECK(roughness.dtype()==torch::kFloat32, "roughness must be float32");

    if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
        AT_ERROR("means3D must have dimensions (num_points, 3)");
    }
  
    const int P = means3D.size(0);
    const int H = image_height;
    const int W = image_width;

    TORCH_CHECK(depths.size(0) == P, "depths must have same num_points as means3D");
    TORCH_CHECK(normals.size(0) == P, "normals must have same num_points as means3D");
    TORCH_CHECK(basecolors.size(0) == P, "basecolors must have same num_points as means3D");
    TORCH_CHECK(roughness.size(0) == P, "roughness must have same num_points as means3D");

    auto int_opts = means3D.options().dtype(torch::kInt32);
    auto float_opts = means3D.options().dtype(torch::kFloat32);

    torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
    torch::Tensor out_invdepth = torch::full({1, H, W}, 0.0, float_opts).contiguous();
    float* out_invdepthptr = out_invdepth.data<float>();

    torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts).contiguous();
    torch::Tensor out_normal = torch::full({3, H, W}, 0.0, float_opts).contiguous();
    float* out_depthptr = out_depth.data<float>();
    float* out_normalptr = out_normal.data<float>();

    torch::Tensor out_basecolor = torch::full({3, H, W}, 0.0, float_opts).contiguous();
    torch::Tensor out_roughness = torch::full({1, H, W}, 0.0, float_opts).contiguous();
    float* out_basecolorptr = out_basecolor.data<float>();
    float* out_roughnessptr = out_roughness.data<float>();

    torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));

    torch::Device device(torch::kCUDA);
    torch::TensorOptions options(torch::kByte);
    torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
    torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
    torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
    std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
    std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
    std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

    int rendered = 0;
    if(P != 0)
    {
        int M = 0;
        if(sh.size(0) != 0)
        {
            M = sh.size(1);
        }
        
        const float* Hptr = nullptr;
        if (homo_grid) {
            Hptr = Hmat.contiguous().data_ptr<float>();
        }
        
        rendered = CudaRasterizer::Rasterizer::forward(
            geomFunc,
            binningFunc,
            imgFunc,
            P, degree, M,
            background.contiguous().data<float>(),
            W, H,
            means3D.contiguous().data<float>(),
            sh.contiguous().data_ptr<float>(),
            colors.contiguous().data<float>(), 
            opacity.contiguous().data<float>(), 
            scales.contiguous().data_ptr<float>(),
            scale_modifier,
            rotations.contiguous().data_ptr<float>(),
            cov3D_precomp.contiguous().data<float>(), 
            viewmatrix.contiguous().data<float>(), 
            projmatrix.contiguous().data<float>(),
            campos.contiguous().data<float>(),
            tan_fovx,
            tan_fovy,
            prefiltered,
            out_color.contiguous().data<float>(),
            out_invdepthptr,
            antialiasing,
            radii.contiguous().data<int>(),
            debug,
            homo_grid,
            Hptr,
            depths.contiguous().data<float>(),
            normals.contiguous().data<float>(),
            out_depthptr,
            out_normalptr,
            basecolors.contiguous().data<float>(),
            roughness.contiguous().data<float>(),
            out_basecolorptr,
            out_roughnessptr
        );
    }
    
    return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, 
                           imgBuffer, out_invdepth, out_depth, out_normal, 
                           out_basecolor, out_roughness);  // 11개 ✅
}

// ★ 수정: 12개 요소로 변경
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, 
           torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>  // 12개
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
    const torch::Tensor& basecolors,
    const torch::Tensor& roughness,
    const torch::Tensor& dL_dout_basecolor,
    const torch::Tensor& dL_dout_roughness,
    const bool antialiasing,
    const bool debug)
{
    const int P = means3D.size(0);
    const int H = dL_dout_color.size(1);
    const int W = dL_dout_color.size(2);
    
    int M = 0;
    if(sh.size(0) != 0)
    { 
        M = sh.size(1);
    }

    torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
    torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
    torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
    torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
    torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
    torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
    torch::Tensor dL_dinvdepths = torch::zeros({P, 1}, means3D.options());

    float* dL_dinvdepthsptr = nullptr;
    float* dL_dout_invdepthptr = nullptr;

    if(dL_dout_invdepth.size(0) != 0)
    {
        dL_dinvdepthsptr = dL_dinvdepths.data<float>();
        dL_dout_invdepthptr = dL_dout_invdepth.data<float>();
    }

    torch::Tensor dL_ddepths = torch::zeros({P, 1}, means3D.options());
    torch::Tensor dL_dnormals = torch::zeros({P, 3}, means3D.options());
    float* dL_ddepthsptr = dL_ddepths.data<float>();
    float* dL_dnormalsptr = dL_dnormals.data<float>();

    torch::Tensor dL_dbasecolors = torch::zeros({P, 3}, means3D.options());
    torch::Tensor dL_droughness = torch::zeros({P, 1}, means3D.options());
    float* dL_dbasecolorsptr = dL_dbasecolors.data<float>();
    float* dL_droughnessptr = dL_droughness.data<float>();

    float* dL_dout_basecolorptr = nullptr;
    float* dL_dout_roughnessptr = nullptr;
    if (dL_dout_basecolor.size(0) != 0) {
        dL_dout_basecolorptr = dL_dout_basecolor.data<float>();
    }
    if (dL_dout_roughness.size(0) != 0) {
        dL_dout_roughnessptr = dL_dout_roughness.data<float>();
    }
    
    float* dL_dout_depthptr = nullptr;
    float* dL_dout_normalptr = nullptr;
    if (dL_dout_depth.size(0) != 0) {
        dL_dout_depthptr = dL_dout_depth.data<float>();
    }
    if (dL_dout_normal.size(0) != 0) {
        dL_dout_normalptr = dL_dout_normal.data<float>();
    }

    if(P != 0)
    {  
        // ★ 수정: 파라미터 순서 수정
        CudaRasterizer::Rasterizer::backward(
            P, degree, M, R,
            background.contiguous().data<float>(),
            W, H, 
            means3D.contiguous().data<float>(),
            sh.contiguous().data<float>(),
            colors.contiguous().data<float>(),
            opacities.contiguous().data<float>(),
            scales.data_ptr<float>(),
            scale_modifier,
            rotations.data_ptr<float>(),
            cov3D_precomp.contiguous().data<float>(),
            viewmatrix.contiguous().data<float>(),
            projmatrix.contiguous().data<float>(),
            campos.contiguous().data<float>(),
            tan_fovx,
            tan_fovy,
            radii.contiguous().data<int>(),
            reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
            reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
            dL_dout_color.contiguous().data<float>(),
            dL_dout_invdepthptr,
            dL_dout_depthptr,
            dL_dout_normalptr,
            dL_dout_basecolorptr,          // ★ 여기로 이동
            dL_dout_roughnessptr,          // ★ 여기로 이동
            dL_dmeans2D.contiguous().data<float>(),
            dL_dconic.contiguous().data<float>(),  
            dL_dopacity.contiguous().data<float>(),
            dL_dcolors.contiguous().data<float>(),
            dL_dinvdepthsptr,
            dL_ddepthsptr,                 // ★ 변수명 수정
            dL_dnormalsptr,                // ★ 변수명 수정
            dL_dbasecolorsptr,             // ★ 변수명 수정
            dL_droughnessptr,              // ★ 변수명 수정
            dL_dmeans3D.contiguous().data<float>(),
            dL_dcov3D.contiguous().data<float>(),
            dL_dsh.contiguous().data<float>(),
            dL_dscales.contiguous().data<float>(),
            dL_drotations.contiguous().data<float>(),
            depths.contiguous().data<float>(),
            normals.contiguous().data<float>(),
            basecolors.contiguous().data<float>(),
            roughness.contiguous().data<float>(),
            antialiasing,
            debug);
    }

    return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, 
                           dL_dcov3D, dL_dsh, dL_dscales, dL_drotations, 
                           dL_ddepths, dL_dnormals, dL_dbasecolors, 
                           dL_droughness);  // 12개 ✅
}

torch::Tensor markVisible(
        torch::Tensor& means3D,
        torch::Tensor& viewmatrix,
        torch::Tensor& projmatrix)
{ 
    const int P = means3D.size(0);
    torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
    if(P != 0)
    {
        CudaRasterizer::Rasterizer::markVisible(P,
            means3D.contiguous().data<float>(),
            viewmatrix.contiguous().data<float>(),
            projmatrix.contiguous().data<float>(),
            present.contiguous().data<bool>());
    }
    return present;
}
