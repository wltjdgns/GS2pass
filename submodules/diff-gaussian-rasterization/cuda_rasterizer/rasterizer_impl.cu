
#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

// =========================================
// STEP 1: GeometryState::fromChunk 메서드 수정
// =========================================
CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
    GeometryState geom;
    obtain(chunk, geom.depths, P, 128);
    obtain(chunk, geom.clamped, P * 3, 128);
    obtain(chunk, geom.internal_radii, P, 128);
    obtain(chunk, geom.means2D, P, 128);
    obtain(chunk, geom.cov3D, P * 6, 128);
    obtain(chunk, geom.conic_opacity, P, 128);
    obtain(chunk, geom.rgb, P * 3, 128);
    obtain(chunk, geom.tiles_touched, P, 128);
    
    // NEW: Allocate space for depth/normal parameters
    obtain(chunk, geom.depths_param, P, 128);      // (N×1)
    obtain(chunk, geom.normals_param, P * 3, 128); // (N×3)

    // === 추가: Basecolor/Roughness ===
    obtain(chunk, geom.basecolors_param, P * 3, 128);  // (N×3)
    obtain(chunk, geom.roughness_param, P, 128);       // (N×1)
    
    cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
    obtain(chunk, geom.scanning_space, geom.scan_size, 128);
    obtain(chunk, geom.point_offsets, P, 128);
    return geom;
}

// =========================================
// STEP 2: ImageState::fromChunk 메서드 수정
// =========================================
CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
    ImageState img;
    obtain(chunk, img.accum_alpha, N, 128);
    obtain(chunk, img.n_contrib, N, 128);
    obtain(chunk, img.ranges, N, 128);
    
    // NEW: Allocate space for rendered depth/normal buffers
    obtain(chunk, img.out_depth, N, 128);        // (H×W)
    obtain(chunk, img.out_normal, N * 3, 128);   // (H×W×3)

    // === 추가: Basecolor/Roughness ===
    obtain(chunk, img.out_basecolor, N * 3, 128);  // (H×W×3)
    obtain(chunk, img.out_roughness, N, 128);      // (H×W)
    
    return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// =========================================
// STEP 3: forward() 메서드 시그니처 수정 + 호출
// =========================================
int CudaRasterizer::Rasterizer::forward(
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
    int* radii,
    bool debug,
    bool homo_grid,
    const float* Hmat,
    // NEW: Depth/Normal parameters and outputs
    const float* depths,
    const float* normals,
    float* out_depth,
    float* out_normal,
    // === 추가 ===
    const float* basecolors,
    const float* roughness,
    float* out_basecolor,
    float* out_roughness)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// NEW: Copy depth/normal parameters to GPU memory
    if (depths != nullptr)
    {
        CHECK_CUDA(cudaMemcpy(geomState.depths_param, depths, P * sizeof(float), cudaMemcpyHostToDevice), debug);
    }
    if (normals != nullptr)
    {
        CHECK_CUDA(cudaMemcpy(geomState.normals_param, normals, P * 3 * sizeof(float), cudaMemcpyHostToDevice), debug);
    }
    
    // === 추가: Basecolor/Roughness 복사 ===
    if (basecolors != nullptr)
    {
        CHECK_CUDA(cudaMemcpy(geomState.basecolors_param, basecolors, P * 3 * sizeof(float), cudaMemcpyHostToDevice), debug);
    }
    if (roughness != nullptr)
    {
        CHECK_CUDA(cudaMemcpy(geomState.roughness_param, roughness, P * sizeof(float), cudaMemcpyHostToDevice), debug);
    }

    // NEW: Initialize output buffers to zero
    if (out_depth != nullptr)
    {
        CHECK_CUDA(cudaMemset(imgState.out_depth, 0, width * height * sizeof(float)), debug);
    }
    if (out_normal != nullptr)
    {
        CHECK_CUDA(cudaMemset(imgState.out_normal, 0, 3 * width * height * sizeof(float)), debug);
    }

    // === 추가: Basecolor/Roughness 출력 초기화 ===
    if (out_basecolor != nullptr)
    {
        CHECK_CUDA(cudaMemset(imgState.out_basecolor, 0, 3 * width * height * sizeof(float)), debug);
    }
    if (out_roughness != nullptr)
    {
        CHECK_CUDA(cudaMemset(imgState.out_roughness, 0, width * height * sizeof(float)), debug);
    }

	// Run preprocessing per-Gaussian
    CHECK_CUDA(FORWARD::preprocess(
        P, D, M,
        means3D,
        (glm::vec3*)scales,
        scale_modifier,
        (glm::vec4*)rotations,
        opacities,
        shs,
        geomState.clamped,
        cov3D_precomp,
        colors_precomp,
        viewmatrix, projmatrix,
        (glm::vec3*)cam_pos,
        width, height,
        focal_x, focal_y,
        tan_fovx, tan_fovy,
        radii,
        geomState.means2D,
        geomState.depths,
        geomState.cov3D,
        geomState.rgb,
        geomState.conic_opacity,
        tile_grid,
        geomState.tiles_touched,
        prefiltered,
        antialiasing,
        homo_grid,
        Hmat
    ), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// ... (기존 binning/sorting 코드 유지) ...

    // NEW: Pass depth/normal to forward render
    const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
    CHECK_CUDA(FORWARD::render(
        tile_grid, block,
        imgState.ranges,
        binningState.point_list,
        width, height,
        geomState.means2D,
        feature_ptr,
        geomState.conic_opacity,
        imgState.accum_alpha,
        imgState.n_contrib,
        background,
        out_color,
        geomState.depths,
        depth,
        geomState.normals_param,    // NEW
        imgState.out_depth,          // NEW
        imgState.out_normal,
        // === 추가 ===
        geomState.basecolors_param,
        geomState.roughness_param,
        imgState.out_basecolor,
        imgState.out_roughness), debug)
    
    // NEW: Copy rendered depth/normal back to host
    if (out_depth != nullptr)
    {
        CHECK_CUDA(cudaMemcpy(out_depth, imgState.out_depth, width * height * sizeof(float), cudaMemcpyDeviceToHost), debug);
    }
    if (out_normal != nullptr)
    {
        CHECK_CUDA(cudaMemcpy(out_normal, imgState.out_normal, 3 * width * height * sizeof(float), cudaMemcpyDeviceToHost), debug);
    }

    // === 추가: Basecolor/Roughness 결과 복사 ===
    if (out_basecolor != nullptr)
    {
        CHECK_CUDA(cudaMemcpy(out_basecolor, imgState.out_basecolor, 3 * width * height * sizeof(float), cudaMemcpyDeviceToHost), debug);
    }
    if (out_roughness != nullptr)
    {
        CHECK_CUDA(cudaMemcpy(out_roughness, imgState.out_roughness, width * height * sizeof(float), cudaMemcpyDeviceToHost), debug);
    }

    return num_rendered;
}

// =========================================
// STEP 4: backward() 메서드 시그니처 수정 + 호출
// =========================================
void CudaRasterizer::Rasterizer::backward(
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
    char* img_buffer,
    const float* dL_dpix,
    const float* dL_invdepths,
    // DepthNormal loss gradients
    const float* dL_depths,        // [H,W]
    const float* dL_normals,       // [H,W,3]
    // BasecolorRoughness loss gradients
    const float* dL_basecolors,    // [H,W,3]
    const float* dL_roughness,     // [H,W]
    float* dL_dmean2D,
    float* dL_dconic,
    float* dL_dopacity,
    float* dL_dcolor,
    float* dL_dinvdepth,
    // DepthNormal parameter gradients
    float* dL_ddepths,             // [N,1] ← 언더스코어 수정
    float* dL_dnormals,            // [N,3]
    // BasecolorRoughness parameter gradients  
    float* dL_dbasecolors,         // [N,3]
    float* dL_droughness,          // [N,1]
    float* dL_dmean3D,
    float* dL_dcov3D,
    float* dL_dsh,
    float* dL_dscale,
    float* dL_drot,
    // Input parameters
    const float* depths_in,        // [N,1]
    const float* normals_in,       // [N,3]
    const float* basecolors_in,    // [N,3]
    const float* roughness_in,     // [N,1]
    bool antialiasing,
    bool debug)
{
    GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
    BinningState binningState = BinningState::fromChunk(binning_buffer, R);
    ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

    if (radii == nullptr)
    {
        radii = geomState.internal_radii;
    }

    const float focal_y = height / (2.0f * tan_fovy);
    const float focal_x = width / (2.0f * tan_fovx);

    const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    const dim3 block(BLOCK_X, BLOCK_Y, 1);

    // Copy depth/normal loss gradients to GPU
    float* dL_depths_gpu = nullptr;
    float* dL_normals_gpu = nullptr;
    if (dL_depths != nullptr)
    {
        CHECK_CUDA(cudaMalloc(&dL_depths_gpu, width * height * sizeof(float)), debug);
        CHECK_CUDA(cudaMemcpy(dL_depths_gpu, dL_depths, width * height * sizeof(float), cudaMemcpyHostToDevice), debug);
    }
    if (dL_normals != nullptr)
    {
        CHECK_CUDA(cudaMalloc(&dL_normals_gpu, 3 * width * height * sizeof(float)), debug);
        CHECK_CUDA(cudaMemcpy(dL_normals_gpu, dL_normals, 3 * width * height * sizeof(float), cudaMemcpyHostToDevice), debug);
    }

    // Copy basecolor/roughness loss gradients to GPU
    float* dL_basecolors_gpu = nullptr;
    float* dL_roughness_gpu = nullptr;
    if (dL_basecolors != nullptr)
    {
        CHECK_CUDA(cudaMalloc(&dL_basecolors_gpu, 3 * width * height * sizeof(float)), debug);
        CHECK_CUDA(cudaMemcpy(dL_basecolors_gpu, dL_basecolors, 3 * width * height * sizeof(float), cudaMemcpyHostToDevice), debug);
    }
    if (dL_roughness != nullptr)
    {
        CHECK_CUDA(cudaMalloc(&dL_roughness_gpu, width * height * sizeof(float)), debug);
        CHECK_CUDA(cudaMemcpy(dL_roughness_gpu, dL_roughness, width * height * sizeof(float), cudaMemcpyHostToDevice), debug);
    }

    // Copy input parameters to GPU
    if (basecolors_in != nullptr)
    {
        CHECK_CUDA(cudaMemcpy(geomState.basecolors_param, basecolors_in, P * 3 * sizeof(float), cudaMemcpyHostToDevice), debug);
    }
    if (roughness_in != nullptr)
    {
        CHECK_CUDA(cudaMemcpy(geomState.roughness_param, roughness_in, P * sizeof(float), cudaMemcpyHostToDevice), debug);
    }
    if (depths_in != nullptr)
    {
        CHECK_CUDA(cudaMemcpy(geomState.depths_param, depths_in, P * sizeof(float), cudaMemcpyHostToDevice), debug);
    }
    if (normals_in != nullptr)
    {
        CHECK_CUDA(cudaMemcpy(geomState.normals_param, normals_in, P * 3 * sizeof(float), cudaMemcpyHostToDevice), debug);
    }

    const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
    
    // Allocate output gradients
    float* dL_ddepths_gpu = nullptr;       // ← 변수명 수정
    float* dL_dnormals_gpu = nullptr;      // ← 변수명 수정
    float* dL_dbasecolors_gpu = nullptr;   // ← 변수명 수정
    float* dL_droughness_gpu = nullptr;    // ← 변수명 수정
    
    if (dL_ddepths != nullptr)
    {
        CHECK_CUDA(cudaMalloc(&dL_ddepths_gpu, P * sizeof(float)), debug);
        CHECK_CUDA(cudaMemset(dL_ddepths_gpu, 0, P * sizeof(float)), debug);
    }
    if (dL_dnormals != nullptr)
    {
        CHECK_CUDA(cudaMalloc(&dL_dnormals_gpu, P * 3 * sizeof(float)), debug);
        CHECK_CUDA(cudaMemset(dL_dnormals_gpu, 0, P * 3 * sizeof(float)), debug);
    }
    if (dL_dbasecolors != nullptr)
    {
        CHECK_CUDA(cudaMalloc(&dL_dbasecolors_gpu, P * 3 * sizeof(float)), debug);
        CHECK_CUDA(cudaMemset(dL_dbasecolors_gpu, 0, P * 3 * sizeof(float)), debug);
    }
    if (dL_droughness != nullptr)
    {
        CHECK_CUDA(cudaMalloc(&dL_droughness_gpu, P * sizeof(float)), debug);
        CHECK_CUDA(cudaMemset(dL_droughness_gpu, 0, P * sizeof(float)), debug);
    }

    // ★ 수정: BACKWARD::render 호출 (basecolors/roughness 입력 추가)
    CHECK_CUDA(BACKWARD::render(
        tile_grid,
        block,
        imgState.ranges,
        binningState.point_list,
        width, height,
        background,
        geomState.means2D,
        geomState.conic_opacity,
        color_ptr,
        geomState.depths,
        geomState.normals_param,
        imgState.accum_alpha,              // final_Ts
        imgState.n_contrib,
        dL_dpix,
        dL_invdepths,
        dL_depths_gpu,
        dL_normals_gpu,
        dL_basecolors_gpu,                 // loss gradient
        dL_roughness_gpu,                  // loss gradient
        geomState.basecolors_param,        // ★ 추가: 입력값
        geomState.roughness_param,         // ★ 추가: 입력값
        (float3*)dL_dmean2D,
        (float4*)dL_dconic,
        dL_dopacity,
        dL_dcolor,
        dL_dinvdepth,
        dL_ddepths_gpu,                    // ← 변수명 수정
        dL_dnormals_gpu,                   // ← 변수명 수정
        dL_dbasecolors_gpu,                // ← 변수명 수정
        dL_droughness_gpu), debug);        // ← 변수명 수정

    // Preprocess backward
    const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
    CHECK_CUDA(BACKWARD::preprocess(P, D, M,
        (float3*)means3D,
        radii,
        shs,
        geomState.clamped,
        opacities,
        (glm::vec3*)scales,
        (glm::vec4*)rotations,
        scale_modifier,
        cov3D_ptr,
        viewmatrix,
        projmatrix,
        focal_x, focal_y,
        tan_fovx, tan_fovy,
        (glm::vec3*)campos,
        (float3*)dL_dmean2D,
        dL_dconic,
        dL_dinvdepth,
        dL_dopacity,
        (glm::vec3*)dL_dmean3D,
        dL_dcolor,
        dL_dcov3D,
        dL_dsh,
        (glm::vec3*)dL_dscale,
        (glm::vec4*)dL_drot,
        antialiasing), debug);

    // Copy gradients back to host
    if (dL_ddepths != nullptr && dL_ddepths_gpu != nullptr)
    {
        CHECK_CUDA(cudaMemcpy(dL_ddepths, dL_ddepths_gpu, P * sizeof(float), cudaMemcpyDeviceToHost), debug);
        CHECK_CUDA(cudaFree(dL_ddepths_gpu), debug);
    }
    if (dL_dnormals != nullptr && dL_dnormals_gpu != nullptr)
    {
        CHECK_CUDA(cudaMemcpy(dL_dnormals, dL_dnormals_gpu, P * 3 * sizeof(float), cudaMemcpyDeviceToHost), debug);
        CHECK_CUDA(cudaFree(dL_dnormals_gpu), debug);
    }
    if (dL_dbasecolors != nullptr && dL_dbasecolors_gpu != nullptr)
    {
        CHECK_CUDA(cudaMemcpy(dL_dbasecolors, dL_dbasecolors_gpu, P * 3 * sizeof(float), cudaMemcpyDeviceToHost), debug);
        CHECK_CUDA(cudaFree(dL_dbasecolors_gpu), debug);
    }
    if (dL_droughness != nullptr && dL_droughness_gpu != nullptr)
    {
        CHECK_CUDA(cudaMemcpy(dL_droughness, dL_droughness_gpu, P * sizeof(float), cudaMemcpyDeviceToHost), debug);
        CHECK_CUDA(cudaFree(dL_droughness_gpu), debug);
    }
    
    // Cleanup
    if (dL_depths_gpu != nullptr) CHECK_CUDA(cudaFree(dL_depths_gpu), debug);
    if (dL_normals_gpu != nullptr) CHECK_CUDA(cudaFree(dL_normals_gpu), debug);
    if (dL_basecolors_gpu != nullptr) CHECK_CUDA(cudaFree(dL_basecolors_gpu), debug);
    if (dL_roughness_gpu != nullptr) CHECK_CUDA(cudaFree(dL_roughness_gpu), debug);
}