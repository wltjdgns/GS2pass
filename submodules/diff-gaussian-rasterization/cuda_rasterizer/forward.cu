
#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#ifndef WARP_GRID_EPS
#define WARP_GRID_EPS 1e-8f
#endif

// === [ADD] Homography warp + Jacobian ===
__device__ inline bool warp_point_and_jacobian(
    float x, float y, const float* H,
    float2 &mp, float4 &J) // mp=(x',y'), J=(j00,j01,j10,j11)
{
    const float h00=H[0], h01=H[1], h02=H[2];
    const float h10=H[3], h11=H[4], h12=H[5];
    const float h20=H[6], h21=H[7], h22=H[8];

    const float a = h00*x + h01*y + h02;
    const float b = h10*x + h11*y + h12;
    const float w = h20*x + h21*y + h22;
	if (fabsf(w) < WARP_GRID_EPS) {
        if (blockIdx.x==0 && threadIdx.x==0) {
            printf("[DBG] warp FAIL: w≈0 at (x=%.1f,y=%.1f)\n", x, y);
        }
        return false;
    }

    const float invw = 1.0f / w;
    mp.x = a * invw;
    mp.y = b * invw;

    const float w2 = w*w;
    const float j00 = (h00*w - a*h20) / w2;
    const float j01 = (h01*w - a*h21) / w2;
    const float j10 = (h10*w - b*h20) / w2;
    const float j11 = (h11*w - b*h21) / w2;
    J = make_float4(j00, j01, j10, j11);
    return true;
}

__device__ inline bool pullback_conic_and_alpha(
    const float4 &J, float &A, float &B, float &Cc, float &alpha)
{
    const float j00=J.x, j01=J.y, j10=J.z, j11=J.w;
    const float det = j00*j11 - j01*j10;
    const float absdet = fabsf(det);
	if (absdet < 1e-12f) {
        if (blockIdx.x==0 && threadIdx.x==0) {
            printf("[DBG] pullback SKIP: detJ≈0\n");
        }
        return false;
    }

    const float invdet = 1.0f / det;
    // J^{-1}
    const float i00 =  j11*invdet;
    const float i01 = -j01*invdet;
    const float i10 = -j10*invdet;
    const float i11 =  j00*invdet;

    // Q' = J^{-T} Q J^{-1}
    const float t00 = A*i00 + B*i10;
    const float t01 = A*i01 + B*i11;
    const float t10 = B*i00 + Cc*i10;
    const float t11 = B*i01 + Cc*i11;
    const float A2 = i00*t00 + i10*t10;
    const float B2 = i00*t01 + i10*t11;
    const float C2 = i01*t01 + i11*t11;
    A = A2; B = B2; Cc = C2;

    const float amin=1e-4f, amax=0.99f;
    alpha = fminf(amax, fmaxf(amin, alpha / (absdet + 1e-8f)));
    return true;
}


namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
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
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing,
	// [ADD]
	bool homo_grid,
   const float* __restrict__ Hmat)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	constexpr float h_var = 0.3f;
	const float det_cov = cov.x * cov.z - cov.y * cov.y;
	cov.x += h_var;
	cov.z += h_var;
	const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;
	float h_convolution_scaling = 1.0f;

	if(antialiasing)
		h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability

	// Invert covariance (EWA algorithm)
	const float det = det_cov_plus_h_cov;

	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// === [REPLACE] Π′(워프 그리드)로의 프리미티브 풀백을
	// ===           타일 AABB 계산 '이전'에 수행한다.

	// (1) 먼저 원래 image-space 중심(Π) 좌표를 산출
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };

	// (2) 기본 opacity (AA 스케일 반영)
	float opacity = opacities[idx] * h_convolution_scaling;

	// (3) homo_grid 이면, 중심과 conic/opacity를 Π′로 덮어쓴다.
	//     (위에 정의한 헬퍼: warp_point_and_jacobian, pullback_conic_and_alpha 사용)
	if (homo_grid) {
		float2 mp;        // warped center (x', y')
		float4 J;         // Jacobian 2x2 packed as (j00,j01,j10,j11)
		if (warp_point_and_jacobian(point_image.x, point_image.y, Hmat, mp, J)) {
			// conic(A,B,C)와 alpha를 풀백: Q' = J^{-T} Q J^{-1}, alpha' = alpha / |detJ|
			float A = conic.x, B = conic.y, Cc = conic.z;
			float a = opacity;
			if (pullback_conic_and_alpha(J, A, B, Cc, a)) {
				conic.x = A; conic.y = B; conic.z = Cc;
				opacity = a;
			}
			// 중심 좌표 교체
			point_image = mp;
			
		}
		// warp 실패(w≈0 등) 시엔 원본 유지(안정성)
	}

	// (4) 타일 AABB는 conic'에 맞춘 반경으로 계산해야 한다.
	//     conic은 inverse-covariance이므로, 2x2 역행렬로 cov'를 만들고
	//     기존과 동일하게 eigenvalue 기반 반경을 구한다.
	float detQ = conic.x * conic.z - conic.y * conic.y;
	if (detQ <= 0.0f)
		return; // 비양정/수치불안정 방지
	float invA =  conic.z / detQ;   // cov'.xx
	float invB = -conic.y / detQ;   // cov'.xy
	float invC =  conic.x / detQ;   // cov'.yy

	float mid = 0.5f * (invA + invC);
	float disc = max(0.1f, mid * mid - (invA * invC - invB * invB));
	float lambda1 = mid + sqrt(disc);
	float lambda2 = mid - sqrt(disc);
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	

	// (5) 이제 Π′ 기준 중심/반경으로 타일 사각형 계산
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);


	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// === [기존 흐름 유지] 색상(precomputed 없으면 SH → RGB)
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// === [출력 버퍼들 Π′ 값으로 덮어쓰기]
	depths[idx] = p_view.z;                  // 깊이는 카메라 기준 scalar라 변경 없음
	radii[idx]  = my_radius;
	points_xy_image[idx] = point_image;      // ← m' (Π′ 중심)
	conic_opacity[idx]   = {                 // ← Q'와 α'
		conic.x, conic.y, conic.z, opacity
	};

	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);

}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ depths,
	float* __restrict__ invdepth,
    
    // NEW: 깊이와 법선 입력 파라미터
    const float* __restrict__ normals,      // (N×3) 각 Gaussian의 법선
    
    // NEW: 출력 버퍼
    float* __restrict__ out_depth,          // (H×W) 렌더링된 depth
    float* __restrict__ out_normal,
    // === 추가 ===
    const float* __restrict__ basecolors,      // (N×3)
    const float* __restrict__ roughness,       // (N×1)
    float* __restrict__ out_basecolor,         // (H×W×3)
    float* __restrict__ out_roughness          // (H×W)
	)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// NEW: 깊이와 법선 shared memory
    __shared__ float collected_depths[BLOCK_SIZE];
    __shared__ float collected_normals[3 * BLOCK_SIZE];  // 3 channels

	// === 추가: Albedo/Roughness shared memory ===
	__shared__ float collected_basecolors[3 * BLOCK_SIZE];  // RGB
	__shared__ float collected_roughness[BLOCK_SIZE];       // 1 channel

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float accum[CHANNELS] = { 0 };

	// NEW: 깊이와 법선 accumulator
    float accum_depth = 0.0f;
    float accum_normal[3] = { 0.0f, 0.0f, 0.0f };

	// === 추가: Albedo/Roughness accumulator ===
	float accumbasecolor[3] = {0.0f, 0.0f, 0.0f};  // RGB
	float accumroughness = 0.0f;

	float expected_invdepth = 0.0f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];

			// NEW: 깊이와 법선 데이터 로드
            collected_depths[block.thread_rank()] = depths[coll_id];
            
            // 법선은 3채널이므로 별도 처리
            for (int c = 0; c < 3; c++)
                collected_normals[block.thread_rank() * 3 + c] = normals[coll_id * 3 + c];

			// === 추가: Basecolor/Roughness 로딩 ===
			if (basecolors) {
				for (int c = 0; c < 3; c++)
					collected_basecolors[block.thread_rank() * 3 + c] = basecolors[coll_id * 3 + c];
			}
			if (roughness) {
				collected_roughness[block.thread_rank()] = roughness[coll_id];
			}
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				accum[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			if(invdepth)
			expected_invdepth += (1 / depths[collected_id[j]]) * alpha * T;

			// ===== NEW: Depth blending =====
            if (out_depth)
                accum_depth += collected_depths[j] * alpha * T;

            // ===== NEW: Normal blending (3 channels) =====
            if (out_normal)
            {
                for (int c = 0; c < 3; c++)
                    accum_normal[c] += collected_normals[j * 3 + c] * alpha * T;
            }

			// === 추가: Basecolor blending ===
			if (out_basecolor)
			{
				for (int c = 0; c < 3; c++)
					accumbasecolor[c] += collected_basecolors[j * 3 + c] * alpha * T;
			}

			// === 추가: Roughness blending ===
			if (out_roughness)
			{
				accumroughness += collected_roughness[j] * alpha * T;
			}

			T = test_T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = accum[ch] + T * bg_color[ch];

		if (invdepth)
		invdepth[pix_id] = expected_invdepth;// 1. / (expected_depth + T * 1e3);

		// NEW: Depth output
        if (out_depth)
            out_depth[pix_id] = accum_depth;

        // NEW: Normal output (3 channels, planar layout)
		if (out_normal)
		{
			// Normalize before output
			float norm_length = sqrtf(accum_normal[0]*accum_normal[0] + 
									accum_normal[1]*accum_normal[1] + 
									accum_normal[2]*accum_normal[2]);
			
			if (norm_length > 1e-8f) 
			{
				float inv_norm = 1.0f / norm_length;
				for (int c = 0; c < 3; c++)
					out_normal[c * H * W + pix_id] = accum_normal[c] * inv_norm;
			} 
			else 
			{
				// Zero vector fallback (up direction or zero)
				out_normal[0 * H * W + pix_id] = 0.0f;
				out_normal[1 * H * W + pix_id] = 0.0f;
				out_normal[2 * H * W + pix_id] = 0.0f;
			}
		}


		// === 추가: Basecolor output ===
		if (out_basecolor)
		{
			for (int c = 0; c < 3; c++)
				out_basecolor[c * H * W + pix_id] = accumbasecolor[c];
		}

		// === 추가: Roughness output ===
		if (out_roughness)
		{
			out_roughness[pix_id] = accumroughness;
		}
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* depths,
	float* depth,
    
    // NEW 파라미터들
    const float* normals,
    float* out_depth,
    float* out_normal,
    // === 추가 ===
    const float* basecolors,
    const float* roughness,
    float* out_basecolor,
    float* out_roughness)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depths, 
		depth,
        
        // NEW 파라미터들 전달
        normals,
        out_depth,
        out_normal,
	    basecolors, 
		roughness, 
		out_basecolor, 
		out_roughness
	);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
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
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool antialiasing,
	bool homo_grid,
	const float* Hmat)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		antialiasing,
		homo_grid,
		Hmat
		);
    #if !defined(__CUDA_ARCH__)
    {
        cudaError_t __e = cudaGetLastError();
        if (__e != cudaSuccess) {
            printf("[DBG] preprocessCUDA launch error: %s\n", cudaGetErrorString(__e));
        }
        cudaDeviceSynchronize(); // printf flush & 런타임 에러 조기노출
    }
    #endif
	
}