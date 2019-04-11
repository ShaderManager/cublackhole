#include <stdint.h>
#include <utility>
#include <math_constants.h>

static const uint32_t DEFAULT_NUM_CUDA_THREADS = 512;

static inline uint32_t cudaComputeBlocks(uint32_t N, uint32_t threads_per_block = DEFAULT_NUM_CUDA_THREADS)
{
	return (N + threads_per_block - 1) / (threads_per_block);
}

#define CUDA_FOR(index, count) \
	for (auto index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += blockDim.x * gridDim.x)

__device__ __host__ __inline__ float3 operator - (float3 a)
{
	return float3{ -a.x, -a.y, -a.z };
}

__device__ __host__ __inline__ float3 operator *(float3 a, float val)
{
	return float3{ a.x * val, a.y * val, a.z * val };
}

__device__ __host__ __inline__ float3 operator *(float val, float3 a)
{
	return float3{ a.x * val, a.y * val, a.z * val };
}

__device__ __host__ __inline__ float3 operator * (float3 a, float3 b)
{
	return float3{ a.x * b.x, a.y * b.y, a.z * b.z };
}

__device__ __host__ __inline__ float3 operator + (float3 a, float3 b)
{
	return float3{ a.x + b.x, a.y + b.y, a.z + b.z };
}

__device__ __host__ __inline__ float3 operator - (float3 a, float3 b)
{
	return a + (-b);
}

__device__ __host__ __inline__ float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ __inline__ float length(float3 a)
{
    return sqrtf(dot(a, a));
}

__device__ __host__ __inline__ float3 normalize(float3 a)
{
    return a * (1.0f / length(a));
}

__device__ __host__ __inline__ float3 cross(float3 a, float3 b)
{
    return float3 {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

__device__ __host__ __inline__ float4 make_float4(float3 a, float w)
{
	return make_float4(a.x, a.y, a.z, w);
}

struct Camera
{
	float3 pos;
	float3 dir;
	float3 up;
	float3 right;
};

__global__ void generate_photons(const Camera cam, uint32_t fb_width, uint32_t fb_height, float4* points, float4* velocity, float3* angular_moment)
{
	CUDA_FOR(idx, fb_width * fb_height)
	{
        uint32_t x = idx % fb_width;
        uint32_t y = (idx / fb_width); // % height

        float u = float(x) / fb_width;
        float v = float(y) / fb_height;

        u = u * 2 - 1;
        v = 1 - v * 2;

        float3 point = cam.pos;
        float3 vel = normalize(u * cam.right + v * cam.up + cam.dir);

        points[idx] = make_float4(point, 0);
        velocity[idx] = make_float4(vel, 1.0f);
        angular_moment[idx] = cross(point, vel);
	}
}

__device__ __inline__ float3 F_r(const float3 r, const float h2)
{
	return -1.5f * h2 * r * (1.0f / powf(dot(r, r), 2.5f));
}

__global__ void trace_photons(uint32_t num_photons, uint32_t subiters, float step, float4* points, float4* velocity, const float3* angular_moment)
{
    CUDA_FOR(idx, num_photons)
    {
        for(uint32_t i = 0; i < subiters; i++)
        {
			const float h2 = dot(angular_moment[idx], angular_moment[idx]);			

			const float mask = velocity[idx].w;
			const float timestep = mask * step;

			float3 r{ points[idx].x, points[idx].y, points[idx].z };
			float3 v{ velocity[idx].x, velocity[idx].y, velocity[idx].z };

			// 4th order Yoshida integrator
			const float c1 = 0.6756;
			const float c2 = -0.1756;
			const float c3 = c2;
			const float c4 = c1;
			const float d1 = c1 * 2;
			const float d2 = -1.7024;
			const float d3 = d1;

			r = r + c1 * timestep * v;
			v = v + d1 * timestep * F_r(r, h2);

			r = r + c2 * timestep * v;
			v = v + d2 * timestep * F_r(r, h2);

			r = r + c3 * timestep * v;
			v = v + d3 * timestep * F_r(r, h2);

			r = r + c4 * timestep * v;

			points[idx].x = r.x;
			points[idx].y = r.y;
			points[idx].z = r.z;

			velocity[idx].x = v.x;
			velocity[idx].y = v.y;
			velocity[idx].z = v.z;

            if (dot(r, r) <= 1) // If photon crossed event horizon, it marked as inactive
			{
				velocity[idx].w = 0;
			}
        }
    }
}

__device__ __inline__ float4 rgbaUcharToFloat(uchar4 c)
{
	float4 rgba;
	rgba.x = c.x * 0.003921568627f;  //  /255.0f;
	rgba.y = c.y * 0.003921568627f;  //  /255.0f;
	rgba.z = c.z * 0.003921568627f; //  /255.0f;
	rgba.w = c.w * 0.003921568627f; //  /255.0f;
	return rgba;
}

__device__ __inline__ float4 fetch_tex2D(cudaTextureObject_t tex, float u, float v)
{
	return rgbaUcharToFloat(tex2D<uchar4>(tex, u, v));
}

__global__ void splat_photons(float3* framebuffer, uint32_t width, uint32_t height, const float4* points, const float4* dirs, cudaTextureObject_t bg_tex)
{
    CUDA_FOR(idx, width * height)
    {        
        //uint32_t x = idx % width;
        //uint32_t y = (idx / width); // % height

        //float u = float(x) / width;
        //float v = float(y) / height;

        //float phi = atan2f(dirs[idx].x, dirs[idx].z);
        //float theta = atan2f(dirs[idx].y, sqrtf(dirs[idx].x * dirs[idx].x + dirs[idx].z * dirs[idx].z));
        //float u = (phi + 4.5) / (CUDART_PI_F * 2);
        //float v = (theta + CUDART_PIO2_F) / CUDART_PI_F;

		float4 ray_pos = points[idx];
		float4 ray_dir = dirs[idx];

		float3 pixel_value{ 0, 0, 0 };

		if (dirs[idx].w > 0)
		{
			float theta = acos(ray_dir.y);
			float phi = atan2(ray_dir.z, ray_dir.x);
			float u = (phi + CUDART_PI_F) / CUDART_PI_F;
			float v = theta / CUDART_PI_F;

			float4 bg_value = fetch_tex2D(bg_tex, u, v);

			pixel_value.x = bg_value.x;
			pixel_value.y = bg_value.y;
			pixel_value.z = bg_value.z;
			//pixel_value.x = u;
			//pixel_value.y = v;
			//pixel_value.z = 0;
		}

        pixel_value = pixel_value * 255.f; // Convert to [0, 255] range

        framebuffer[idx] = pixel_value;
    }
}

template<typename Kernel, typename... Args> bool launchKernel(uint32_t nthreads, Kernel kernel, Args&&... args)
{	
    kernel<<<cudaComputeBlocks(nthreads), DEFAULT_NUM_CUDA_THREADS>>>(std::forward<Args>(args)...);	
    
    return true;    
}

__device__ __host__ __inline__ float lerp(float a, float b, float t)
{
	return a + (b - a) * t;
}

struct TracerState
{
	float3* device_framebuffer = nullptr;
	float4* points = nullptr;
	float4* velocity = nullptr;
	float3* angular_moment = nullptr;
};

void init_tracer_state(TracerState** state, uint32_t width, uint32_t height)
{
	TracerState* new_state = new TracerState();

	uint32_t num_photons = width * height;

	cudaMalloc(&new_state->device_framebuffer, width * height * sizeof(float3));

	cudaMalloc(&new_state->points, num_photons * sizeof(float4));
	cudaMalloc(&new_state->velocity, num_photons * sizeof(float4));
	cudaMalloc(&new_state->angular_moment, num_photons * sizeof(float3));

	*state = new_state;
}

void destroy_tracer_state(TracerState* state)
{
	cudaFree(state->device_framebuffer);
	cudaFree(state->points);
	cudaFree(state->velocity);
	cudaFree(state->angular_moment);
}

void trace_scene(TracerState* state, float* framebuffer, uint32_t width, uint32_t height, cudaTextureObject_t bg_tex, uint32_t frame)
{	
    float fov = 75;
	//float fov = lerp(35, 120, cosf(frame / 180.f * CUDART_PI_F) * 0.5f + 0.5f);

	float aspect = float(width) / height;

	float v_len = tanf(0.5f * fov * CUDART_PI_F / 180.0f);
	float u_len = aspect * v_len;

	Camera cam;
    //cam.pos = make_float3(0, 1.0f, -20.f);
	cam.pos = make_float3(20 * cosf(frame / 180.f * CUDART_PI_F), 1.0f, 20 * sinf(frame / 180.f * CUDART_PI_F));
    cam.dir = normalize(-cam.pos);
    cam.up = make_float3(0, 1, 0) * v_len;
    cam.right = normalize(cross(cam.up, cam.dir)) * u_len;

    launchKernel(width * height, generate_photons, cam, width, height, state->points, state->velocity, state->angular_moment);

    for(uint32_t iter = 0; iter < 50; iter++)
    {
        launchKernel(width * height, trace_photons, width * height, 10, 0.05f, state->points, state->velocity, state->angular_moment);
    }

    launchKernel(width * height, splat_photons, state->device_framebuffer, width, height, state->points, state->velocity, bg_tex);
       
    cudaMemcpy(framebuffer, state->device_framebuffer, width * height * sizeof(float3), cudaMemcpyDeviceToHost);
}
