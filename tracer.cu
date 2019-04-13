#include <stdint.h>
#include <utility>
#include <math_constants.h>

#include "tracer.h"

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

template<typename T> __device__ __host__ __inline__ T lerp(T a, T b, float t)
{
	return a + (b - a) * t;
}

struct Cylinder
{
	float4 ptr1; // Base point+outer radius squared
	float4 ptr2; // Top point+inner radius squared

	float3 X, Y, Z; // Coordinate frame with Y axis aligned along ptr2-ptr1 vector
	float heightsq; // Squared height of cylinder

	static __device__ __host__ Cylinder construct(const float3 pt1, const float3 pt2, float outer_radius, float inner_radius)
	{
		Cylinder result;

		float3 C = pt2 - pt1;
		float heightsq = dot(C, C);

		// Transform reference frame so cylinder is aligned along Y axis
		float3 Y = normalize(C);
		float3 X = cross(float3{ 0, 0, 1 }, Y);

		if (dot(X, X) == 0)
		{
			X = cross(float3{ 0, 1, 0 }, Y);
		}

		float3 Z = cross(X, Y);

		result.ptr1 = make_float4(pt1, outer_radius * outer_radius);
		result.ptr2 = make_float4(pt2, inner_radius * inner_radius);
		result.X = X;
		result.Y = Y;
		result.Z = Z;
		result.heightsq = heightsq;

		return result;
	}

	__device__ __host__ __inline__ float3 base_point() const
	{
		return float3{ ptr1.x, ptr1.y, ptr1.z };
	}

	__device__ __host__ __inline__ float3 top_point() const
	{
		return float3{ ptr2.x, ptr2.y, ptr2.z };
	}

	__device__ __host__ __inline__ float outer_radius_sq() const
	{
		return ptr1.w;
	}

	__device__ __host__ __inline__ float inner_radius_sq() const
	{
		return ptr2.w;
	}
};

struct SceneState
{
	Cylinder accretion_disk;
	cudaTextureObject_t bg_tex;
};

__device__ float point_in_cylinder(const Cylinder& cylinder, const float3 p)
{
	float3 pt1 = cylinder.base_point();
	float3 pt2 = cylinder.top_point();

	float3 d = pt2 - pt1; // translate so pt1 is origin.
	float3 pd = p - pt1; // vector from pt1 to test point.	

	float outer_radius_sq = cylinder.outer_radius_sq();
	float inner_radius_sq = cylinder.inner_radius_sq();

	// Dot the d and pd vectors to see if point lies behind the 
	// cylinder cap at pt1.x, pt1.y, pt1.z

	float dDotPd = dot(d, pd);
	float heightsq = cylinder.heightsq;

	// If dot is less than zero the point is behind the pt1 cap.
	// If greater than the cylinder axis line segment length squared
	// then the point is outside the other end cap at pt2.

	if (dDotPd < 0.0f || dDotPd > heightsq)
	{
		return -1.0f;
	}
	else
	{
		// Point lies within the parallel caps, so find
		// distance squared from point to line, using the fact that sin^2 + cos^2 = 1
		// the dot = cos() * |d||pd|, and cross*cross = sin^2 * |d|^2 * |pd|^2
		// Carefull: '*' means mult for scalars and dotproduct for vectors
		// In short, where dist is pt distance to cyl axis: 
		// dist = sin( pd to d ) * |pd|
		// distsq = dsq = (1 - cos^2( pd to d)) * |pd|^2
		// dsq = ( 1 - (pd * d)^2 / (|pd|^2 * |d|^2) ) * |pd|^2
		// dsq = pd * pd - dot * dot / lengthsq
		//  where lengthsq is d*d or |d|^2 that is passed into this function 

		// distance squared to the cylinder axis:

		float dsq = dot(pd, pd) - dDotPd * dDotPd / heightsq;

		if (dsq > outer_radius_sq || dsq < inner_radius_sq)
		{
			return -1.0f;
		}
		else
		{
			return dsq - inner_radius_sq;		// return distance squared to inner radius
		}
	}
}

// Ray-cylinder with outer and inner radius
// In fact, there are 4 intersection tests: with upper and bottom disks and with outer and inner cylinders
// UVW: U - normalized angle between point and "X" axis; V - normalized distance between point and inner radius; W - normalized height from pt1 to point
__device__ int ray_cylinder_intersection(const Cylinder& cyl, const float3 ray_origin, const float3 ray_dir, float3& uvw)
{
	float3 pt1 = cyl.base_point();
	float3 pt2 = cyl.top_point();
	float radiussq = cyl.outer_radius_sq();
	float inner_radiussq = cyl.inner_radius_sq();		

	float3 OC
	{
		dot(pt1, cyl.X),
		dot(pt1, cyl.Y),
		dot(pt1, cyl.Z)
	};

	float3 O
	{
		dot(ray_origin, cyl.X) - OC.x,
		dot(ray_origin, cyl.Y) - OC.y,
		dot(ray_origin, cyl.Z) - OC.z
	};

	float3 D
	{
		dot(ray_dir, cyl.X),
		dot(ray_dir, cyl.Y),
		dot(ray_dir, cyl.Z),
	};

	// Fast path when ray origin is already inside of cylinder
	float dist = point_in_cylinder(cyl, ray_origin);

	if (dist >= 0)
	{
		uvw = make_float3(0.5f * atan2f(O.z, O.x) / CUDART_PI + 0.5f, dist / (radiussq - inner_radiussq), O.y * O.y / cyl.heightsq);
		return 1;
	}

	float a = D.x * D.x + D.z * D.z;
	float b = D.x * O.x + D.z * O.z;
	float c = O.x * O.x + O.z * O.z - radiussq;
	float c2 = O.x * O.x + O.z * O.z - inner_radiussq;	

	float t = 1e30;

	int mask = 0;

	float N1 = dot(cyl.Y, ray_dir);
	float N2 = dot(-cyl.Y, ray_dir);
	float disc = b * b - a * c;
	float disc2 = b * b - a * c2;

	/// TODO: make less comparisons
	if (N1 != 0)
	{
		float t0 = dot(cyl.Y, pt2 - ray_origin) / N1;
		float3 p = ray_origin + t0 * ray_dir;
		float v = dot(p - pt2, p - pt2);

		if (t0 >= 0 && t0 < t && v <= radiussq && v >= inner_radiussq)
		{
			uvw = make_float3(0.5f * atan2f(O.z + D.z * t0, O.x + D.x * t0) / CUDART_PI + 0.5f, (v - inner_radiussq) / (radiussq - inner_radiussq), 1.f);

			t = t0;
			mask |= 1;
		}
	}	

	if (N2 != 0)
	{
		float t0 = dot(-cyl.Y, pt1 - ray_origin) / N2;
		float3 p = ray_origin + t0 * ray_dir;
		float v = dot(p - pt1, p - pt1);

		if (t0 >= 0 && t0 < t && v <= radiussq && v >= inner_radiussq)
		{
			uvw = make_float3(0.5f * atan2f(O.z + D.z * t0, O.x + D.x * t0) / CUDART_PI + 0.5f, (v - inner_radiussq) / (radiussq - inner_radiussq), 0.f);

			t = t0;
			mask |= 1;
		}
	}

	if (disc > 0)
	{
		float sdisc = sqrtf(disc);

		float t0 = (-b - sdisc) / a;
		float t1 = (-b + sdisc) / a;

		float y0 = O.y + D.y * t0;
		float y1 = O.y + D.y * t1;

		if (t0 < t && y0 >= 0 && y0 * y0 <= cyl.heightsq)
		{
			 uvw = make_float3(0.5f * atan2f(O.z + D.z * t0, O.x + D.x * t0) / CUDART_PI + 0.5f, 1.0f, (y0 * y0) / cyl.heightsq);

			t = t0;
			mask |= 1;
		}

		if (t1 < t && y1 >= 0 && y1 * y1 <= cyl.heightsq)
		{
			uvw = make_float3(0.5f * atan2f(O.z + D.z * t1, O.x + D.x * t1) / CUDART_PI + 0.5f, 1.0f, (y1 * y1) / cyl.heightsq);

			t = t1;
			mask |= 1;
		}	
	}	

	if (disc2 > 0)
	{
		float sdisc = sqrtf(disc2);

		float t0 = (-b - sdisc) / a;
		float t1 = (-b + sdisc) / a;

		float y0 = O.y + D.y * t0;
		float y1 = O.y + D.y * t1;

		if (t0 < t && y0 >= 0 && y0 * y0 <= cyl.heightsq)
		{
			uvw = make_float3(0.5f * atan2f(O.z + D.z * t0, O.x + D.x * t0) / CUDART_PI + 0.5f, 0.0f, (y0 * y0) / cyl.heightsq);

			t = t0;
			mask |= 1;
		}

		if (t1 < t && y1 >= 0 && y1 * y1 <= cyl.heightsq)
		{
			uvw = make_float3(0.5f * atan2f(O.z + D.z * t1, O.x + D.x * t1) / CUDART_PI + 0.5f, 0.0f, (y1 * y1) / cyl.heightsq);

			t = t1;
			mask |= 1;
		}
	}

	return mask;
}

__global__ void generate_photons(const Camera cam, uint32_t fb_width, uint32_t fb_height, float4* points, float4* velocity)
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
		float3 vel = u * cam.right + v * cam.up + cam.dir;

		float3 angular_moment = cross(point, vel);
		float h2 = dot(angular_moment, angular_moment);

		points[idx] = make_float4(point, h2);
		velocity[idx] = make_float4(vel, 1.0f);
	}
}

__device__ __inline__ float3 F_r(const float3 r, const float h2)
{
	return -1.5f * h2 * r * (1.0f / powf(dot(r, r), 2.5f));
	//return make_float3(0, 0, 0);
}

__global__ void integrate_photons(uint32_t num_photons, uint32_t subiters, float step, float4* points, float4* velocity, const SceneState scene)
{
    CUDA_FOR(idx, num_photons)
    {
        for(uint32_t i = 0; i < subiters; i++)
        {
			const int mask = velocity[idx].w > 0;

			if (mask == 0)
				break;

			const float h2 = points[idx].w;	

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

            if (dot(r, r) <= 1) // If photon crossed event horizon, it's marked as inactive
			{
				velocity[idx].w = 0;
			}
			else
			{
				float dist = point_in_cylinder(scene.accretion_disk, r);

				// If photon ends within accretion disk volume, terminate it too
				// Trick is if photon is outside of volume, point_in_cylinder returns -1 value, so after negation we get proper "1" value
				// i.e. photon is still alive
				velocity[idx].w = -dist;
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

__global__ void trace_photons(uint32_t num_photons, const float4* points, const float4* dirs, float4* radiance, const SceneState scene)
{
	CUDA_FOR(idx, num_photons)
	{	
		float3 ray_pos = make_float3(points[idx].x, points[idx].y, points[idx].z);
		float3 ray_dir = normalize(make_float3(dirs[idx].x, dirs[idx].y, dirs[idx].z));

		float3 color_value{ 0, 0, 0 };				
		float3 uvw{ 0, 0, 0 };

		if (dirs[idx].w != 0)
		{
			if (ray_cylinder_intersection(scene.accretion_disk, ray_pos, ray_dir, uvw) != 0) // Our photon has been terminated inside of accretion disk
			{
				color_value = uvw;
			}
			else if (dirs[idx].w > 0) // Photon flies away in space
			{
				// Alternative sampling method from starless
				//float phi = atan2f(dirs[idx].x, dirs[idx].z);
				//float theta = atan2f(dirs[idx].y, sqrtf(dirs[idx].x * dirs[idx].x + dirs[idx].z * dirs[idx].z));
				//float u = (phi + 4.5) / (CUDART_PI_F * 2);
				//float v = (theta + CUDART_PIO2_F) / CUDART_PI_F;

				// Latlong sampling
				float theta = acos(ray_dir.y);
				float phi = atan2(ray_dir.z, ray_dir.x);
				float u = (phi + CUDART_PI_F) / CUDART_PI_F;
				float v = theta / CUDART_PI_F;

				float4 bg_value = fetch_tex2D(scene.bg_tex, u, v);

				color_value.x = bg_value.x;
				color_value.y = bg_value.y;
				color_value.z = bg_value.z;
			}
		}

		radiance[idx] = make_float4(color_value, 1.0f);
	}
}

__global__ void splat_photons(float3* framebuffer, uint32_t width, uint32_t height, const float4* radiance)
{
    CUDA_FOR(idx, width * height)
    {                
		float3 pixel_value{ radiance[idx].x, radiance[idx].y, radiance[idx].z };		

        pixel_value = pixel_value * 255.f; // Convert to [0, 255] range

        framebuffer[idx] = pixel_value;
    }
}

template<typename Kernel, typename... Args> bool launchKernel(uint32_t nthreads, Kernel kernel, Args&&... args)
{	
    kernel<<<cudaComputeBlocks(nthreads), DEFAULT_NUM_CUDA_THREADS>>>(std::forward<Args>(args)...);	
    
    return true;    
}

struct TracerState
{
	float3* device_framebuffer = nullptr;
	float4* points = nullptr;
	float4* velocity = nullptr;
	float4* radiance = nullptr;
};

void init_tracer_state(TracerState** state, uint32_t width, uint32_t height)
{
	TracerState* new_state = new TracerState();

	uint32_t num_photons = width * height;

	cudaMalloc(&new_state->device_framebuffer, width * height * sizeof(float3));

	cudaMalloc(&new_state->points, num_photons * sizeof(float4));
	cudaMalloc(&new_state->velocity, num_photons * sizeof(float4));
	cudaMalloc(&new_state->radiance, num_photons * sizeof(float4));

	*state = new_state;
}

void destroy_tracer_state(TracerState* state)
{
	cudaFree(state->device_framebuffer);
	cudaFree(state->points);
	cudaFree(state->velocity);
	cudaFree(state->radiance);
}

void setup_camera(Camera& cam, const float3 eye, const float3 look_at, const float3 up, float fov, float aspect)
{
	float v_len = tanf(0.5f * fov * CUDART_PI_F / 180.0f);
	float u_len = aspect * v_len;

	float3 Z = normalize(look_at - eye);
	float3 X = normalize(cross(up, Z));
	float3 Y = cross(Z, X);

	cam.pos = eye;
    cam.dir = Z;
    cam.up = Y * v_len;
	cam.right = X * u_len;
}

void trace_scene(TracerState* state, const Camera& cam, float* framebuffer, uint32_t width, uint32_t height, cudaTextureObject_t bg_tex)
{	    
    launchKernel(width * height, generate_photons, cam, width, height, state->points, state->velocity);

	SceneState scene;
	scene.accretion_disk = Cylinder::construct(float3{ 0, -0.1, 0 }, float3{ 0, 0.1, 0 }, 14, 3);
	scene.bg_tex = bg_tex;

    for(uint32_t iter = 0; iter < 50; iter++)
    {
        launchKernel(width * height, integrate_photons, width * height, 10, 0.05f, state->points, state->velocity, scene);
    }

	launchKernel(width * height, trace_photons, width * height, state->points, state->velocity, state->radiance, scene);
    launchKernel(width * height, splat_photons, state->device_framebuffer, width, height, state->radiance);
       
    cudaMemcpy(framebuffer, state->device_framebuffer, width * height * sizeof(float3), cudaMemcpyDeviceToHost);
}
