#pragma once

struct Camera
{
	float3 pos;
	float3 dir;
	float3 up;
	float3 right;
};

struct TracerState;

void init_tracer_state(TracerState** state, uint32_t width, uint32_t height);
void destroy_tracer_state(TracerState* state);

void setup_camera(Camera& cam, const float3 eye, const float3 look_at, const float3 up, float fov, float aspect);

void trace_scene(TracerState* state, const Camera& cam, float* framebuffer, uint32_t width, uint32_t height, cudaTextureObject_t bg_tex);
