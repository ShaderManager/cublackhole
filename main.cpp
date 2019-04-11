#include <stdio.h>
#include <string>

#define cimg_use_jpeg 1
#include "CImg.h"

#include <cuda_runtime.h>

struct TracerState;

void init_tracer_state(TracerState** state, uint32_t width, uint32_t height);
void destroy_tracer_state(TracerState* state);
void trace_scene(TracerState* state, float* framebuffer, uint32_t width, uint32_t height, cudaTextureObject_t, uint32_t frame);

bool load_texture_to_cuda(const char* filename, cudaTextureObject_t& tex_object)
{
	using namespace cimg_library;

	try
	{
		CImg<unsigned char> image(filename);
		CImg<unsigned char> image4(image.width(), image.height(), 1, 4);

		// copy and pad 4th channel
		cimg_forXY(image, x, y)
		{
			image4(x, y, 0, 0) = image(x, y, 0, 0);
			image4(x, y, 0, 1) = image(x, y, 0, 1);
			image4(x, y, 0, 2) = image(x, y, 0, 2);
			image4(x, y, 0, 3) = 255;
		}

		void* image_data = nullptr;
		size_t pitch;
		size_t src_pitch = image4.width() * 4;

		cudaMallocPitch(&image_data, &pitch, src_pitch, image4.height());

		// Describe as pitch2D resource
		cudaResourceDesc res_desc;
		memset(&res_desc, 0, sizeof(res_desc));
		res_desc.resType = cudaResourceTypePitch2D;
		res_desc.res.pitch2D.width = image4.width();
		res_desc.res.pitch2D.height = image4.height();
		res_desc.res.pitch2D.pitchInBytes = pitch;
		res_desc.res.pitch2D.desc = cudaCreateChannelDesc<uchar4>();

		image4.permute_axes("cxyz"); // Swizzle to interleaved order

		cudaMemcpy2D(image_data, pitch, image4.data(), src_pitch, src_pitch, res_desc.res.pitch2D.height, cudaMemcpyHostToDevice);
		res_desc.res.pitch2D.devPtr = image_data;

		cudaTextureDesc tex_desc;
		memset(&tex_desc, 0, sizeof(tex_desc));
		tex_desc.readMode = cudaReadModeElementType;
		tex_desc.normalizedCoords = true; // Use texcoords in [0, 1] range

		tex_desc.addressMode[0] = cudaAddressModeWrap;
		tex_desc.addressMode[1] = cudaAddressModeWrap;

		cudaError_t error = cudaCreateTextureObject(&tex_object, &res_desc, &tex_desc, nullptr);

		if (error != cudaSuccess)
		{
			printf("%s: %s\n", cudaGetErrorName(error), cudaGetErrorString(error));
			return false;
		}
	}
	catch (CImgIOException& exc)
	{
		printf("Error: %s\n", exc.what());
		return false;
	}

	return true;
}

int main(int arg, char** argv)
{
	using namespace cimg_library;

	unsigned int fb_width = 1024;
	unsigned int fb_height = 768;

	cudaTextureObject_t background_texture = 0;

	load_texture_to_cuda("data/background.jpg", background_texture);

	float* framebuffer = new float[fb_width * fb_height * 3];
	
	CImgDisplay display(fb_width, fb_height, 0);

	TracerState* state = nullptr;
	init_tracer_state(&state, fb_width, fb_height);

	uint32_t frame = 0;
	while (!display.is_closed())
	{
		memset(framebuffer, 0, fb_width * fb_height * 3 * sizeof(float));

		trace_scene(state, framebuffer, fb_width, fb_height, background_texture, frame++);

		CImg<float> cimg_fb(framebuffer, 3, fb_width, fb_height, 1, true);
		cimg_fb.permute_axes("yzcx");

		//cimg_rof(cimg_fb, ptr, float) *ptr *= 255.f;

		display.set_title(std::to_string(frame).c_str());
		display.render(cimg_fb);
		display.paint();
	}

	destroy_tracer_state(state);

	return 0;
}