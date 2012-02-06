/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// Utilities and system includes
#include <cuda.h>
#include <cuda_runtime_api.h>


#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdint.h>
#include "dxt_cuda.h"
__constant__ int constAlpha = 0xff << 24;
//__constant__ float  constHueColorSpaceMat[9];

//#define NUM_THREADS 256
//#define MUL(x,y)    (x*y)


/* cuda debug mode */
//#define DEBUG_MODE 
#define _DEBUG
cudaError_t error;

//#define NUM_THREADS 64        // Number of threads per block.


__device__ void YUV2RGB(unsigned int *yuvi, float *red, float *green, float *blue, float *constHueColorSpaceMat)
{
        float luma, chromaCb, chromaCr;

//	const int bid = ( blockIdx.x ) * blockDim.x ;
//      const int idx = threadIdx.x;


        // Prepare for hue adjustment
        luma     = (float)yuvi[0];
        chromaCb = (float)((int)yuvi[1] - 512.0f);
        chromaCr = (float)((int)yuvi[2] - 512.0f);

        // Convert YUV To RGB with hue adjustment
        *red  = MUL(luma,     constHueColorSpaceMat[0]) +
                MUL(chromaCb, constHueColorSpaceMat[1]) +
                MUL(chromaCr, constHueColorSpaceMat[2]);
        *green= MUL(luma,     constHueColorSpaceMat[3]) +
                MUL(chromaCb, constHueColorSpaceMat[4]) +
                MUL(chromaCr, constHueColorSpaceMat[5]);
        *blue = MUL(luma,     constHueColorSpaceMat[6]) +
                MUL(chromaCb, constHueColorSpaceMat[7]) +
                MUL(chromaCr, constHueColorSpaceMat[8]);
	

}

__device__ void YUV2RGB8(unsigned int *yuvi, unsigned int *red, unsigned int *green, unsigned int *blue, float *constHueColorSpaceMat)
{


	*red = ((76284*(yuvi[0]-16)+104595*(yuvi[2]-128)) >> 16);
	*green = ((76284*(yuvi[0]-16)-53281*(yuvi[2]-128)-25625*(yuvi[1]-128)) >> 16);
	*blue = ((76284*(yuvi[0]-16) +132252*(yuvi[1]-128)) >> 16);
}


__device__ unsigned int RGBAPACK_10bit(float red, float green, float blue, unsigned int alpha)
{
        unsigned int ARGBpixel = 0;

        // Clamp final 10 bit results
        red   = min(max(red,   0.0f), 1023.f);
        green = min(max(green, 0.0f), 1023.f);
        blue  = min(max(blue,  0.0f), 1023.f);

        // Convert to 8 bit unsigned integers per color component
	  ARGBpixel = (((unsigned int)blue  >> 2) |
                        (((unsigned int)green >> 2) << 8)  |
                        (((unsigned int)red   >> 2) << 16) );


        return  ARGBpixel;
}

__device__ unsigned int RGBAPACK_8bit(int red, int green, int blue, int alpha)
{
    unsigned int ARGBpixel = 0;

   // Clamp final 10 bit results
   red   = min(max(red,   0), 255);
   green = min(max(green, 0), 255);
   blue  = min(max(blue,  0), 255);


    // Convert to 8 bit unsigned integers per color component
    ARGBpixel = (((unsigned int)blue ) | 
                (((unsigned int)green) << 8)  | 
                (((unsigned int)red  ) << 16)); // | (unsigned int)alpha);

    return  ARGBpixel;

}

__device__ unsigned int RGBAPACK(float red, float green, float blue)
{
    unsigned int ARGBpixel = 0;

   // Clamp final 10 bit results
   red   = min(max((int)red,   0), 255);
   green = min(max((int)green, 0), 255);
   blue  = min(max((int)blue,  0), 255);


    // Convert to 8 bit unsigned integers per color component
    ARGBpixel = (((unsigned int)blue ) | 
                (((unsigned int)green) << 8)  | 
                (((unsigned int)red  ) << 16)); // | (unsigned int)alpha);

    return  ARGBpixel;

}

__global__ void uyvy2rgb(const uint *image, uint *dstImage, float *hue, int blockOffset)
{

        const int bid = ( blockIdx.x + blockOffset ) * blockDim.x ;
        const int idx = threadIdx.x;
	__shared__ float constHueColorSpaceMat[9];
	for (int i=0;i<9;i++) constHueColorSpaceMat[i]=hue[i];

        uint yuvi[6], yuvpack[2];
        float red[2], green[2], blue[2];

	yuvpack[0] =  ((image[(bid+idx)] >> 8) & 0xff) << 2;
	yuvpack[0] |=  ((image[(bid+idx)]) & 0xff) << 12;
	yuvpack[0] |=  ((image[(bid+idx)] >> 16) & 0xff) << 22;

	yuvpack[1] =  ((image[(bid+idx)] >> 24) & 0xff) << 2;
	yuvpack[1] |=  ((image[(bid+idx)]) & 0xff) << 12;
	yuvpack[1] |=  ((image[(bid+idx)] >> 16) & 0xff) << 22;

        yuvi[0] = (yuvpack[0] & 0x3ff);
        yuvi[1] = (yuvpack[0] >> 10) & 0x3ff;
        yuvi[2] = (yuvpack[0] >> 20) & 0x3ff;
        yuvi[3] = (yuvpack[1] & 0x3ff);
        yuvi[4] = (yuvpack[1] >> 10) & 0x3ff;
        yuvi[5] = (yuvpack[1] >> 20) & 0x3ff;

        YUV2RGB(&yuvi[0], &red[0], &green[0], &blue[0], constHueColorSpaceMat);
        YUV2RGB(&yuvi[3], &red[1], &green[1], &blue[1], constHueColorSpaceMat);

        dstImage[(bid+idx)*2] = RGBAPACK_10bit(red[0], green[0], blue[0], constAlpha);
        dstImage[(bid+idx)*2+1] = RGBAPACK_10bit(red[1], green[1], blue[1], constAlpha);


}


__global__ void uyvy2rgba(const uint *image, uint *dstImage, float *hue, int blockOffset)
{

        const int bid = ( blockIdx.x + blockOffset ) * blockDim.x ;
        const int idx = threadIdx.x;

        uint c = image[(bid+idx)];
        int yuvi[4];
        int red[2], green[2], blue[2];

        yuvi[0] = ( (c >> 8) & 0xff) - 16;	// y1
        yuvi[1] = ( (c >> 0) & 0xff) - 128;	// u	
        yuvi[2] = ( (c >> 16) & 0xff) - 128;	// v
        yuvi[3] = ( (c >> 24) & 0xff) - 16;	// y2

	red[0] = (yuvi[0] + (1.5958 * yuvi[2])) * 0.859375;
	green[0] = (yuvi[0] - (0.81290 * yuvi[2]) - (0.39173 * yuvi[1])) * 0.859375;
	blue[0] = (yuvi[0] + (2.017 * yuvi[1])) * 0.859375;

	red[1] = (yuvi[3] + (1.5958 * yuvi[2])) * 0.859375;
	green[1] = (yuvi[3] - (0.81290 * yuvi[2]) - (0.39173 * yuvi[1])) * 0.859375;
	blue[1] = (yuvi[3] + (2.017 * yuvi[1])) * 0.859375;

        dstImage[(bid+idx)*2] = RGBAPACK(red[0], green[0], blue[0]);
        dstImage[(bid+idx)*2+1] = RGBAPACK(red[1], green[1], blue[1]);



}

////////////////////////////////////////////////////////////////////////////////
// Load color block to shared mem
////////////////////////////////////////////////////////////////////////////////
__device__ void loadColorBlock(const uint * image, uchar3 colors[NUM_THREADS][16], int blockOffset)
{
    const int bid = ( blockIdx.x + blockOffset ) * blockDim.x ;
    const int idx = threadIdx.x;
    const int w = 1920;

// i is height each DXT block
// each DXT block index is equal bid + idx

    for(int i=0;i<4;i++) {
	for(int j=0;j<4;j++) {

	/* jyyoo-version (RGBA) */
	colors[idx][4*i+j].x = ((image[(((bid+idx)/(w>>2))*4+i)*w + ((bid+idx)%(w>>2)) * 4 + j] >> 0) & 0xff);
	colors[idx][4*i+j].y = ((image[(((bid+idx)/(w>>2))*4+i)*w + ((bid+idx)%(w>>2)) * 4 + j] >> 8) & 0xff);
	colors[idx][4*i+j].z = ((image[(((bid+idx)/(w>>2))*4+i)*w + ((bid+idx)%(w>>2)) * 4 + j] >> 16) & 0xff);

	/* lucas-version  (ARGB)
	colors[idx][4*i+j].x = ((image[(((bid+idx)/(w>>2))*4+i)*w + ((bid+idx)%(w>>2)) * 4 + j] >> 16) & 0xff);
	colors[idx][4*i+j].y = ((image[(((bid+idx)/(w>>2))*4+i)*w + ((bid+idx)%(w>>2)) * 4 + j] >> 8) & 0xff);
	colors[idx][4*i+j].z = ((image[(((bid+idx)/(w>>2))*4+i)*w + ((bid+idx)%(w>>2)) * 4 + j] >> 0) & 0xff);
	*/
	}

   }
        
        // No need to synchronize, 16 < warp size.
}

////////////////////////////////////////////////////////////////////////////////
// Round color to RGB565 and expand
////////////////////////////////////////////////////////////////////////////////

__device__ void round565(uchar3 color_a[NUM_THREADS], uchar3 color_b[NUM_THREADS], uint2 *result, int blockOffset)
{
    // a = min, b= max
    //const int bid = blockIdx.x * blockDim.x + blockOffset;
    
    const int bid = ( blockIdx.x + blockOffset ) * blockDim.x ;
    const int idx = threadIdx.x;

    ushort2 temp;
    temp.x = ( (color_a[idx].x >> 3) << 11) | ( (color_a[idx].y >> 2) << 5) | (color_a[idx].z >> 3);
    temp.y = ( (color_b[idx].x >> 3) << 11) | ( (color_b[idx].y >> 2) << 5) | (color_b[idx].z >> 3);

    //result[bid].x= (temp.x<< 16) | temp.y;
    result[bid+idx].x= (temp.y<< 16) | temp.x;

}

__device__ void GetMinMaxColorByBBox(uchar3 colors[NUM_THREADS][16], uchar3 min[NUM_THREADS], uchar3 max[NUM_THREADS], int blockOffset)
{

    //const int bid = blockIdx.x * blockDim.x + blockOffset;
    const int idx = threadIdx.x;

    min[idx].x = min[idx].y = min[idx].z = 255;
    max[idx].x = max[idx].y = max[idx].z = 0;

    for (int i=0; i<16; i++) {
	if (colors[idx][i].x < min[idx].x) min[idx].x = colors[idx][i].x;
	if (colors[idx][i].y < min[idx].y) min[idx].y = colors[idx][i].y;
	if (colors[idx][i].z < min[idx].z) min[idx].z = colors[idx][i].z;

	if (colors[idx][i].x > max[idx].x) max[idx].x = colors[idx][i].x;
	if (colors[idx][i].y > max[idx].y) max[idx].y = colors[idx][i].y;
	if (colors[idx][i].z > max[idx].z) max[idx].z = colors[idx][i].z;

    }
	
    uchar3 inset;
    inset.x = (max[idx].x - min[idx].x) >> 4;
    inset.y = (max[idx].y - min[idx].y) >> 4;
    inset.z = (max[idx].z - min[idx].z) >> 4;

    min[idx].x = ( min[idx].x + inset.x <= 255) ? min[idx].x + inset.x : 255;
    min[idx].y = ( min[idx].y + inset.y <= 255) ? min[idx].y + inset.y : 255;
    min[idx].z = ( min[idx].z + inset.z <= 255) ? min[idx].z + inset.z : 255;
    max[idx].x = ( max[idx].x >= inset.x) ? max[idx].x - inset.x : 0;
    max[idx].y = ( max[idx].y >= inset.y) ? max[idx].y - inset.y : 0;
    max[idx].z = ( max[idx].z >= inset.z) ? max[idx].z - inset.z : 0;

}

__device__ void EmitColorIndices(uchar3 colors[NUM_THREADS][16], uchar3 min[NUM_THREADS], uchar3 max[NUM_THREADS], uint2* result, int blockOffset)
{
	//const int bid = blockIdx.x * blockDim.x + blockOffset;

        const int bid = ( blockIdx.x + blockOffset ) * blockDim.x ;
	const int idx = threadIdx.x;
	ushort3 dxt[4];
	uint temp = 0; 

	// c565_5 0xf8
	// c565_6 0xfc

	dxt[0].x = (max[idx].x & 0xf8) | (max[idx].x >> 5);
	dxt[0].y = (max[idx].y & 0xfc) | (max[idx].y >> 6);
	dxt[0].z = (max[idx].z & 0xf8) | (max[idx].z >> 5);
	dxt[1].x = (min[idx].x & 0xf8) | (min[idx].x >> 5);
	dxt[1].y = (min[idx].y & 0xfc) | (min[idx].y >> 6);
	dxt[1].z = (min[idx].z & 0xf8) | (min[idx].z >> 5);

	dxt[2].x = ( 2*dxt[0].x + 1*dxt[1].x) /3;
	dxt[2].y = ( 2*dxt[0].y + 1*dxt[1].y) /3;
	dxt[2].z = ( 2*dxt[0].z + 1*dxt[1].z) /3;
	dxt[3].x = ( 1*dxt[0].x + 2*dxt[1].x) /3;
	dxt[3].y = ( 1*dxt[0].y + 2*dxt[1].y) /3;
	dxt[3].z = ( 1*dxt[0].z + 2*dxt[1].z) /3;

	for (int i=15; i>=0; i--) {
		int c0 = colors[idx][i].x;
		int c1 = colors[idx][i].y;
		int c2 = colors[idx][i].z;
		int d0 = abs(dxt[0].x - c0) + abs(dxt[0].y - c1) + abs(dxt[0].z -c2);
		int d1 = abs(dxt[1].x - c0) + abs(dxt[1].y - c1) + abs(dxt[1].z -c2);
		int d2 = abs(dxt[2].x - c0) + abs(dxt[2].y - c1) + abs(dxt[2].z -c2);
		int d3 = abs(dxt[3].x - c0) + abs(dxt[3].y - c1) + abs(dxt[3].z -c2);

		int b0 = d0 > d3;
		int b1 = d1 > d2;
		int b2 = d0 > d2;
		int b3 = d1 > d3;
		int b4 = d2 > d3;

		int x0 = b1 & b2;
		int x1 = b0 & b3;
		int x2 = b0 & b4;
		
		temp |= ( x2 | ( (x0 | x1) << 1) ) << (i<<1);
	}

	result[bid+idx].y = temp;	
}

////////////////////////////////////////////////////////////////////////////////
// Compress color block
////////////////////////////////////////////////////////////////////////////////
__global__ void ckernel_dxt_compress_pixel_blocks(uint32_t *input, uint16_t input_format, 
		uint2 *output, uint16_t output_format, uint32_t total_dxt_blocks)
{
	// FIXME: the amount of allocated memory of shared and registers should be balanced
	// for registers, don't forget to count constants.
	// 1. check how much memory amount of both shared and registers are in the GPU
	// 2. proportionally balance between shared and registers
	// 3. try to reuse memory


	// FIXME: float* hue is a constant. map it to the constant or texture memory.
//	float hue[9];
//	float hueSin = 0; //sin(0.0);
//	float hueCos = 1; //cos(0.0);

//	hue[0] = 1.1644f;
//	hue[1] = hueSin * 1.5960f;
//	hue[2] = hueCos * 1.5960f;
//	hue[3] = 1.1644f;
//	hue[4] = (hueCos * -0.3918f) - (hueSin * 0.8130f);
//	hue[5] = (hueSin *  0.3918f) - (hueCos * 0.8130f);
//	hue[6] = 1.1644f;
//	hue[7] = hueCos *  2.0172f;
//	hue[8] = hueSin * -2.0172f;

//	const int picture_width = 1920;
	// a thread takes care of a DXT-block (4x4 pixels)
	int gid = blockDim.x * blockIdx.x + threadIdx.x; // better to be register
	int tid = threadIdx.x;
	uint32_t tmp_image;
	int i;


	if( gid >= total_dxt_blocks ) {
		
		return;
	}

	// XXX: Note that in this parallization, we don't need to use shared memory 
	// since each thread does not share any information
	__shared__ uchar3 colors[NUM_THREADS][16];
	__shared__ uchar3 min[NUM_THREADS];
	__shared__ uchar3 max[NUM_THREADS];

	int pxs = 16;
	if( input_format == 0 ) pxs = 16;
	if( input_format == 1 ) pxs = 8;

	// copy image to colors
	for(i=0;i<pxs;i++) 
	{
		tmp_image = input[gid*pxs + i];

		if( input_format == 1 /*YUV*/) {
			// tmp_image stores 2 pixels

			// perform RGB converting
			int yuvi[4];
//			int red[2], green[2], blue[2];

			yuvi[0] = ( (tmp_image >> 8) & 0xff) - 16;	// y1
			yuvi[1] = ( (tmp_image >> 0) & 0xff) - 128;	// u	
			yuvi[2] = ( (tmp_image >> 16) & 0xff) - 128;	// v
			yuvi[3] = ( (tmp_image >> 24) & 0xff) - 16;	// y2

			colors[tid][i*2].x = (yuvi[0] + (1.5958 * yuvi[2])) * 0.859375;
			colors[tid][i*2].y = (yuvi[0] - (0.81290 * yuvi[2]) - (0.39173 * yuvi[1])) * 0.859375;
			colors[tid][i*2].z = (yuvi[0] + (2.017 * yuvi[1])) * 0.859375;

			colors[tid][i*2+1].x = (yuvi[3] + (1.5958 * yuvi[2])) * 0.859375;
			colors[tid][i*2+1].y = (yuvi[3] - (0.81290 * yuvi[2]) - (0.39173 * yuvi[1])) * 0.859375;
			colors[tid][i*2+1].z = (yuvi[3] + (2.017 * yuvi[1])) * 0.859375;
		} else if( input_format == 0 /* RGBA */ ) {

			colors[tid][i].x = (tmp_image >> 0) & 0xff;
			colors[tid][i].y = (tmp_image >> 8) & 0xff;
			colors[tid][i].z = (tmp_image >> 16) & 0xff;
		}
	}

	// find min and max color
	min[tid].x = min[tid].y = min[tid].z = 255;
	max[tid].x = max[tid].y = max[tid].z = 0;

	for (i=0; i<16; i++) {
		if (colors[tid][i].x < min[tid].x) min[tid].x = colors[tid][i].x;
		if (colors[tid][i].y < min[tid].y) min[tid].y = colors[tid][i].y;
		if (colors[tid][i].z < min[tid].z) min[tid].z = colors[tid][i].z;

		if (colors[tid][i].x > max[tid].x) max[tid].x = colors[tid][i].x;
		if (colors[tid][i].y > max[tid].y) max[tid].y = colors[tid][i].y;
		if (colors[tid][i].z > max[tid].z) max[tid].z = colors[tid][i].z;
	}

	uchar3 inset;
	inset.x = (max[tid].x - min[tid].x) >> 4;
	inset.y = (max[tid].y - min[tid].y) >> 4;
	inset.z = (max[tid].z - min[tid].z) >> 4;

	min[tid].x = ( min[tid].x + inset.x <= 255) ? min[tid].x + inset.x : 255;
	min[tid].y = ( min[tid].y + inset.y <= 255) ? min[tid].y + inset.y : 255;
	min[tid].z = ( min[tid].z + inset.z <= 255) ? min[tid].z + inset.z : 255;
	max[tid].x = ( max[tid].x >= inset.x) ? max[tid].x - inset.x : 0;
	max[tid].y = ( max[tid].y >= inset.y) ? max[tid].y - inset.y : 0;
	max[tid].z = ( max[tid].z >= inset.z) ? max[tid].z - inset.z : 0;


	// round the color to RGB565 and expand
	ushort2 temp;
	temp.x = ( (max[tid].x >> 3) << 11) | ( (max[tid].y >> 2) << 5) | (max[tid].z >> 3);
	temp.y = ( (min[tid].x >> 3) << 11) | ( (min[tid].y >> 2) << 5) | (min[tid].z >> 3);

	output[gid].x = (temp.y<< 16) | temp.x;

	
	// emit color index

	ushort3 dxt[4];

	dxt[0].x = (max[tid].x & 0xf8) | (max[tid].x >> 5);
	dxt[0].y = (max[tid].y & 0xfc) | (max[tid].y >> 6);
	dxt[0].z = (max[tid].z & 0xf8) | (max[tid].z >> 5);
	dxt[1].x = (min[tid].x & 0xf8) | (min[tid].x >> 5);
	dxt[1].y = (min[tid].y & 0xfc) | (min[tid].y >> 6);
	dxt[1].z = (min[tid].z & 0xf8) | (min[tid].z >> 5);

	dxt[2].x = ( 2*dxt[0].x + 1*dxt[1].x) /3;
	dxt[2].y = ( 2*dxt[0].y + 1*dxt[1].y) /3;
	dxt[2].z = ( 2*dxt[0].z + 1*dxt[1].z) /3;
	dxt[3].x = ( 1*dxt[0].x + 2*dxt[1].x) /3;
	dxt[3].y = ( 1*dxt[0].y + 2*dxt[1].y) /3;
	dxt[3].z = ( 1*dxt[0].z + 2*dxt[1].z) /3;

	uint tmp;
	int c0, c1, c2;
	int d0, d1, d2, d3;
	int b0, b1, b2, b3, b4;
	int x0, x1, x2;
	for (int i=15; i>=0; i--) {
		c0 = colors[tid][i].x;
		c1 = colors[tid][i].y;
		c2 = colors[tid][i].z;
		d0 = abs(dxt[0].x - c0) + abs(dxt[0].y - c1) + abs(dxt[0].z -c2);
		d1 = abs(dxt[1].x - c0) + abs(dxt[1].y - c1) + abs(dxt[1].z -c2);
		d2 = abs(dxt[2].x - c0) + abs(dxt[2].y - c1) + abs(dxt[2].z -c2);
		d3 = abs(dxt[3].x - c0) + abs(dxt[3].y - c1) + abs(dxt[3].z -c2);

		b0 = d0 > d3;
		b1 = d1 > d2;
		b2 = d0 > d2;
		b3 = d1 > d3;
		b4 = d2 > d3;

		x0 = b1 & b2;
		x1 = b0 & b3;
		x2 = b0 & b4;
		
		tmp |= ( x2 | ( (x0 | x1) << 1) ) << (i<<1);
	}

	output[gid].y = tmp;	
}


////////////////////////////////////////////////////////////////////////////////
// compress color block
////////////////////////////////////////////////////////////////////////////////
__global__ void compress(const uint * image, uint2 * result, int blockOffset)
{

	__shared__ uchar3 colors[NUM_THREADS][16];
	__shared__ uchar3 min[NUM_THREADS], max[NUM_THREADS];

	loadColorBlock(image, colors, blockOffset);

	__syncthreads();

	// find min and max color
	GetMinMaxColorByBBox(colors, min, max, blockOffset);

	__syncthreads();

	round565(max, min, result, blockOffset);
	//EmitColorIndices(colors, min, max, dxt, result, blockOffset);
	EmitColorIndices(colors, min, max, result, blockOffset);

	__syncthreads();
}

extern "C" void CompressDXT2(int test)
{
	printf("hello, cuda! (%d)\n", test);
}

extern "C" void dxt_compress_pixel_blocks(uint32_t *d_data, uint32_t *d_result, uint32_t total_dxt_blocks)
{	

//	int j;
	uint32_t total_cuda_blocks = total_dxt_blocks / NUM_THREADS;
	if( total_dxt_blocks % NUM_THREADS ) total_cuda_blocks ++;

	ckernel_dxt_compress_pixel_blocks<<<total_cuda_blocks, NUM_THREADS>>>(d_data, 0, (uint2*)d_result, 0, total_dxt_blocks);
}

extern "C" void dxt_compress_yuv_pixel_blocks(uint32_t *d_data, uint32_t *d_result, uint32_t total_dxt_blocks)
{	

//	int j;
	uint32_t total_cuda_blocks = total_dxt_blocks / NUM_THREADS;
	if( total_dxt_blocks % NUM_THREADS ) total_cuda_blocks ++;

	ckernel_dxt_compress_pixel_blocks<<<total_cuda_blocks, NUM_THREADS>>>(d_data, 1, (uint2*)d_result, 0, total_dxt_blocks);

	cudaThreadSynchronize();

}

extern "C" void CompressDXT(uint *d_data, uint *d_rgba, uint *d_result, uint blocks4yuv, uint blocksPerLaunch4yuv, uint blocks4dxt, uint blocksPerLaunch4dxt, uint compressedSize, float *HueColorSpaceMat)
{	

	int j;
	if( d_data && d_rgba == NULL )
	{
		for(j=0; j<blocks4yuv; j+=blocksPerLaunch4yuv ) {
			uyvy2rgba<<<min(blocksPerLaunch4yuv, blocks4yuv-j), NUM_THREADS>>>(d_data, d_rgba, HueColorSpaceMat, j);
		}
	}


	printf("calling CUDA kernel blocksPerLaunch4dxt=%d, blocks4dxt=%d num_of_threads=%d \n",
			blocksPerLaunch4dxt,
			blocks4dxt,
			NUM_THREADS
			);
	compress<<<min(blocksPerLaunch4dxt, blocks4dxt-j), NUM_THREADS>>>(d_rgba, (uint2 *)d_result, j);
}

    
