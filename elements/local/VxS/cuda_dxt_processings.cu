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

/* cuda debug mode */
//#define DEBUG_MODE 
#define _DEBUG
cudaError_t error;

#define IMAGE_WIDTH 1920
#define IMAGE_HEIGHT 1080

/******************************************************************************************************* 
 * convert from yuv2 to rgb4
 ******************************************************************************************************/
__global__ void ckernel_convert_from_yuv2_to_rgb4(uint32_t *input, uint32_t *output, uint32_t total_dxt_blocks)
{
	// FIXME: the amount of allocated memory of shared and registers should be balanced
	// for registers, don't forget to count constants.
	// 1. check how much memory amount of both shared and registers are in the GPU
	// 2. proportionally balance between shared and registers
	// 3. try to reuse memory

	// a thread takes care of a DXT-block (4x4 pixels)
	int gid = blockDim.x * blockIdx.x + threadIdx.x; // better to be register
	uint32_t yuv2_2pixels;
	uint32_t rgba_1pixel;
	uint8_t tmp;
	int i;

	if( gid >= total_dxt_blocks ) {
		
		return;
	}

	int pxs = 8; // pixels in a dxt block (4x4)

	// copy image to colors
	for(i=0;i<pxs;i++) 
	{
		/* yuv2_2pixels is 4 byte (2 pixels) */
		yuv2_2pixels = input[gid*pxs + i];

		// perform RGB converting
		int yuvi[4];

		yuvi[0] = ( (yuv2_2pixels >> 8) & 0xff) - 16;	// y1
		yuvi[1] = ( (yuv2_2pixels >> 0) & 0xff) - 128;	// u	
		yuvi[2] = ( (yuv2_2pixels >> 16) & 0xff) - 128;	// v
		yuvi[3] = ( (yuv2_2pixels >> 24) & 0xff) - 16;	// y2

		/* first pixel */
		rgba_1pixel = 0x00000000;
		rgba_1pixel = rgba_1pixel | 0xffu << 24; 		/* alpha */
		tmp = (yuvi[0] + (2.017 * yuvi[1])) * 0.859375;
		rgba_1pixel = rgba_1pixel | tmp << 16;			/* red (R) */
		tmp = (yuvi[0] - (0.81290 * yuvi[2]) - (0.39173 * yuvi[1])) * 0.859375;
		rgba_1pixel = rgba_1pixel | tmp << 8;			/* green (G) */
		tmp = (yuvi[0] + (1.5958 * yuvi[2])) * 0.859375;
		rgba_1pixel = rgba_1pixel | tmp << 0;			/* blue (B) */

		output[gid * pxs + i * 2 + 0] = rgba_1pixel;

		/* first pixel */
		rgba_1pixel = 0x00000000;
		rgba_1pixel = rgba_1pixel | 0xffu << 24; 		/* alpha */
		tmp = (yuvi[3] + (2.017 * yuvi[1])) * 0.859375;
		rgba_1pixel = rgba_1pixel | tmp << 16;			/* red (R) */
		tmp = (yuvi[3] - (0.81290 * yuvi[2]) - (0.39173 * yuvi[1])) * 0.859375;
		rgba_1pixel = rgba_1pixel | tmp << 8;			/* green (G) */
		tmp = (yuvi[3] + (1.5958 * yuvi[2])) * 0.859375;
		rgba_1pixel = rgba_1pixel | tmp << 0;			/* blue (B) */

		output[gid * pxs + i * 2 + 1] = rgba_1pixel;
	}
}

extern "C" void dxt_convert_from_yuv2_to_rgb4(uint32_t *d_data, uint32_t *d_result, uint32_t total_dxt_blocks)
{	
	uint32_t total_cuda_blocks = total_dxt_blocks / NUM_THREADS;
	if( total_dxt_blocks % NUM_THREADS ) total_cuda_blocks ++;

	ckernel_convert_from_yuv2_to_rgb4<<<total_cuda_blocks, NUM_THREADS>>>(d_data, d_result, total_dxt_blocks);

	cudaThreadSynchronize();
}


/******************************************************************************************************* 
 * compress from yuv 2 
 ******************************************************************************************************/
__global__ void ckernel_dxt_compress_from_yuv2_or_rgb4(uint32_t *input, uint16_t input_format, 
		uint2 *output, uint16_t output_format, uint32_t total_dxt_blocks)
{
	// FIXME: the amount of allocated memory of shared and registers should be balanced
	// for registers, don't forget to count constants.
	// 1. check how much memory amount of both shared and registers are in the GPU
	// 2. proportionally balance between shared and registers
	// 3. try to reuse memory

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
	if( input_format == 0 ) pxs = 16; /* input format is RGB4 */
	if( input_format == 1 ) pxs = 8; /* input format is YUV2 */

	// copy image to colors
	for(i=0;i<pxs;i++) 
	{
		tmp_image = input[gid*pxs + i];

		if( input_format == 1 /*YUV*/) {
			// tmp_image stores 2 pixels

			// perform RGB converting
			int yuvi[4];

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

extern "C" void dxt_compress_from_yuv2(uint32_t *d_data, uint32_t *d_result, uint32_t total_dxt_blocks)
{	
	uint32_t total_cuda_blocks = total_dxt_blocks / NUM_THREADS;
	if( total_dxt_blocks % NUM_THREADS ) total_cuda_blocks ++;

	ckernel_dxt_compress_from_yuv2_or_rgb4<<<total_cuda_blocks, NUM_THREADS>>>(d_data, 
			0, /* 1 indicates yuv2; 0 indicates rgb4 */
			(uint2*)d_result, 
			0, 
			total_dxt_blocks);

	cudaThreadSynchronize();
}

/******************************************************************************************************* 
 * decompress to rgb4
 ******************************************************************************************************/
__device__ void loadColorBlock(const uint * image, uint2 dxtBlocks[NUM_THREADS], int blockOffset, uint limit)
{
	const int bid = ( blockIdx.x + blockOffset ) * blockDim.x;
      	const int idx = threadIdx.x;

	if ((bid+threadIdx.x) >= limit) return;

	dxtBlocks[idx].x = image[(bid+idx)*2];
	dxtBlocks[idx].y = image[(bid+idx)*2+1];
}

__device__ void decodeDXT1(uint2 dxtBlocks[NUM_THREADS], uint rgbaBlocks[NUM_THREADS][16], int blockOffset, uint limit )
{
	const int bid = ( blockIdx.x + blockOffset ) * blockDim.x ;
	const int idx = threadIdx.x;
	
	if ((bid+threadIdx.x) >= limit) return;
	uchar3 col0, col1;
	ushort2 color;
	uint palette[4];

	// divide colors from 32bits to two 16bits colors 
	color.y = (dxtBlocks[idx].x >> 16) & 0xffff;
	color.x = dxtBlocks[idx].x & 0xffff;

	// Does bit expansion before interpolation.
	col0.x = (color.x >> 11) & 0x1f;
	col0.y = (color.x >> 5) & 0x3f;
	col0.z = (color.x >> 0) & 0x1f;

	col0.x = (col0.x << 3) | (col0.x >> 2);
	col0.y = (col0.y << 2) | (col0.y >> 4);
	col0.z = (col0.z << 3) | (col0.z >> 2);

	palette[0] = (0xFF << 24) | (col0.x << 16)| (col0.y << 8)| (col0.z);

	col1.x = (color.y >> 11) & 0x1f;
	col1.y = (color.y >> 5) & 0x3f;
	col1.z = (color.y >> 0) & 0x1f;

	col1.x = (col1.x << 3) | (col1.x >> 2);
	col1.y = (col1.y << 2) | (col1.y >> 4);
	col1.z = (col1.z << 3) | (col1.z >> 2);

	palette[1] = (0xFF << 24) | (col1.x << 16)| (col1.y << 8)| (col1.z);

	if( palette[0] > palette[1] ) {
		palette[2] = (0xFF << 24) | 
			((((2 * col0.x + col1.x) / 3) & 0xff) << 16)| 
			((((2 * col0.y + col1.y) / 3) & 0xff) << 8)| 
			((((2 * col0.z + col1.z) / 3) & 0xff )); 

		palette[3] = (0xFF << 24) | 
			((((2 * col1.x + col0.x) / 3) & 0xff) << 16)| 
			((((2 * col1.y + col0.y) / 3) & 0xff) << 8)| 
			((((2 * col1.z + col0.z) / 3) & 0xff)); 
	}
	else {
		// Three-color block: derive the other color.
		palette[2] = (0xFF << 24) | 
			((((col0.x + col1.x +1) / 2) & 0xff) << 16)| 
			((((col0.y + col1.y +1) / 2) & 0xff) << 8)| 
			((((col0.z + col1.z +1) / 2) & 0xff)); 

		palette[3]  = 0x00000000;
	}

	for(int i=0;i<16;i++) {
             rgbaBlocks[idx][i] = palette[(dxtBlocks[idx].y >> i) & 0x3];
	}

}

__device__ void saveColorBlock(uint rgbaBlocks[NUM_THREADS][16], uint * result, int blockOffset, uint limit)
{
	const int bid = ( blockIdx.x + blockOffset ) * blockDim.x ;
        const int idx = threadIdx.x;

        if ((bid+threadIdx.x) >= limit) return;

	for(int i=0;i<4;i++) {
                for(int j=0;j<4;j++) {

                        result[(((bid+idx)/(IMAGE_WIDTH>>2)*4)+i)*IMAGE_WIDTH + ((bid+idx)%(IMAGE_WIDTH>>2))*4 + j] = rgbaBlocks[idx][i*4+j];
		}
	}
}

__global__ void decompress(const uint * image, uint * result, int blockOffset, uint thread_limit)
{
	//const int idx = threadIdx.x;
	__shared__ uint2 dxtBlocks[NUM_THREADS];
	__shared__ uint rgbaBlocks[NUM_THREADS][16];
	uint limit = thread_limit;

	loadColorBlock(image, dxtBlocks, blockOffset, limit);
	__syncthreads();

	decodeDXT1(dxtBlocks, rgbaBlocks, blockOffset, limit);
	__syncthreads();

	saveColorBlock(rgbaBlocks, result, blockOffset, limit);
	__syncthreads();
}

extern "C" void dxt_decompress_to_rgb4(uint32_t *d_data, uint32_t *d_result)
{	
	//uint mpcount = deviceProp.multiProcessorCount;
	uint mpcount = 16;
	const uint memSize = IMAGE_WIDTH * IMAGE_HEIGHT / 2;

	// Determine launch configuration and run timed computation numIterations times
	uint blocks = ((IMAGE_WIDTH + 3) / 4) * ((IMAGE_HEIGHT + 3) / 4); // rounds up by 1 block in each dim if %4 != 0

	// Restrict the numbers of blocks to launch on low end GPUs to avoid kernel timeout
	int blocksPerLaunch = min(blocks, 768 * mpcount);

	for( int j=0; j<(int)blocks; j+=blocksPerLaunch ) {
		decompress<<<min(blocksPerLaunch, blocks-j), NUM_THREADS>>>(d_data, d_result, j, memSize/8);
	}

	cudaThreadSynchronize();

}

/******************************************************************************************************* 
 * scale down: input frame is assumed to be RGB4
 * also, the total_dxt_blocks is the one "AFTER" the processing,
 * thus, it means that gid is the coordination to the frame of "AFTER" processed
 ******************************************************************************************************/
__device__ void scaleColorBlock(const uint * input, uint * output, ushort3 avg_mask[NUM_THREADS], 
		uint32_t total_dxt_blocks, uint32_t orig_width, uint32_t target_width, uint32_t blockOffset)
{
	// gid is the coordination to the after processed (pixel block id)
	int gid = blockDim.x * (blockOffset + blockIdx.x) + threadIdx.x; 
	if( gid >= total_dxt_blocks ) return;

	uint source;

	for (int i=0;i<4;i++) {
		for(int j=0;j<4;j++) {
			avg_mask[threadIdx.x].x = 0;
			avg_mask[threadIdx.x].y = 0;
			avg_mask[threadIdx.x].z = 0;
			source = 0;
			for(int k=0;k<2;k++) {
				for(int l=0;l<2;l++) {
					source = input[((gid+threadIdx.x)/(orig_width>>3)*8+(2*i)+k)*orig_width + 
							((gid+threadIdx.x)%(orig_width>>3)*8)+2*j+l];

					avg_mask[threadIdx.x].x = avg_mask[threadIdx.x].x + ((source >> 16) & 0xff);
					avg_mask[threadIdx.x].y = avg_mask[threadIdx.x].y + ((source >> 8) & 0xff);
					avg_mask[threadIdx.x].z = avg_mask[threadIdx.x].z + ((source) & 0xff);
				}
			}

			output[ ((gid+threadIdx.x)/(target_width>>2)*4+i)*target_width + (((gid+threadIdx.x)%(target_width>>2))*4)+j] = 
				(0xff << 24)| ((avg_mask[threadIdx.x].x/4) << 16) | ((avg_mask[threadIdx.x].y/4) << 8) | ((avg_mask[threadIdx.x].z/4) );
		}
	}
}

__global__ void ckernel_frame_resize(uint32_t *input, uint32_t *output, uint32_t total_dxt_blocks, 
		uint32_t orig_width, uint32_t orig_height, uint32_t target_width, uint32_t target_height,
		uint32_t blockOffset)
{

	// temporal variable for 2x2 average mask
	__shared__ ushort3 avg_mask[NUM_THREADS];

	scaleColorBlock(input, output, avg_mask, total_dxt_blocks,
			orig_width, target_width, blockOffset);

}

extern "C" int frame_resize(uint32_t *d_data, uint32_t *d_result, uint32_t total_dxt_blocks, 
		uint32_t orig_width, uint32_t orig_height, uint32_t target_width, uint32_t target_height )
{	
	/* in this current implementation, we assume making half of it */
	if( orig_width/2 != target_width ||
		orig_height/2 != target_height ) return -1;

	//uint mpcount = deviceProp.multiProcessorCount;
        uint mpcount = 16;

        // Determine launch configuration and run timed computation numIterations times
        uint blocks = ((target_width + 3) / 4) * ((target_height + 3) / 4); // rounds up by 1 block in each dim if %4 != 0

        // Restrict the numbers of blocks to launch on low end GPUs to avoid kernel timeout
        int blocksPerLaunch = min(blocks, 768 * mpcount);

        for( int j=0; j<(int)blocks; j+=blocksPerLaunch ) {

		ckernel_frame_resize<<<min(blocksPerLaunch, blocks-j), NUM_THREADS>>>(d_data, d_result, 
			total_dxt_blocks, orig_width, orig_height, target_width, target_height, j );
        }

	cudaThreadSynchronize();
	return 0;
}

