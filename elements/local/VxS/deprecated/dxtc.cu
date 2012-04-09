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


extern "C" void dxt_compress_yuv_pixel_blocks(uint32_t *d_data, uint32_t *d_result, uint32_t total_dxt_blocks)
{	

//	int j;
	uint32_t total_cuda_blocks = total_dxt_blocks / NUM_THREADS;
	if( total_dxt_blocks % NUM_THREADS ) total_cuda_blocks ++;

	ckernel_dxt_compress_pixel_blocks<<<total_cuda_blocks, NUM_THREADS>>>(d_data, 1, (uint2*)d_result, 0, total_dxt_blocks);

	cudaThreadSynchronize();
}

