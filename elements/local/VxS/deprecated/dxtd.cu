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

/**
#include <shrUtils.h>
#include <shrQATest.h>
#include <cutil_inline.h>
#include <cutil_math.h>
#include "dds.h"
#include <bmpfile.h>
// for output file for validation
*/

// Definitions
#define INPUT_IMAGE "iu3.dds"
#define W 1920
#define H 1080

//#define NUM_THREADS 192        // Number of threads per block.

////////////////////////////////////////////////////////////////////////////////
// Load color block to shared mem
////////////////////////////////////////////////////////////////////////////////
__device__ void loadColorBlock(const uint * image, uint2 dxtBlocks[NUM_THREADS], int blockOffset, uint limit)
{
	const int bid = ( blockIdx.x + blockOffset ) * blockDim.x;
      	const int idx = threadIdx.x;

	if ((bid+threadIdx.x) >= limit) return;

	dxtBlocks[idx].x = image[(bid+idx)*2];
	dxtBlocks[idx].y = image[(bid+idx)*2+1];
}

__device__ void decodeDXT1(uint2 dxtBlocks[NUM_THREADS], uint rgbaBlocks[NUM_THREADS][16], int blockOffset, uint limit )
//__device__ void decodeDXT1(uint2 dxtBlocks[NUM_THREADS], uint * result, int blockOffset, uint limit)
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

                        result[(((bid+idx)/(W>>2)*4)+i)*W + ((bid+idx)%(W>>2))*4 + j] = rgbaBlocks[idx][i*4+j];
		}
	}


}

////////////////////////////////////////////////////////////////////////////////
// Decompress color block
////////////////////////////////////////////////////////////////////////////////
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

extern "C" void dxt_decompress_dxt_blocks(uint32_t *d_data, uint32_t *d_result)
{	
	uint w = W, h = H;
	//uint mpcount = deviceProp.multiProcessorCount;
	uint mpcount = 16;
	const uint memSize = w * h / 2;

	// Determine launch configuration and run timed computation numIterations times
	uint blocks = ((w + 3) / 4) * ((h + 3) / 4); // rounds up by 1 block in each dim if %4 != 0

	// Restrict the numbers of blocks to launch on low end GPUs to avoid kernel timeout
	int blocksPerLaunch = min(blocks, 768 * mpcount);

	for( int j=0; j<(int)blocks; j+=blocksPerLaunch ) {
		decompress<<<min(blocksPerLaunch, blocks-j), NUM_THREADS>>>(d_data, d_result, j, memSize/8);
	}

	cudaThreadSynchronize();

}

