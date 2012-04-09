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
#include "dds.h"

// Definitions
#define INPUT_IMAGE "iu3.dds"
#define W 1920
#define H 1080

#define W_DST 960
#define H_DST 540

#define NUM_THREADS 192        // Number of threads per block.

__device__ void scaleColorBlock(const uint * result, uint * result2, uint average[NUM_THREADS], int blockOffset, uint limit)
{
	const int bid = ( blockIdx.x + blockOffset ) * blockDim.x ;
	const int idx = threadIdx.x;

	//if (bid+idx==0) printf("limit: %d \n", limit);

	if ((bid+threadIdx.x) >= limit) return;

	for(int i1=0;i1<2;i1++) {
		for(int i2=0;i2<2;i2++) {
			for(int j1=0;j1<2;j1++) {
				for(int j2=0;j2<2;j2++) {
					// v0.1
					
					result2[(((bid+idx)/(W_DST>>2)*4)+(2*i1+i2))*W_DST + (((bid+idx)%(W_DST>>2))*4)+(2*j1+j2)] = result[(((bid+idx)/(W>>2)*16)+(4*i1)+i2+1)*W + ((bid+idx)%(W>>2))*8 + (4*j1)+j2+1];
				}
			}
		}
	}
}

__global__ void resize(const uint * image, uint * result, int blockOffset, uint thread_limit)
{
	__shared__ uint average[NUM_THREADS];
	uint limit = thread_limit;

	scaleColorBlock(image, result, average, blockOffset, limit);
	__syncthreads();
}

extern "C" void dxt_scale_yuv_pixel_blocks(uint32_t *d_data, uint32_t *d_result, uint32_t total_dxt_blocks)
{

}
