// -*- c-basic-offset: 4 -*-
/*
 * 
 * Jae-Yong Yoo
 *
 * Copyright (c) 2011 Gwangju Institute of Science and Technology, Korea
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, subject to the conditions listed in the Click LICENSE
 * file. These conditions include: you must preserve this copyright
 * notice, and you cannot mention the copyright holders in advertising
 * related to the Software without their permission.  The Software is
 * provided WITHOUT ANY WARRANTY, EXPRESS OR IMPLIED. This notice is a
 * summary of the Click LICENSE file; the license in that file is
 * legally binding.
 */

#include <click/config.h>
#include <unistd.h>
#include <sys/time.h>
#include "VxSInNetworkComputeDXT.hh"
#include "VxSInNetworkRawBatcher.hh"


#define NUM_THREADS 256
#define MUL(x,y)    (x*y)


/* TODO: make this compatible to config.h */
/* XXX: compile with the below line if the machine has CUDA driver.
 * unless, link error will happen. 
 */
#define COMPILE_THIS 0

#if COMPILE_THIS
#include <linux/cuda.h>
#include <cuda_runtime_api.h>
#include </home/netcs/NVIDIA_GPU_Computing_SDK/shared/inc/shrUtils.h>
extern "C" void dxt_compress_yuv_pixel_blocks(uint32_t *d_data, uint32_t *d_result, uint32_t total_dxt_blocks);
#endif

CLICK_DECLS

VxSInNetworkComputeDXT::VxSInNetworkComputeDXT(const char *name) : VxSInNetworkCompute(name)
{
#if COMPILE_THIS
        cudaError_t err;

	/* allocate the size of a dxt-memory */
	err = cudaMalloc((void**)&_cuda_input, VXS_MAX_DXT_CUDA_MEM_SIZE);
	if (err == cudaErrorMemoryAllocation) {
		click_chatter("cudaMalloc((void**)&cuda_input, memSizwe=%d) - cudaGetError: %s \n",
				VXS_MAX_DXT_CUDA_MEM_SIZE, cudaGetErrorString(err));
	}

	err = cudaMalloc((void**)&_cuda_output, VXS_MAX_DXT_CUDA_MEM_SIZE);
	if (err == cudaErrorMemoryAllocation) {
		click_chatter("cudaMalloc((void**)&cuda_output, memSizwe=%d) - cudaGetError: %s \n",
				VXS_MAX_DXT_CUDA_MEM_SIZE, cudaGetErrorString(err));
	}
#endif

}

VxSInNetworkComputeDXT::~VxSInNetworkComputeDXT()
{
}

int VxSInNetworkComputeDXT::compute(VxSInNetworkSegment *segment)
{
#if COMPILE_THIS
	cudaError_t err;
	VxSInNetworkRawSegment *raw_s = (VxSInNetworkRawSegment *)segment;
	int segment_size = raw_s->getSize();

	err = cudaMemcpy( _cuda_input, raw_s->getSegment(), segment_size, cudaMemcpyHostToDevice );
	if( err != cudaSuccess ) {
		click_chatter("cudaMemcpy %s\n", cudaGetErrorString(err) );
		return -1;
	}

//	dxt_compress_yuv_pixel_blocks( _cuda_input, _cuda_output, raw_s->getNumPixelBlocks() );


	/* FIXME: here, we just use segment_size for copying back to raw_s->segment, 
	 * but we should use 1/4 value which is the compression rate, but, for this we 
	 * need more general way thus, let's wait some time for some better idea comes into mind 
	 */
	int after_processed_size = raw_s->afterProcessedSize(); /* this one supposed to be 1/4 of the segment size */
	err = cudaMemcpy( raw_s->getSegment(), _cuda_output, after_processed_size, cudaMemcpyDeviceToHost );
	raw_s->setWrittenSize( after_processed_size );
//	err = cudaMemcpy( raw_s->getSegment(), _cuda_output, segment_size/4, cudaMemcpyDeviceToHost );
//	raw_s->setWrittenSize( segment_size/4 );

	if( err != cudaSuccess ) {
		click_chatter("cudaMemcpy%s \n", cudaGetErrorString(err));
		return -1;
	} 
		
#endif
	return 0;
}


CLICK_ENDDECLS
ELEMENT_PROVIDES(VxSInNetworkComputeDXT)
