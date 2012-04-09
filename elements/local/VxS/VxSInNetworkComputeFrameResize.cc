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
#include "VxSInNetworkComputeFrameResize.hh"
#include "VxSInNetworkRawBatcher.hh"
#include <linux/cuda.h>
#include <cuda_runtime_api.h>
#include </home/netcs/NVIDIA_GPU_Computing_SDK/shared/inc/shrUtils.h>

#define NUM_THREADS 256
#define MUL(x,y)    (x*y)

CLICK_DECLS

extern "C" int frame_resize(uint32_t *d_data, uint32_t *d_result, uint32_t total_dxt_blocks,
                uint32_t orig_width, uint32_t orig_height, uint32_t target_width, uint32_t target_height );

VxSInNetworkComputeFrameResize::VxSInNetworkComputeFrameResize(const char *name) : VxSInNetworkCompute(name)
{
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

}

VxSInNetworkComputeFrameResize::~VxSInNetworkComputeFrameResize()
{
}

int VxSInNetworkComputeFrameResize::compute(VxSInNetworkSegment *segment)
{
	cudaError_t err;
	VxSInNetworkRawSegment *raw_s = (VxSInNetworkRawSegment *)segment;
	int segment_size = raw_s->getSize();

	err = cudaMemcpy( _cuda_input, raw_s->getSegment(), segment_size, cudaMemcpyHostToDevice );
	if( err != cudaSuccess ) {
		click_chatter("cudaMemcpy %s\n", cudaGetErrorString(err) );
		return -1;
	}

	int orig_width = raw_s->getWidth();
	int orig_height = raw_s->getHeight();
	int new_width  = orig_width/2;
	int new_height = orig_height/2;
	raw_s->setWidthHeight( new_width, new_height );

	frame_resize( _cuda_input, _cuda_output, raw_s->getCurrentNumPixelBlocks(), 
			orig_width, orig_height, new_width, new_height );

	/* TODO: scale */
        raw_s->setBytePerPixelBlocks( 64 ); /* 1 pixel block of DXT-compressed is 8 byte */
        int after_processed_size = raw_s->getCurrentNumPixelBlocks() * raw_s->getBytePerPixelBlocks() /* byte per pixel */;

        if( raw_s->prepareSegment( after_processed_size ) == 0 ) {
                err = cudaMemcpy( raw_s->getSegment(), _cuda_output, after_processed_size, cudaMemcpyDeviceToHost );
                raw_s->setWrittenSize( after_processed_size );
        } else {
                click_chatter("Error: segment prepareing segment size failed\n");
        }
	if( err != cudaSuccess ) {
		click_chatter("cudaMemcpy%s \n", cudaGetErrorString(err));
		return -1;
	} 
		
	return 0;
}


CLICK_ENDDECLS
ELEMENT_PROVIDES(VxSInNetworkComputeFrameResize)
