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
//#include <cutil_inline.h>
//#include <cutil_math.h>
#include "dxt_cuda.h"
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
__constant__ unsigned int constAlpha = 0xff << 24;
//__constant__ float  constHueColorSpaceMat[9];

/* cuda debug mode */
//#define DEBUG_MODE 
#define _DEBUG
cudaError_t error;

//#define NUM_THREADS 64        // Number of threads per block.

//extern "C" void CompressDXTCompressDXT(sageBuffer, width, height, FORMAT_DXT1, block_image, d_data, d_result, blocks, blocksPerLaunch);

__device__ void YUV2RGB(unsigned int *yuvi, float *red, float *green, float *blue, float *constHueColorSpaceMat)
{
        float luma, chromaCb, chromaCr;

	const int bid = ( blockIdx.x ) * blockDim.x ;
        const int idx = threadIdx.x;


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
	
	if ( (bid==0) && (idx==0)) {
	printf("in YUV2RGB, ");
	for (int i=0; i<9; i++) printf("%f ", constHueColorSpaceMat[i]);
	printf("\n");
	}

}

__device__ unsigned int RGBAPACK_10bit(float red, float green, float blue, unsigned int alpha)
{
        unsigned int ARGBpixel = 0;

        // Clamp final 10 bit results
        red   = min(max(red,   0.0f), 1023.f);
        green = min(max(green, 0.0f), 1023.f);
        blue  = min(max(blue,  0.0f), 1023.f);

        // Convert to 8 bit unsigned integers per color component
/*        ARGBpixel = (((unsigned int)blue  >> 2) |
                        (((unsigned int)green >> 2) << 8)  |
                        (((unsigned int)red   >> 2) << 16) | (unsigned int)alpha);
*/
	  ARGBpixel = (((unsigned int)blue  >> 2) |
                        (((unsigned int)green >> 2) << 8)  |
                        (((unsigned int)red   >> 2) << 16) );


        return  ARGBpixel;
}
__global__ void uyvy2rgb(const uint *image, uint *dstImage, float *hue, int blockOffset)
{

        const int bid = ( blockIdx.x + blockOffset ) * blockDim.x ;
        const int idx = threadIdx.x;
	__shared__ float constHueColorSpaceMat[9];
	for (int i=0;i<9;i++) constHueColorSpaceMat[i]=hue[i];

        uint c = image[(bid+idx)];
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


	if( (bid==0) && (idx==0)) {
		printf("raw value: y: %u, u: %u, v: %u \n", (image[(bid+idx)] >> 8) & 0xff , (image[(bid+idx)] >> 0)&0xff, (image[(bid+idx)] >> 16)&0xff);
		printf("in uyvy2rgb8, yuvi: %08x, y: %u, u:%u, v: %u, r: %f, g: %f, b: %f, stImage[(bid+idx)*2]:%08x \n ",image[(bid+idx)], yuvi[0], yuvi[1], yuvi[2], red[0], green[0], blue[0], dstImage[(bid+idx)*2]);

	}

}

////////////////////////////////////////////////////////////////////////////////
// Compress color block
////////////////////////////////////////////////////////////////////////////////
#ifdef DEBUG_MODE
__global__ void compress(const uint * image, uint3 * result, int blockOffset)
#else
__global__ void compress(const uint * image, uint2 * result, int blockOffset)
//__global__ void compress(const unsigned char * image, uint2 * result, int width, int blockOffset)
#endif
{
	// FIXME: the amount of allocated memory of shared and registers should be balanced
	// for registers, don't forget to count constants.
	// 1. check how much memory amount of both shared and registers are in the GPU
	// 2. proportionally balance between shared and registers
	// 3. try to reuse memory


	// FIXME: float* hue is a constant. map it to the constant or texture memory.
	float hue[9];
	float hueSin = 0; //sin(0.0);
	float hueCos = 1; //cos(0.0);


	hue[0] = 1.1644f;
	hue[1] = hueSin * 1.5960f;
	hue[2] = hueCos * 1.5960f;
	hue[3] = 1.1644f;
	hue[4] = (hueCos * -0.3918f) - (hueSin * 0.8130f);
	hue[5] = (hueSin *  0.3918f) - (hueCos * 0.8130f);
	hue[6] = 1.1644f;
	hue[7] = hueCos *  2.0172f;
	hue[8] = hueSin * -2.0172f;

	const int picture_width = 1920;
	// a thread takes care of a DXT-block (4x4 pixels)
	int gid = blockDim.x * blockIdx.x + threadIdx.x; // better to be register
	int tid = threadIdx.x;
	int x_DXT_block;
	int y_DXT_block;
	int tmp_image;
	int x;
	int y;
	int i;

	// XXX: Note that in this parallization, we don't need to use shared memory 
	// since each thread does not share any information
	__shared__ uchar3 colors[NUM_THREADS][16];
	__shared__ uchar3 min[NUM_THREADS];
	__shared__ uchar3 max[NUM_THREADS];

	// copy image to colors
	for(y=0;y<4;y++) 
	{
		for(x=0;x<2;x++) 
		{
			x_DXT_block = gid % (picture_width / 4);
			y_DXT_block = gid / (picture_width / 4);

			// FIXME: the unit of reading global memory (image variable) is 32 bytes
			// need to batching this image reading
			tmp_image = image[4*y_DXT_block*picture_width + x_DXT_block*4 + y * 4 + x ];
			// this tmp_image contains 2 pixels (4 byte) 

			// perform RGB converting
			int yuvi[4];
			int red[2], green[2], blue[2];

			yuvi[0] = ( (tmp_image >> 8) & 0xff) - 16;	// y1
			yuvi[1] = ( (tmp_image >> 0) & 0xff) - 128;	// u	
			yuvi[2] = ( (tmp_image >> 16) & 0xff) - 128;	// v
			yuvi[3] = ( (tmp_image >> 24) & 0xff) - 16;	// y2

			colors[tid][y*4+x*2].x = (yuvi[0] + (1.5958 * yuvi[2])) * 0.859375;
			colors[tid][y*4+x*2].y = (yuvi[0] - (0.81290 * yuvi[2]) - (0.39173 * yuvi[1])) * 0.859375;
			colors[tid][y*4+x*2].z = (yuvi[0] + (2.017 * yuvi[1])) * 0.859375;

			colors[tid][y*4+x*2 + 1].x = (yuvi[3] + (1.5958 * yuvi[2])) * 0.859375;
			colors[tid][y*4+x*2 + 1].y = (yuvi[3] - (0.81290 * yuvi[2]) - (0.39173 * yuvi[1])) * 0.859375;
			colors[tid][y*4+x*2 + 1].z = (yuvi[3] + (2.017 * yuvi[1])) * 0.859375;
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

	uchar3 temp_uchar3;
	temp_uchar3.x = (max[tid].x - min[tid].x) >> 4;
	temp_uchar3.y = (max[tid].y - min[tid].y) >> 4;
	temp_uchar3.z = (max[tid].z - min[tid].z) >> 4;

	min[tid].x = ( min[tid].x + temp_uchar3.x <= 255) ? min[tid].x + temp_uchar3.x : 255;
	min[tid].y = ( min[tid].y + temp_uchar3.y <= 255) ? min[tid].y + temp_uchar3.y : 255;
	min[tid].z = ( min[tid].z + temp_uchar3.z <= 255) ? min[tid].z + temp_uchar3.z : 255;
	max[tid].x = ( max[tid].x >= temp_uchar3.x) ? max[tid].x - temp_uchar3.x : 0;
	max[tid].y = ( max[tid].y >= temp_uchar3.y) ? max[tid].y - temp_uchar3.y : 0;
	max[tid].z = ( max[tid].z >= temp_uchar3.z) ? max[tid].z - temp_uchar3.z : 0;


	// round the color to RGB565 and expand
	ushort2 temp;
	temp.x = ( (max[tid].x >> 3) << 11) | ( (max[tid].y >> 2) << 5) | (max[tid].z >> 3);
	temp.y = ( (min[tid].x >> 3) << 11) | ( (min[tid].y >> 2) << 5) | (min[tid].z >> 3);

	result[gid].x= (temp.y<< 16) | temp.x;

	
	// emit color index

	ushort3 dxt[4];
	uint temp = 0; 

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

	int c0, c1, c2;
	int d0, d1, d2, d3;
	int b0, b1, b2, b3;
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
		
		temp |= ( x2 | ( (x0 | x1) << 1) ) << (i<<1);
	}

	result[gid].y = temp;	
}


extern "C" void CompressDXT(unsigned char *out, uint* block_image, uint *d_data, uint *d_rgba, uint *d_result, uint blocks4yuv, uint blocksPerLaunch4yuv, uint blocks4dxt, uint blocksPerLaunch4dxt, uint compressedSize, float *HueColorSpaceMat)
{	

	int j;
	for(j=0; j<blocks4dxt; j+=blocksPerLaunch4dxt ) {
			compress<<<min(blocksPerLaunch4dxt, blocks4dxt-j), NUM_THREADS>>>(d_rgba, (uint2 *)d_result, j);
	}
}

    
