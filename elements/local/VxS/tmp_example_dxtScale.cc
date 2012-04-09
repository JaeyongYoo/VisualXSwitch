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
#include <shrUtils.h>
#include <shrQATest.h>
#include <cutil_inline.h>
#include <cutil_math.h>
#include "dds.h"

// Definitions
#define INPUT_IMAGE "iu3.dds"
#define W 1920
#define H 1080

#define W_DST 960
#define H_DST 540

#define NUM_THREADS 192        // Number of threads per block.

// for output file for validation
#include <bmpfile.h>


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{
	shrQAStart(argc, argv);
	shrSetLogFileName ("dxtc.txt");
	shrLog("[%s] Starting...\n\n", argv[0]); 

	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") )
		cutilDeviceInit(argc, argv);
	else
		cudaSetDevice( cutGetMaxGflopsDeviceId() );

	// Load input image.
	//uint W, H;

	char* image_path = shrFindFilePath(INPUT_IMAGE, argv[0]);
	if (image_path == 0) 
	{
		shrLog("Error, unable to find source image  <%s>\n", image_path);
		cutilDeviceReset();
		shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
	}

	unsigned char* data;
/*
	if (!shrLoadPPM4ub(image_path, &data, &W, &H))
	{
        shrLog("Error, unable to open source image file <%s>\n", image_path);
        cutilDeviceReset();
        shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
   	}
*/
	uint w = W, h = H;
	const uint memSize = w * h / 2;


	data = (unsigned char*)malloc(memSize);
	FILE* fp = fopen(image_path,"rb");
	fseek(fp, sizeof(DDSHeader), SEEK_SET);
	uint readSize = fread(data, 1, memSize, fp);
	if (readSize!=memSize) printf("file read failed, read=%u\n",readSize);
	
	//printf("The size of DDSHeader: %d \n", sizeof(DDSHeader));
	//unsigned char* data = shrLoadRawFile(image_path, memSize);

	if (!data) 
	{
		shrLog("Error, unable to open source image file <%s>\n", image_path);
		cutilDeviceReset();
		shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
	}

	shrLog("DXT Image Loaded '%s', %d x %d pixels\n\n", image_path, w, h);

	// Allocate input image.
	cutilCondition( 0 != memSize );

	//printf("memSize: %d \n", memSize);

	// copy into global mem
	uint * d_data = NULL;
	cutilSafeCall( cudaMalloc((void**) &d_data, memSize) );


	// Result
	uint* d_result = NULL;
	const uint decodedSize = w * h * 4;
	cutilSafeCall(cudaMalloc((void**)&d_result, decodedSize) );

	uint * d_data2 = NULL;
	cutilSafeCall( cudaMalloc((void**) &d_data2, decodedSize) );


	uint* d_result_ = NULL;
	cutilSafeCall(cudaMalloc((void**)&d_result_, decodedSize>>2) );

	uint* h_result = (uint *)malloc(decodedSize);
	uint* h_result2 = (uint *)malloc(decodedSize>>2);

	//printf("decodedSize: %d \n", decodedSize);
	// create a timer
	uint timer;
	cutilCheckError(cutCreateTimer(&timer));

	// Copy image from host to device
	cutilSafeCall(cudaMemcpy(d_data, data, memSize, cudaMemcpyHostToDevice) );

	// Determine launch configuration and run timed computation numIterations times
	uint blocks = ((w + 3) / 4) * ((h + 3) / 4); // rounds up by 1 block in each dim if %4 != 0
	uint reblocks = blocks/4; // rounds up by 1 block in each dim if %4 != 0

	printf("reblocks: %d \n", reblocks);
	int devID;
	cudaDeviceProp deviceProp;

	// get number of SMs on this GPU
	cutilSafeCall(cudaGetDevice(&devID));
	cutilSafeCall(cudaGetDeviceProperties(&deviceProp, devID));

	// Restrict the numbers of blocks to launch on low end GPUs to avoid kernel timeout
	int blocksPerLaunch = min(blocks, 768 * deviceProp.multiProcessorCount);

	shrLog("Running DXT Compression on %u x %u image...\n", w, h);
	shrLog("\n%u Blocks, %u Threads per Block, %u Threads in Grid...\n\n", 
			blocks, NUM_THREADS, blocks * NUM_THREADS);

	//cutilSafeCall(cutilDeviceSynchronize()); 
	cutilSafeCall(cutilDeviceSynchronize()); 
	cutilCheckError(cutStartTimer(timer));   
	int j;
	for(j=0; j<(int)blocks; j+=blocksPerLaunch ) {
		decompress<<<min(blocksPerLaunch, blocks-j), NUM_THREADS>>>(d_data, d_result, j, memSize/8);
	}

	cutilSafeCall(cudaMemcpy(h_result, d_result, decodedSize, cudaMemcpyDeviceToHost));

	// Copy image from host to device
	cutilSafeCall(cudaMemcpy(d_data2, h_result, decodedSize, cudaMemcpyHostToDevice) );


	blocksPerLaunch = min(reblocks, 768 * deviceProp.multiProcessorCount);
	for(j=0; j<(int)reblocks; j+=blocksPerLaunch ) {
		resize<<<min(blocksPerLaunch, reblocks-j), NUM_THREADS>>>(d_result, d_result_, j, reblocks);
	}


	cutilCheckMsg("decompress");

	// sync to host, stop timer, record perf
	cutilSafeCall(cutilDeviceSynchronize());
	cutilCheckError(cutStopTimer(timer));
	double dAvgTime = 1.0e-3 * cutGetTimerValue(timer);
	shrLogEx(LOGBOTH | MASTER, 0, "dxtc, Throughput = %.4f MPixels/s, Time = %.5f s, Size = %u Pixels, NumDevsUsed = %i, Workgroup = %d\n", 
			(1.0e-6 * (double)(W * H)/ dAvgTime), dAvgTime, (W * H), 1, NUM_THREADS); 

	// copy result data from device to host
	//cutilSafeCall(cudaMemcpy(h_result, d_result, decodedSize, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(h_result2, d_result_, decodedSize/4, cudaMemcpyDeviceToHost));


	bmpfile_t *bmp;
	rgb_pixel_t pixel;

	char output_filename[1024];
        strcpy(output_filename, image_path);
        strcpy(output_filename + strlen(image_path) - 3, "bmp");

	bmp = bmp_create(W_DST, H_DST, 32);
	//bmp = bmp_create(W, H, 32);

	for (int h1=0; h1<H_DST; h1++) {
		for (int w1=0; w1<W_DST; w1++) {
			pixel.red = (h_result2[h1*W_DST+w1] >> 16 ) & 0xff;
			pixel.green = (h_result2[h1*W_DST+w1] >> 8 ) & 0xff;
			///pixel.green = 0;
			pixel.blue = (h_result2[h1*W_DST+w1] >> 0 ) & 0xff;
			//pixel.blue = 0;
			//pixel.alpha = (h_result[h1*1920+w1] >> 24 ) & 0xff;
			bmp_set_pixel(bmp, w1, h1, pixel);
			//if (w1==0 && h1==0) printf("pixel: %x \n", h_result[h1*1920+w1]);
		}
	}

	bmp_save(bmp, output_filename);
  	bmp_destroy(bmp);

	// Write out result data to DDS file
/*
	char output_filename[1024];
	strcpy(output_filename, image_path);
	strcpy(output_filename + strlen(image_path) - 3, "ppm");
	FILE* fp = fopen(output_filename, "wb");
	if (fp == 0) {
		shrLog("Error, unable to open output image <%s>\n", output_filename);
		cutilDeviceReset();
		shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
	}

	//fwrite(&header, sizeof(DDSHeader), 1, fp);
	fwrite(h_result, decodedSize, 1, fp);
	fclose(fp);
*/
	/*
	// Make sure the generated image is correct.
	const char* reference_image_path = shrFindFilePath(REFERENCE_IMAGE, argv[0]);
	if (reference_image_path == 0) {
	shrLog("Error, unable to find reference image\n");
	cutilDeviceReset();
	shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
	}
	fp = fopen(reference_image_path, "rb");
	if (fp == 0) {
	shrLog("Error, unable to open reference image\n");
	cutilDeviceReset();
	shrQAFinishExit(argc, (const char **)argv, QA_FAILED);
	}
	fseek(fp, sizeof(DDSHeader), SEEK_SET);
	uint referenceSize = (W / 4) * (H / 4) * 8;
	uint* reference = (uint *)malloc(referenceSize);
	fread(reference, referenceSize, 1, fp);
	fclose(fp);

	shrLog("\nChecking accuracy...\n");
	float rms = 0;
	for (uint y = 0; y < h; y += 4)
	{
	for (uint x = 0; x < w; x += 4)
	{
	uint referenceBlockIdx = ((y/4) * (W/4) + (x/4));
	uint resultBlockIdx = ((y/4) * (w/4) + (x/4));

	int cmp = compareBlock(((BlockDXT1 *)h_result) + resultBlockIdx, ((BlockDXT1 *)reference) + referenceBlockIdx);
	if (cmp != 0.0f) {
	shrLog("Deviation at (%4d,%4d):\t%f rms\n", x/4, y/4, float(cmp)/16/3);
	}
	rms += cmp;
	}
	}
	rms /= w * h * 3;
	 */
	// Free allocated resources and exit
	cutilSafeCall(cudaFree(d_data));
	cutilSafeCall(cudaFree(d_result));
	shrFree(image_path);
	free(data);
	free(h_result);
	cutilCheckError(cutDeleteTimer(timer));
	cutilDeviceReset();

}
