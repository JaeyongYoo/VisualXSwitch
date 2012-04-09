#ifndef __DXTC_H__
#define __DXTC_H__

extern "C" void CompressDXT(unsigned char *out, uint* block_image, uint *d_data, uint *d_result, uint blocks, uint blocksPerLaunch, uint compressedSize);

#endif
