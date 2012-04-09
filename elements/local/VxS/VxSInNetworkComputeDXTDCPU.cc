// -*- c-basic-offset: 4 -*-
/*
 * 
 * Namgon Kim
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
#include "VxSInNetworkComputeDXTDCPU.hh"
#include "VxSInNetworkRawBatcher.hh"


CLICK_DECLS

// Helper structs and functions to validate the output of the compressor.
// We cannot simply do a bitwise compare, because different compilers produce different
// results for different targets due to floating point arithmetic.

union Color32 {
    struct {
        unsigned char b, g, r, a;
    };
    unsigned int u;
};

union Color16 {
    struct {
        unsigned short b : 5;
        unsigned short g : 6;
        unsigned short r : 5;
    };
    unsigned short u;
};

struct BlockDXT1
{
    Color16 col0;
    Color16 col1;
    union {
        unsigned char row[4];
        unsigned int indices;
    };
    
    void decompress(Color32 colors[16]) const;
};

void BlockDXT1::decompress(Color32 * colors) const
{
    Color32 palette[4];
    
    // Does bit expansion before interpolation.
    palette[0].b = (col0.b << 3) | (col0.b >> 2);
    palette[0].g = (col0.g << 2) | (col0.g >> 4);
    palette[0].r = (col0.r << 3) | (col0.r >> 2);
    palette[0].a = 0xFF;
    
    palette[1].r = (col1.r << 3) | (col1.r >> 2);
    palette[1].g = (col1.g << 2) | (col1.g >> 4);
    palette[1].b = (col1.b << 3) | (col1.b >> 2);
    palette[1].a = 0xFF;
    
    if( col0.u > col1.u ) {
        // Four-color block: derive the other two colors.
        palette[2].r = (2 * palette[0].r + palette[1].r) / 3;
        palette[2].g = (2 * palette[0].g + palette[1].g) / 3;
        palette[2].b = (2 * palette[0].b + palette[1].b) / 3;
        palette[2].a = 0xFF;
        
        palette[3].r = (2 * palette[1].r + palette[0].r) / 3;
        palette[3].g = (2 * palette[1].g + palette[0].g) / 3;
        palette[3].b = (2 * palette[1].b + palette[0].b) / 3;
        palette[3].a = 0xFF;
    }
    else {
        // Three-color block: derive the other color.
        palette[2].r = (palette[0].r + palette[1].r) / 2;
        palette[2].g = (palette[0].g + palette[1].g) / 2;
        palette[2].b = (palette[0].b + palette[1].b) / 2;
        palette[2].a = 0xFF;

        palette[3].r = 0x00;
        palette[3].g = 0x00;
        palette[3].b = 0x00;
        palette[3].a = 0x00;
    }

    for (int i = 0; i < 16; i++)
    {
        colors[i] = palette[(indices >> (2*i)) & 0x3];
    }
}

VxSInNetworkComputeDXTDCPU::VxSInNetworkComputeDXTDCPU(const char *name) : VxSInNetworkCompute(name)
{
	/* initialize input data pointer; the address of segment will be used as _input  */	
	_input = 0;

	/* allocate the size of a dxt-memory; 
	   this will be shared if compute is called multiple times  */
	int _size = 1920 * 1080 * sizeof(Color32);  // W * H * sizeof (Color32)
	_output = (uint8_t *)malloc( _size );
	if( _output == NULL ) {
		click_chatter("Error: out of memory: requesting size=%d\n", _size);
	}
}

VxSInNetworkComputeDXTDCPU::~VxSInNetworkComputeDXTDCPU()
{
	/* free _output if destroyed */
	free(_output);
}

int VxSInNetworkComputeDXTDCPU::compute(VxSInNetworkSegment *segment)
{
	VxSInNetworkRawSegment *raw_s = (VxSInNetworkRawSegment *)segment;

	uint w = 1920;
	uint h = 1080;

	_input = raw_s->getSegment();

	Color32* colors0;
	BlockDXT1* block;
        for (uint y = 0; y < h; y += 4)
        {
            for (uint x = 0; x < w; x += 4)
            {
                uint resultBlockIdx = ((y/4) * (w/4) + (x/4));
		block = ((BlockDXT1 *)_input) + resultBlockIdx;
		colors0 = ((Color32 *)_output) + 16 * resultBlockIdx;
		block->decompress(colors0);
            }
        }

	int after_processed_size = w * h * sizeof(Color32);

	/* copy output to segment */
	memcpy(raw_s->getSegment(), _output, after_processed_size);	
	raw_s->setWrittenSize( after_processed_size );
	return 0;
}


CLICK_ENDDECLS
ELEMENT_PROVIDES(VxSInNetworkComputeDXTDCPU)
