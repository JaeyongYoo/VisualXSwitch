#ifndef __VXS_INNETWORK_COMPUTE_DXTDCPU__
#define __VXS_INNETWORK_COMPUTE_DXTDCPU__

#include <click/config.h>
#include <click/element.hh>
#include "VxSInNetworkCompute.hh"

CLICK_DECLS

#define VXS_MAX_DXT_CUDA_MEM_SIZE 2048*4096*3 /* 3-byte RGB with 4K */

/** 
 * Desc: A base class for in-network processing
 */
class VxSInNetworkComputeDXTDCPU : public VxSInNetworkCompute {
public:

	VxSInNetworkComputeDXTDCPU(const char *name);
	~VxSInNetworkComputeDXTDCPU();

	int compute(VxSInNetworkSegment *segment);

private:
        uint8_t *_input;
        uint8_t *_output;

};

CLICK_ENDDECLS
#endif
