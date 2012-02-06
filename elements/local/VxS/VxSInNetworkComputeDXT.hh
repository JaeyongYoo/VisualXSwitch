#ifndef __VXS_INNETWORK_COMPUTE_DXT__
#define __VXS_INNETWORK_COMPUTE_DXT__

#include <click/config.h>
#include <click/element.hh>
#include "VxSInNetworkCompute.hh"

CLICK_DECLS

#define VXS_MAX_DXT_CUDA_MEM_SIZE 2048*4096*3 /* 3-byte RGB with 4K */

/** 
 * Desc: A base class for in-network processing
 */
class VxSInNetworkComputeDXT : public VxSInNetworkCompute {
public:

	VxSInNetworkComputeDXT(const char *name);
	~VxSInNetworkComputeDXT();

	int compute(VxSInNetworkSegment *segment);

private:
        uint *_cuda_input;
        uint *_cuda_output;

};

CLICK_ENDDECLS
#endif
