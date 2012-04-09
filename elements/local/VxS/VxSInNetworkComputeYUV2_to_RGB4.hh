#ifndef __VXS_INNETWORK_COMPUTE_YUV2_TO_RGB4__
#define __VXS_INNETWORK_COMPUTE_YUV2_TO_RGB4__

#include <click/config.h>
#include <click/element.hh>
#include "VxSInNetworkCompute.hh"

CLICK_DECLS


/** 
 * Desc: A base class for in-network processing
 */
class VxSInNetworkComputeYUV2_to_RGB4 : public VxSInNetworkCompute {
public:

	VxSInNetworkComputeYUV2_to_RGB4(const char *name);
	~VxSInNetworkComputeYUV2_to_RGB4();

	int compute(VxSInNetworkSegment *segment);

private:
        uint *_cuda_input;
        uint *_cuda_output;

};

CLICK_ENDDECLS
#endif
