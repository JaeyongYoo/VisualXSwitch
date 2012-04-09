#ifndef __VXS_INNETWORK_COMPUTE_FRAME_RESIZE__
#define __VXS_INNETWORK_COMPUTE_FRAME_RESIZE__

#include <click/config.h>
#include <click/element.hh>
#include "VxSInNetworkCompute.hh"

CLICK_DECLS

/** 
 * Desc: A base class for in-network processing
 */
class VxSInNetworkComputeFrameResize : public VxSInNetworkCompute {
public:

	VxSInNetworkComputeFrameResize(const char *name);
	~VxSInNetworkComputeFrameResize();

	int compute(VxSInNetworkSegment *segment);

private:
        uint *_cuda_input;
        uint *_cuda_output;

};

CLICK_ENDDECLS
#endif
