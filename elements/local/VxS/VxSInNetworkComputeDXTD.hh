#ifndef __VXS_INNETWORK_COMPUTE_DXTD__
#define __VXS_INNETWORK_COMPUTE_DXTD__

#include <click/config.h>
#include <click/element.hh>
#include "VxSInNetworkCompute.hh"

CLICK_DECLS

/** 
 * Desc: A base class for in-network processing
 */
class VxSInNetworkComputeDXTD : public VxSInNetworkCompute {
public:

	VxSInNetworkComputeDXTD(const char *name);
	~VxSInNetworkComputeDXTD();

	int compute(VxSInNetworkSegment *segment);

private:
        uint *_cuda_input;
        uint *_cuda_output;

};

CLICK_ENDDECLS
#endif
