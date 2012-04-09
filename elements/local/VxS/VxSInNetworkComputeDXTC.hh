#ifndef __VXS_INNETWORK_COMPUTE_DXT__
#define __VXS_INNETWORK_COMPUTE_DXT__

#include <click/config.h>
#include <click/element.hh>
#include "VxSInNetworkCompute.hh"

CLICK_DECLS

/** 
 * Desc: A base class for in-network processing
 */
class VxSInNetworkComputeDXTC : public VxSInNetworkCompute {
public:

	VxSInNetworkComputeDXTC(const char *name);
	~VxSInNetworkComputeDXTC();

	int compute(VxSInNetworkSegment *segment);

private:
        uint *_cuda_input;
        uint *_cuda_output;

};

CLICK_ENDDECLS
#endif
