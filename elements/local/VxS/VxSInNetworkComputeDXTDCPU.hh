#ifndef __VXS_INNETWORK_COMPUTE_DXTDCPU__
#define __VXS_INNETWORK_COMPUTE_DXTDCPU__

#include <click/config.h>
#include <click/element.hh>
#include "VxSInNetworkCompute.hh"

CLICK_DECLS

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
