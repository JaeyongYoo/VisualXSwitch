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

	void set_input_mode(int yuv) { _input_mode = yuv; };


private:
	/** 
	 * input mode to compute:
	 *   0 indicates rgb4
	 *   1 indicates yuv2 
	 */
	uint32_t _input_mode;

        uint *_cuda_input;
        uint *_cuda_output;

};

CLICK_ENDDECLS
#endif
