#ifndef _BW_SHAPE_PROPORTIONAL_HH
#define _BW_SHAPE_PROPORTIONAL_HH

#include <click/config.h>


#include "../../common/FlowCommon.hh"
#include "../../common/BWShape.hh"

class PFShapeFW;

struct CBShapeProportional {
	int queue_length;
};

class VcBWShapeProportional : public VcBWShape {

public:
	VcBWShapeProportional(PFShapeFW *shape);
	~VcBWShapeProportional();

public:
	int do_we_send(Flow* flow, Packet* p, const Element::Port &e);

	void queue_length_changed(struct FlowID* fid, const Packet* p, uint32_t ql);

public:
	PFShapeFW *_pfshape;	/* container framework of this algorithm */
private:
	uint32_t _target_rate; /* the unit is bytes per sec */

	/* we compute the current rate by using EWMA */
	uint32_t _averaged_rate; /* the unit is bytyes per sec */
	uint32_t _alpha; /* alpha in 100x scale */
	struct timeval _tv_last_packet_out;
	
};

#endif
