#ifndef _BW_SHAPE_SIMPLE_HH
#define _BW_SHAPE_SIMPLE_HH

#include <click/config.h>


#include "../../common/FlowCommon.hh"
#include "../../common/BWShape.hh"

class PFShapeFW;

struct CBShapeSimple {
	bool congested;
};

class VcBWShapeSimple : public VcBWShape {

public:
	VcBWShapeSimple(PFShapeFW *shape);
	~VcBWShapeSimple();

public:
	int do_we_send(Flow* flow, Packet* p, const Element::Port &e);

        void congestion_action(struct FlowID* fid, const Packet* p);
        void nocongestion_action(struct FlowID* fid, const Packet* p);

public:
	PFShapeFW *_pfshape;	/* container framework of this algorithm */
};

#endif
