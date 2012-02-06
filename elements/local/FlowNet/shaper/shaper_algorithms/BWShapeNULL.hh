#ifndef _BW_SHAPE_NULL_HH
#define _BW_SHAPE_NULL_HH

#include <click/config.h>


#include "../../common/FlowCommon.hh"
#include "../../common/BWShape.hh"


class VcBWShapeNULL : public VcBWShape {

public:
	VcBWShapeNULL();
	~VcBWShapeNULL();

public:
	int do_we_send(Flow* flow, Packet* p, const Element::Port &e);
};

#endif
