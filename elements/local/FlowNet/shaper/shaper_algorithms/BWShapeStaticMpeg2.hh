#ifndef _BW_SHAPE_STATICMPEG2_HH
#define _BW_SHAPE_STATICMPEG2_HH

#include <click/config.h>


#include "../../common/FlowCommon.hh"
#include "../../common/BWShape.hh"



class VcBWShapeStaticMpeg2 : public VcBWShape {

public:
	VcBWShapeStaticMpeg2(int accept_frame);
	~VcBWShapeStaticMpeg2();

public:
	int do_we_send(Flow* flow, Packet* p, const Element::Port &e);

	void change_rate( int r );
	int get_rate();

	int _accept_frame;
};

#endif
