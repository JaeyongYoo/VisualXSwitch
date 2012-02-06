#ifndef _BW_SHAPE_BYCORE_HH
#define _BW_SHAPE_BYCORE_HH

#include <click/config.h>


#include "../../common/FlowCommon.hh"
#include "../../common/BWShape.hh"

class PFShapeFW;

struct CBShapeByCDCORE {
	int accept_frame;
	VcCongestionDetection*	cd;
	struct timeval tv_last_congestion;
};

class VcBWShapeByCDCORE : public VcBWShape {

public:
	VcBWShapeByCDCORE(PFShapeFW* shape);
	~VcBWShapeByCDCORE();

	virtual void toString(Flow* flow, char* buf, int len);

public:
	int do_we_send(Flow* flow, Packet* p, const Element::Port &e);

	void congestion_action(struct FlowID* fid, const Packet* p);
	void nocongestion_action(struct FlowID* fid, const Packet* p);
	
private:

	PFShapeFW* _pfshape;
	struct timeval _tv_last_signal;
	VcCongestionDetection* _cd;
	int _accept_frame;


};

#endif
