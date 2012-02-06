#ifndef _BW_SHAPE_BYCORE_CFB_HH
#define _BW_SHAPE_BYCORE_CFB_HH

#include <click/config.h>


#include "../../common/FlowCommon.hh"
#include "../../common/BWShape.hh"


/*
 * jyyoo NOTE: CFB stands for Cross-Flow Balancing
 * When CD notifies congestion for the corresponding flow, 
 * it does not decrease this flow's shaping rate but the
 * the one that has the highest flow's shaping rate in 
 * this router. This is for fair sharing. But we need to 
 * consider which fairness are we targetting. This one
 * more suits with absolute fairness 
 */

class PFShapeFW;

struct CBhapeByCDCORE_CFB {
	int accept_frame;
	VcCongestionDetection*	cd;
	struct timeval tv_last_congestion;
};

class VcBWShapeByCDCORE_CFB : public VcBWShape {

public:
	VcBWShapeByCDCORE_CFB(PFShapeFW* shape);
	~VcBWShapeByCDCORE_CFB();

	virtual void toString(Flow* flow, char* buf, int len);

public:
	int do_we_send(Flow* flow, Packet* p, const Element::Port &e);

	void congestion_action(struct FlowID* fid, const Packet* p);
	void nocongestion_action(struct FlowID* fid, const Packet* p);

private:

	PFShapeFW *_pfshape;
	struct timeval _tv_last_signal;
	VcCongestionDetection *_cd;
	int _accept_frame;
};

#endif
