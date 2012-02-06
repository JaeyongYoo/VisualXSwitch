#ifndef __CD_PROPORTIONAL_H__
#define __CD_PROPORTIONAL_H__

CLICK_DECLS
#include <float.h>

#include "../../common/CD.hh"

/* Queue loss based */
class VcCDDirect : public VcCongestionDetection {
public:
	VcCDDirect(uint32_t);
	~VcCDDirect();

        virtual int packet_enter(Flow* flow, const Packet* p);
        virtual int packet_leave(Flow* flow, const Packet* p);

	uint32_t _queuelen_threshold;	

};

CLICK_ENDDECLS

#endif
