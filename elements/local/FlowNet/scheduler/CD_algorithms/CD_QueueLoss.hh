#ifndef __CD_QUEUE_LOSS_H__
#define __CD_QUEUE_LOSS_H__

CLICK_DECLS
#include <float.h>

#include "../../common/CD.hh"

/* Queue loss based */
class VcCDQueueLoss : public VcCongestionDetection {
public:
	VcCDQueueLoss();
	~VcCDQueueLoss();

        virtual int packet_enter(Flow* flow, const Packet* p);
        virtual int packet_leave(Flow* flow, const Packet* p);
	

};

CLICK_ENDDECLS

#endif
