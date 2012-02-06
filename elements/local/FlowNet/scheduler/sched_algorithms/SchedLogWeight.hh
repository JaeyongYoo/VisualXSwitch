#ifndef __SCHED_LOG_WEIGHT_H__
#define __SCHED_LOG_WEIGHT_H__

#include <click/config.h>

#include "../../common/FlowCommon.hh"
#include "../../common/Schedule.hh"

CLICK_DECLS

#define BACKPRESSURE_QUANTIZATION_STEP 30

class VcScheduleLogWeight : public VcSchedule {
public:
	VcScheduleLogWeight();
	~VcScheduleLogWeight();
public:
        virtual int pre_push(Flow* flow, Packet* p);
        virtual int post_push(Flow* flow, Packet* p);
        virtual int schedule(VcTable<Flow>* tbl, Flow** flow);
        virtual int listen_promisc(Flow* flow);
        virtual int queue_monitor_policy(Flow* flow, int16_t* len);
        virtual int l2_mapping(Flow* flow, uint8_t* l2_index);
};


CLICK_ENDDECLS

#endif
