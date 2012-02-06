#ifndef __SCHED_SBB_LOGWEIGHT_H__
#define __SCHED_SBB_LOGWEIGHT_H__

/* 
 * schedule sleep-based backpressure 
 */

#include <click/config.h>

#include "../../common/FlowCommon.hh"
#include "../../common/Schedule.hh"

CLICK_DECLS

#define BACKPRESSURE_QUANTIZATION_STEP 30

class PFSchedFW;

struct SBBLogWeight_SchedInfo {
	int is_init;
	double alpha;
};

class VcScheduleSBBLogWeight : public VcSchedule {
public:
	VcScheduleSBBLogWeight(PFSchedFW *, double alpha);
	~VcScheduleSBBLogWeight();
public:
        virtual int pre_push(Flow *flow, Packet *p);
        virtual int post_push(Flow *flow, Packet *p);
        virtual int schedule(VcTable<Flow> *tbl, Flow **flow);
        virtual int listen_promisc(Flow *flow);
        virtual int queue_monitor_policy(Flow *flow, int16_t *len);
        virtual int l2_mapping(Flow *flow, uint8_t *l2_index);

	/* for the interaction to the outer loop control */
	/* TODO: this should go to VsSchedule virtual function */
	double add_sbb_to_flow(Flow *flow, double );
	double get_sbb_to_flow(Flow *flow);
private:
	double _alpha;
	PFSchedFW *_pfSched;
};

CLICK_ENDDECLS
#endif
