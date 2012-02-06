// -*- c-basic-offset: 4 -*-
#ifndef _SCHEDULE_HH
#define _SCHEDULE_HH
#include <click/glue.hh>
#include <click/element.hh>
#include <click/timer.hh>
#include <click/notifier.hh>

#include "Algorithm.hh"
#include "FlowCommon.hh"
#include "Flow.hh"
#include "Table.hh"

CLICK_DECLS


/*!
 * virtual functions of scheduling algorithms
 */

class VcSchedule : public Algorithm {
public:
        VcSchedule() {};
        ~VcSchedule() {};
public:

	/* generic scheduling functions */
        virtual int pre_push(Flow* flow, Packet* p) = 0;
        virtual int post_push(Flow* flow, Packet* p) = 0;
        virtual int schedule(VcTable<Flow>* tbl, Flow** flow) = 0;
        virtual int listen_promisc(Flow* flow) = 0;
        virtual int queue_monitor_policy(Flow* flow, int16_t* len) = 0;
        virtual int l2_mapping(Flow* flow, uint8_t* l2_index) = 0;

	/* the one that expects to control the outer-loop control should implement this */
	/* TODO: rename required since the name of below two functions are for
	 * SBB (Sleep-Based Backpressure) algorithm */
	virtual double add_sbb_to_flow(Flow *, double ) { click_chatter("add_sbb_to_flow not implemented in %s\n", name()); return 0.0; }
	virtual double get_sbb_to_flow(Flow *) { click_chatter("get_sbb_to_flow not implemented in %s\n", name()); return 0.0; }

protected:
};

CLICK_ENDDECLS
#endif
