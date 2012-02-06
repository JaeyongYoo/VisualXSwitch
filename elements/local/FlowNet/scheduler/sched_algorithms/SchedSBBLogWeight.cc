// -*- c-basic-offset: 4 -*-
/*
 * 
 * Jae-Yong Yoo
 *
 * Copyright (c) 2010 Gwangju Institute of Science and Technology, Korea
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, subject to the conditions listed in the Click LICENSE
 * file. These conditions include: you must preserve this copyright
 * notice, and you cannot mention the copyright holders in advertising
 * related to the Software without their permission.  The Software is
 * provided WITHOUT ANY WARRANTY, EXPRESS OR IMPLIED. This notice is a
 * summary of the Click LICENSE file; the license in that file is
 * legally binding.
 */
#include <click/config.h>

#include <clicknet/ip.h>
#include <clicknet/udp.h>
#include <clicknet/ether.h>

#include "../../common/Table.hh"
#include "../PFSchedFW.hh"
#include "SchedSBBLogWeight.hh"
#include "../FlowSchedulable.hh"

#include "../../utils/FlowNetUtils.hh"

CLICK_DECLS

/*
 * jyyoo debugging : indenting stack depth 
 */
D_DEFINE_EXTERN;

/* 
 * note that this static function is thread-unsafe 
 * TODO: Make this not the static function
 */
static float compute_backpressure( FlowSchedulable *flow )
{
        struct SchedInfo *si = flow->getSchedInfo();
        float x = flow->queue_length();
        float y = si->ni.queuelen;
        if( x == 0.0f ) { x = 0.00001; } /* small enough value but not zero */
        if( y == 0.0f ) { y = 0.00001; } /* small enough value but not zero */

        return log(x) - log(y);
}

VcScheduleSBBLogWeight::VcScheduleSBBLogWeight(PFSchedFW *pf, double alpha)
{
	_pfSched = pf;
	_alpha = alpha;
	set_name("ScheduleSBBLogWeight");
}

VcScheduleSBBLogWeight::~VcScheduleSBBLogWeight()
{
}

int VcScheduleSBBLogWeight::pre_push(Flow* flow_in, Packet* p)
{
	D_START_FUNCTION;
	FlowSchedulable* flow = (FlowSchedulable*) flow_in;

        struct click_ether      *ether  = (struct click_ether*) p->data();
        struct click_ip         *ip     = (struct click_ip*)(ether+1);
        IPAddress       da = IPAddress(ip->ip_dst);


	/* update nexthop queue length if next hop is my desitnation */
	IPAddress nexthop_ip = p->dst_ip_anno();
	if( nexthop_ip == da ) {
		flow->si.ni.queuelen = 100;
	}

	D_END_FUNCTION;
	return 0;
}

int VcScheduleSBBLogWeight::post_push(Flow* flow_in, Packet*)
{
	D_START_FUNCTION;
	FlowSchedulable* flow = (FlowSchedulable*) flow_in;

	SchedInfo *si;
	si = flow->getSchedInfo();
	si->backpressure_value = compute_backpressure( flow );

	D_END_FUNCTION;
	return 0;
}

int VcScheduleSBBLogWeight::schedule(VcTable<Flow> *tbl_in, Flow** flow_in)
{
	FlowSchedulable **flow = (FlowSchedulable **) flow_in;
	VcTable<FlowSchedulable> *tbl = (VcTable<FlowSchedulable> *)tbl_in;
	D_START_FUNCTION;

	struct SBBLogWeight_SchedInfo *sbb_si;
	float max_bp_value = -10000.0f;
	int s = tbl->size();
	int smallest_sleep_time=INT_MAX;
	SchedInfo *si;
	FlowSchedulable *f;
	*flow = NULL;
	for( int i = 0; i<s; i++ ) {

		tbl->getAt(i, (FlowSchedulable**)&f);

		if( f == NULL )	{
			printf("Error! scheduler gets NULL flow\n");
			exit(-1);
		}

		si = f->getSchedInfo();

		/* note that cb_lhalf is used by inner scheduler */
		/* TODO: this action should be packed by FlowSchedulerable functions */
		sbb_si = (struct SBBLogWeight_SchedInfo *)si->cb_lhalf;

		/* this is the first detected flow, set the default alpha value */
		if( sbb_si->is_init == 0 ) { 
			sbb_si->is_init = 1;
			sbb_si->alpha = _alpha;
		}

		/* it does not have any packet to schedule */
		if( f->queue_length() == 0 ) continue;

		/*
		 * core of SBB (sleep-based backpressure) scheduling
		 * do not schedule if it is not enough to schedule the next packet 
		 */
		/* TODO: make this as a function */
		if( sbb_si->alpha != 0.0 ) {
			struct timeval tv;
			int32_t diff;
			int sleep_time;
			int required_sleep_time;

			if( si->backpressure_value > 0 ) {
				sleep_time = 0;
			} else {
				double absolute_bp = -si->backpressure_value;

//				sleep_time = pow(absolute_bp,  sbb_si->alpha); /* use logarithmic */

				sleep_time = absolute_bp * sbb_si->alpha; /* use linear */
			}

			gettimeofday( &tv, NULL );
			diff = (int32_t)timevaldiff( &(si->tv_last_schedule), &tv );

			
			if( diff < 0 ) {
				gettimeofday( &(si->tv_last_schedule), NULL );
			}

			required_sleep_time = sleep_time - diff;

			if( smallest_sleep_time > required_sleep_time ) {
				smallest_sleep_time = required_sleep_time;
			}

			if( required_sleep_time > 0 ) {
				continue;
			} else {
			}
		}

		if( si->backpressure_value > max_bp_value ) {
			*flow = f;
			max_bp_value = si->backpressure_value;
		} else if( si->backpressure_value == max_bp_value && rand() % 2 ) {
			*flow = f;
			max_bp_value = si->backpressure_value;
		}
	}

	/* we schedule this flow */
	if( *flow != NULL ) {
		f = *flow;
		si = f->getSchedInfo();
		gettimeofday( &(si->tv_last_schedule), NULL );
	} else {

		if( smallest_sleep_time != INT_MAX ) {
			_pfSched->turnon_waker( smallest_sleep_time/1000 );
		}
	}
		
	D_END_FUNCTION;
	return 0;
}

int VcScheduleSBBLogWeight::listen_promisc(Flow* flow_in)
{
	D_START_FUNCTION;
	FlowSchedulable* flow = (FlowSchedulable*) flow_in;

	SchedInfo *si;
	si = flow->getSchedInfo();
	si->backpressure_value = compute_backpressure( flow );

	D_END_FUNCTION;
	return 0;
}

int VcScheduleSBBLogWeight::queue_monitor_policy(Flow* flow_in, int16_t* len)
{
	D_START_FUNCTION;
	FlowSchedulable* flow = (FlowSchedulable*) flow_in;

	*len = flow->queue_length();
	D_END_FUNCTION;
	return 0;
}

int VcScheduleSBBLogWeight::l2_mapping(Flow* flow_in, uint8_t* l2_index)
{
	D_START_FUNCTION;
	FlowSchedulable* flow = (FlowSchedulable*) flow_in;
	float bpvalue = compute_backpressure( flow );

	float sp = bpvalue;

	if( sp  > BACKPRESSURE_QUANTIZATION_STEP*3 )
	{
//		stat.sched_VO ++;
		//*le_index = 0x28;      /* voice */ /* make the voice priority only control signal */
		*l2_index = 0x30;      /* voice */ /* XXX: At the moment, control signal is over wired connection */
	}
	else if( sp   <= BACKPRESSURE_QUANTIZATION_STEP*3 
			&& sp   > BACKPRESSURE_QUANTIZATION_STEP*2 )
	{
//		stat.sched_VI ++;
		*l2_index = 0x28;      /* Video */
	}
	else if( sp   <= BACKPRESSURE_QUANTIZATION_STEP*2 
			&& sp   > BACKPRESSURE_QUANTIZATION_STEP*1 )
	{
//		stat.sched_BE ++;
		*l2_index = 0x00;      /* Best effort */
	}
	else if( sp   <= BACKPRESSURE_QUANTIZATION_STEP*1 
			&& sp   > BACKPRESSURE_QUANTIZATION_STEP*0 )
	{
//		stat.sched_BK ++;
		*l2_index = 0x08;      /* Background */
	}
	else if( sp   <= BACKPRESSURE_QUANTIZATION_STEP*0 
			&& sp  > BACKPRESSURE_QUANTIZATION_STEP*-1 )
	{
//		stat.sched_P4 ++;
		*l2_index = 0x04;      /* P4 */
	}
	else if( sp   <= BACKPRESSURE_QUANTIZATION_STEP*-1 
			&& sp   > BACKPRESSURE_QUANTIZATION_STEP*-2 )
	{
//		stat.sched_P5 ++;
		*l2_index = 0x05;      /* P5 */
	}
	else if( sp   <= BACKPRESSURE_QUANTIZATION_STEP*-2 
			&& sp  > BACKPRESSURE_QUANTIZATION_STEP*-3 )
	{
//		stat.sched_P6 ++;
		*l2_index = 0x06;      /* P6 */
	}
	else if( sp  <= BACKPRESSURE_QUANTIZATION_STEP*-3 
			&& sp  > BACKPRESSURE_QUANTIZATION_STEP*-4 )
	{
//		stat.sched_P7 ++;
		*l2_index = 0x07;      /* P7 */
	} else {
		*l2_index = 0x00;
		D_END_FUNCTION;
		return -1;
	}

	D_END_FUNCTION;
	return 0;
}

/* for the interaction to the outer loop control */
/* TODO: this should go to VsSchedule virtual function */
double VcScheduleSBBLogWeight::add_sbb_to_flow(Flow *flow_in, double add) 
{
	FlowSchedulable* f = (FlowSchedulable*) flow_in;
        SchedInfo *si;
	struct SBBLogWeight_SchedInfo *sbb_si;
	si = f->getSchedInfo();

	/* note that cb_lhalf is used by inner scheduler */
	/* TODO: this action should be packed by FlowSchedulerable functions */
	sbb_si = (struct SBBLogWeight_SchedInfo *)si->cb_lhalf;

	sbb_si->alpha += add;
	if( sbb_si->alpha < 0.0 ) sbb_si->alpha = 0.0;
	if( sbb_si->alpha > 500 ) sbb_si->alpha = 500;

	return sbb_si->alpha;
}

double VcScheduleSBBLogWeight::get_sbb_to_flow(Flow *flow_in)
{
	FlowSchedulable* f = (FlowSchedulable*) flow_in;
        SchedInfo *si;
	struct SBBLogWeight_SchedInfo *sbb_si;
	si = f->getSchedInfo();

	/* note that cb_lhalf is used by inner scheduler */
	/* TODO: this action should be packed by FlowSchedulerable functions */
	sbb_si = (struct SBBLogWeight_SchedInfo *)si->cb_lhalf;

	return sbb_si->alpha;
}

CLICK_ENDDECLS
ELEMENT_PROVIDES(ScheduleSBBLogWeight)
