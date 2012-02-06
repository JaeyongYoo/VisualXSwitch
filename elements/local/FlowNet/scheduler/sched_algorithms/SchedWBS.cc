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
#include "SchedWBS.hh"
#include "../FlowSchedulable.hh"

CLICK_DECLS

/*
 * jyyoo debugging : indenting stack depth 
 */
D_DEFINE_EXTERN;

/* 
 * note that this static function is thread-unsafe 
 * TODO: Make this not the static function
 */
static float compute_backpressure( FlowSchedulable *flow, double b, double g )
{
	struct SchedInfo *si = flow->getSchedInfo();
	 
	return (float)( b*flow->queue_length() - g*si->ni.queuelen);
}

VcScheduleWBS::VcScheduleWBS(double b, double g)
{
	_beta = b;
	_gamma = g;
	set_name("ScheduleWBS");
}

VcScheduleWBS::~VcScheduleWBS()
{
}

int VcScheduleWBS::pre_push(Flow* flow_in, Packet* p)
{
	D_START_FUNCTION;
	FlowSchedulable* flow = (FlowSchedulable*) flow_in;

        struct click_ether      *ether  = (struct click_ether*) p->data();
        struct click_ip         *ip     = (struct click_ip*)(ether+1);
        IPAddress       da = IPAddress(ip->ip_dst);

	/* update nexthop queue length if next hop is my desitnation */
	IPAddress nexthop_ip = p->dst_ip_anno();
	if( nexthop_ip == da )
	{
		flow->si.ni.queuelen = 100;
	}

	D_END_FUNCTION;
	return 0;
}

int VcScheduleWBS::post_push(Flow* flow_in, Packet*)
{
	D_START_FUNCTION;
	FlowSchedulable* flow = (FlowSchedulable*) flow_in;

	SchedInfo *si;
	si = flow->getSchedInfo();
	si->backpressure_value = compute_backpressure( flow, _beta, _gamma );

	D_END_FUNCTION;
	return 0;
}

int VcScheduleWBS::schedule(VcTable<Flow> *tbl_in, Flow** flow_in)
{
	FlowSchedulable** flow = (FlowSchedulable**) flow_in;
	VcTable<FlowSchedulable>* tbl = (VcTable<FlowSchedulable>*)tbl_in;
	D_START_FUNCTION;

	float max_bp_value = -10000.0f;
	int s = tbl->size();
	SchedInfo *si;
	FlowSchedulable* f;
	*flow = NULL;
	for( int i = 0; i<s; i++ )
	{
		tbl->getAt(i, (FlowSchedulable**)&f);

		if( f == NULL )
		{
			printf("Error! scheduler gets NULL flow\n");
			exit(-1);
		}

		/* it does not have any packet to schedule */
		if( f->queue_length() == 0 ) continue;

		si = f->getSchedInfo();

		if( si->backpressure_value > max_bp_value ) {
			*flow = f;
			max_bp_value = si->backpressure_value;
		} else if( si->backpressure_value == max_bp_value && rand() % 2 ) {
			*flow = f;
			max_bp_value = si->backpressure_value;
		}
	}
	
	D_END_FUNCTION;
	return 0;
}

int VcScheduleWBS::listen_promisc(Flow* flow_in)
{
	D_START_FUNCTION;
	FlowSchedulable* flow = (FlowSchedulable*) flow_in;

	SchedInfo *si;
	si = flow->getSchedInfo();
	si->backpressure_value = compute_backpressure( flow, _beta, _gamma );

	D_END_FUNCTION;
	return 0;
}

int VcScheduleWBS::queue_monitor_policy(Flow* flow_in, int16_t* len)
{
	D_START_FUNCTION;
	FlowSchedulable* flow = (FlowSchedulable*) flow_in;

	*len = flow->queue_length();
	D_END_FUNCTION;
	return 0;
}

int VcScheduleWBS::l2_mapping(Flow* flow_in, uint8_t* l2_index)
{
	D_START_FUNCTION;
	FlowSchedulable* flow = (FlowSchedulable*) flow_in;
	float bpvalue = compute_backpressure( flow, _beta, _gamma );

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

CLICK_ENDDECLS
ELEMENT_PROVIDES(VcScheduleWBS)
