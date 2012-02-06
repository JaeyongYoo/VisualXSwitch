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
#include "SchedNoSchedule.hh"
#include "../FlowSchedulable.hh"

CLICK_DECLS

/*
 * jyyoo debugging : indenting stack depth 
 */
D_DEFINE_EXTERN;


VcScheduleNoSchedule::VcScheduleNoSchedule()
{
	set_name("ScheduleNoSchedule");
}
VcScheduleNoSchedule::~VcScheduleNoSchedule()
{
}

int VcScheduleNoSchedule::pre_push(Flow *, Packet *)
{
	D_START_FUNCTION;

	D_END_FUNCTION;
	return 0;
}

int VcScheduleNoSchedule::post_push(Flow *, Packet *)
{
	D_START_FUNCTION;

	D_END_FUNCTION;
	return 0;
}

int VcScheduleNoSchedule::schedule(VcTable<Flow> *tbl, Flow **flow)
{
	D_START_FUNCTION;

	int s = tbl->size();

	if( s == 1 ) {
		Flow* f;
		tbl->getAt(0, &f);
		*flow = f;
	
		/* do not pass no-packet flow to the PFSched framework */
		if( f->queue_len == 0 ) *flow = NULL;

	} else if ( s == 0 ) {
		/* do nothing */
		*flow = NULL;
	} else {
		/* if s is greater than 1, means that scheduler framework has
		 * classifier, then, we need to do some actual scheduling.
		 * show some warning.
		 */
		click_chatter("Error! NoScheduler has multiple queues.\n");
		click_chatter("You need to setup valid scheduler.\n");
	}
	
	D_END_FUNCTION;
	return 0;
}

int VcScheduleNoSchedule::listen_promisc(Flow* )
{
	D_START_FUNCTION;

	D_END_FUNCTION;
	return 0;
}

int VcScheduleNoSchedule::queue_monitor_policy(Flow* , int16_t* q)
{
	D_START_FUNCTION;
	*q = 0;
	D_END_FUNCTION;
	return 0;
}

int VcScheduleNoSchedule::l2_mapping(Flow* , uint8_t* l2_index)
{
	D_START_FUNCTION;

	/* no link scheduling: always best effort */

	*l2_index = 0x00;      /* Best effort */

	D_END_FUNCTION;
	return 0;
}

CLICK_ENDDECLS
ELEMENT_PROVIDES(ScheduleNoSchedule)
