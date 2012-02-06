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
#include "SchedOuterVirtualpressure.hh"
#include "../FlowSchedulable.hh"
#include "../PFSchedFW.hh"

CLICK_DECLS

/*
 * jyyoo debugging : indenting stack depth 
 */
D_DEFINE_EXTERN;

VcScheduleOuterVirtualpressure::VcScheduleOuterVirtualpressure()
{
	set_name("ScheduleOuterVirtualpressure");
	_inner_scheduler = NULL;
	_sched = NULL;
	_no_closs_interval_counter = 0;

	/* for getting information from madwifi */
        _socket = socket(AF_INET, SOCK_DGRAM, 0);
        if( _socket < 0 ) {
		click_chatter("Error!: can not open socket for madwifi\n");
		perror("socket()");
		_socket = 0;
	} else {
		_ifr.ifr_data = (caddr_t) &_cur_ath_stats;

		/* FIXME: we have to make "ath0" as a variable */
		strncpy(_ifr.ifr_name, "wifi0", sizeof (_ifr.ifr_name));
	}
	_operation_counter = 0;
}

VcScheduleOuterVirtualpressure::~VcScheduleOuterVirtualpressure()
{
	if( _socket ) {
		close( _socket );
	}
}

void VcScheduleOuterVirtualpressure::periodic_monitor( int *time ) 
{
	/* 
	 * in this periodic monitor, we monitor the queue loss and channel loss packets 
	 */

	if( _sched == NULL || _inner_scheduler == NULL ) return;

	/* TODO: this getting _vcTable should be packed as function */
        VcTable<FlowSchedulable> *tbl = (VcTable<FlowSchedulable> *)_sched->_vcTable;
        int s = tbl->size();
        FlowSchedulable *f;

	char str[512];
	
	if( _socket ) {

		struct ath_stats ath_stats;
		memcpy( &ath_stats, &_cur_ath_stats, sizeof(_cur_ath_stats) );

		if (ioctl(_socket, SIOCGATHSTATS, &_ifr) < 0) {
			click_chatter("Error!: madwifi was not be able to read: ioctl fail\n");
		} else {
			int the_diff;
			the_diff = _cur_ath_stats.ast_tx_xretries - ath_stats.ast_tx_xretries;
			/* everything is fine, we got it */

			if( the_diff >= 5 /* TODO: import threshold write a comment on it */ ) { 
				_sched->add_slot_unit( 500 /* TODO: also important threshold write a comment on it */ );
				_no_closs_interval_counter = 0;
			} else {
				_no_closs_interval_counter ++;
			}

			if( _no_closs_interval_counter >= 5 ) /* if it is OK for 5 seconds */ {
				_sched->add_slot_unit( -500 );
			}

			sprintf(str, "SBB = %d CLOSS[ %d ] ", _sched->get_slot_unit(), the_diff );
		}

	} else {
		click_chatter("Error!: madwifi was not be able to read: no socket\n");
	}

        for( int i = 0; i<s; i++ ) {
		double sbb;
                tbl->getAt(i, (FlowSchedulable**)&f);
                if( f == NULL ) {
                        printf("Error! scheduler gets NULL flow\n");
                        exit(-1);
                }

		/* 
		 * XXX: note that this part is the control-interaction point to the inner-loop scheduler 
		 * This part should be carefully and clearly commented for future extension
		 */
		/* Rev 1: we do not use qdrop here, we must use the next-hop queue length */
//		if( f->qdrop_now > 5 ) { 

		/* TODO: make this as a wrapping function of FlowSchedulerable */
		if( f->si.ni.queuelen > 220 ) {
	
			sbb = _inner_scheduler->add_sbb_to_flow( f, 20 );
		} else {
			sbb = _inner_scheduler->add_sbb_to_flow( f, -1 );
		}

		VcFlowClassify* clfy = f->fid.classifier;
		char buf[256];
		clfy->to_string(&(f->fid), buf, 256 );
	
		sprintf( str + strlen(str), "[ %s  %d (%.2f) ]", buf, f->qdrop_now, sbb );
		f->qdrop_now = 0;

	}
	click_chatter("[%d] %s\n", _operation_counter++, str );

	/* 
	 * we manipulate the next call time by setting this time (milli-sec) value 
	 */
	*time = 1000;
}
void VcScheduleOuterVirtualpressure::act()
{
	/* do nothing */
}
int VcScheduleOuterVirtualpressure::bind( VcSchedule *s, PFSchedFW *f )
{
	_inner_scheduler = s;
	_sched = f;

	/* at the moment (2011-11-04), we only support SBB scheduler for inner-scheduler of virtual pressure */
	if(	strcmp( _inner_scheduler->name(), "ScheduleSBB" ) != 0 && 
		strcmp( _inner_scheduler->name(), "ScheduleSBBLogWeight") != 0 ) {
		click_chatter("Error! Virtual scheduler only takes inner scheduler as ScheduleSBB\n");
	}
	
	/* what ever it is, just return 0 */
	return 0;
}

CLICK_ENDDECLS
ELEMENT_PROVIDES(VcScheduleOuterVirtualpressure)
