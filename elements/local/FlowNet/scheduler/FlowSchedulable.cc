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
#include <clicknet/ether.h>

#include <stdarg.h>

#include "FlowSchedulable.hh"
#include "../utils/FlowNetUtils.hh"
CLICK_DECLS
/*
 * jyyoo debugging : indenting stack depth 
 */
D_DEFINE_EXTERN;

FlowSchedulable::FlowSchedulable() 
{
	memset( &si, 0, sizeof(si) );	
	memset( ci, 0, sizeof(ci));
}

FlowSchedulable::~FlowSchedulable()
{
}

void FlowSchedulable::clear()
{
	memset( &si, 0, sizeof(si) );

	for( int i = 0; i<MAX_CD_ALGORITHMS; i++ ) {
		if( ci[i].cb != NULL ) {

			/* we use generic free function */
			::free( ci[i].cb );
			ci[i].cb = NULL;
		}
	}

	schedule_status = SCHEDULE_STATUS_QLEN_MONITOR_SEQ; /* default is sequence checking */

	Flow::clear();
}

void FlowSchedulable::setNexthopInfo(uint8_t* macaddr, IPAddress ip)
{
	memcpy( si.ni.macaddr, macaddr, WIFI_ADDR_LEN );
	si.ni.ipaddr = ip;
}
void FlowSchedulable::toString(char* buf, int len)
{
	if( len > 100 )
	{
		sprintf(buf, "[nql:%d qdrp:%d/%d sent:%d]", si.ni.queuelen, qdrop_now, qdrop, total_sent );

//		qdrop_now = 0;
//		sent_now = 0;
	}
}

int FlowSchedulable::update_nexthop_queuelen(Packet* p)
{
	D_START_FUNCTION;

	click_ether*	ethdr = (click_ether*)p->data();
	click_ip*	iphdr = (click_ip*)(ethdr+1);
	bool 		update=false;
	uint16_t 	seq = iphdr->ip_id;
	struct NexthopInfo* ni = &(si.ni);


	/* this packet is not from my next-hop node */
	if( memcmp( ni->macaddr, ethdr->ether_shost, WIFI_ADDR_LEN ) ) 
	{	
		return 0;
	}

	if( schedule_status & SCHEDULE_STATUS_QLEN_MONITOR_SEQ )	{

		int int16_max = 32767; /* for carrier */
		if( ni->queuelen_monitor_seq == 0 ) /* this is probably the starting flow */
		{
			update = true;
		} else {
			int diff = seq - ni->queuelen_monitor_seq;

			if(	diff < -int16_max  || /* this is probably carrier round robin */
					diff > 0 ) update = true;
		}

	} else {
		update = true;
	}

	if( update ) {

		/* extract queue length */
		ni->queuelen = (iphdr->ip_off & 0xff00) >> 8;
	}
	D_END_FUNCTION;
	return 0;
}

CLICK_ENDDECLS
ELEMENT_PROVIDES(Flow)
