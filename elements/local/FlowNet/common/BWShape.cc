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
#include <click/ipaddress.hh>
#include <click/confparse.hh>
#include <click/error.hh>
#include <click/glue.hh>
#include <click/straccum.hh>
#include <click/router.hh>
#include <sys/time.h>


#include "BWShape.hh"

CLICK_DECLS

VcBWShape::VcBWShape()
{
	_cd = NULL;
}

VcBWShape::~VcBWShape()
{
}

void VcBWShape::toString( Flow *, char *buf, int  )
{
	sprintf(buf, "VcBWShape toString Not implemented\n");
}

void VcBWShape::registerCDCallback(VcCongestionDetection* CD)
{
	_cd = CD;
	if( _cd ) {
		_cd->register_congestion_callback( _congestion_detected, this );
		_cd->register_nocongestion_callback( _nocongestion_detected, this );
		_cd->register_queue_length_change_notify_callback( _queue_length_changed, this );
	} else {
		click_chatter("Warning!! congestion detection is null\n");
	}
}  
void VcBWShape::congestion_action(struct FlowID *, const Packet *)
{
	/* TODO: make this usage as a sub-function */
	click_chatter("congestion_action function complains!\n");
	click_chatter("Do not call me! (I'm not implemented in %s algorithm\n",
		name() );
	click_chatter("You probably mix the wrong combination of Shaping and Congestion Detection (CD) algorithms\n");
	click_chatter("Solution: read README and find the proper combination\n");
}

void VcBWShape::nocongestion_action(struct FlowID *, const Packet *)
{
	click_chatter("nocongestion_action function complains!\n");
	click_chatter("Do not call me! (I'm not implemented in %s algorithm\n",
		name() );
	click_chatter("You probably mix the wrong combination of Shaping and Congestion Detection (CD) algorithms\n");
	click_chatter("Solution: read README and find the proper combination\n");
}

void VcBWShape::queue_length_changed(struct FlowID* fid, const Packet* p, uint32_t ql)
{
	click_chatter("queue_length_changed function complains!\n");
	click_chatter("Do not call me! (I'm not implemented in %s algorithm\n",
		name() );
	click_chatter("You probably mix the wrong combination of Shaping and Congestion Detection (CD) algorithms\n");
	click_chatter("Solution: read README and find the proper combination\n");
}

void VcBWShape::_congestion_detected(struct CongestionNotification *cn)
{
        VcBWShape *shaper = (VcBWShape*) cn->object;

        if( shaper ) {
                shaper->congestion_action(cn->fid, cn->packet);
        }
}

void VcBWShape::_nocongestion_detected(struct CongestionNotification *cn)
{
        VcBWShape *shaper = (VcBWShape*) cn->object;

        if( shaper ) {
                shaper->nocongestion_action(cn->fid, cn->packet);
        }
}

void VcBWShape::_queue_length_changed(struct QueueLengthChangeNotification *qn)
{
        VcBWShape *shaper = (VcBWShape*) qn->object;

        if( shaper ) {
                shaper->queue_length_changed(qn->fid, qn->packet, qn->queue_length);
        }
}



CLICK_ENDDECLS
ELEMENT_PROVIDES(VcBWShape)




