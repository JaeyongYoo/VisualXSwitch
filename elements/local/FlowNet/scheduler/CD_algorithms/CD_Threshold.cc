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

#include "CD_Threshold.hh"
#include "../FlowSchedulable.hh"
#include "../../utils/FlowNetUtils.hh"
CLICK_DECLS
/*
 * jyyoo debugging : indenting stack depth 
 */
D_DEFINE_EXTERN;

VcCDThreshold::VcCDThreshold(uint32_t threshold)
{
	set_name("CDThreshold");
	_queuelen_threshold = threshold;
}
VcCDThreshold::~VcCDThreshold()
{
}

int VcCDThreshold::packet_enter(Flow* flow_in, const Packet* p)
{
	struct CongestionNotification cn;
	
	if( flow_in != NULL ) {
		/* 
		 * TODO: We can hide this private_data by making it 
		 * the member function of VcCD 
		 */
	        cn.object = private_data;
		cn.fid = &(flow_in->fid);
		cn.packet = p;

		if( flow_in->queue_length() > _queuelen_threshold ) {
			if( congest_detected ) {
				congest_detected( &cn );
			}
		} else {
			if( nocongest_detected ) {
				nocongest_detected( &cn );
			}
		}
	}
	return 0;
}

int VcCDThreshold::packet_leave(Flow *flow_in, const Packet *p)
{
	/* do nothing */
	struct CongestionNotification cn;
	
	if( flow_in != NULL ) {
		/* 
		 * TODO: We can hide this private_data by making it 
		 * the member function of VcCD 
		 */
	        cn.object = private_data;
		cn.fid = &(flow_in->fid);
		cn.packet = p;

		if( flow_in->queue_length() > _queuelen_threshold ) {
			if( congest_detected ) {
				congest_detected( &cn );
			}
		} else {
			if( nocongest_detected ) {
				nocongest_detected( &cn );
			}
		}
	}
	return 0;

	return 0;
}


CLICK_ENDDECLS
ELEMENT_PROVIDES(VcCDThreshold)
