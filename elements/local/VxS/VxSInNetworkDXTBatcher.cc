// -*- c-basic-offset: 4 -*-
/*
 * 
 * Jae-Yong Yoo
 *
 * Copyright (c) 2011 Gwangju Institute of Science and Technology, Korea
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
#include <string.h>
#include <stdlib.h>
#include "VxSInNetworkDXTBatcher.hh"
        
CLICK_DECLS

/**
 * implementation of VxSInNetworkDXTSegment
 */
VxSInNetworkDXTSegment::VxSInNetworkDXTSegment()
{
}
VxSInNetworkDXTSegment::~VxSInNetworkDXTSegment()
{
}

/**
 * implementation of VxSInNetworkDXTBatcher
 */
VxSInNetworkDXTBatcher::VxSInNetworkDXTBatcher(const struct sw_flow_key *fid, 
		VxSInNetworkTaskQueue *tq_in, VxSInNetworkTaskQueue *tq_out) 
	: VxSInNetworkFlowBatcher( fid, tq_in, tq_out )
{
	strncpy( _media_type_name, media_type_name[VXS_MEDIA_TYPE_DXT], VXS_MAX_FLOW_TYPE_NAME );
}

VxSInNetworkDXTBatcher::~VxSInNetworkDXTBatcher()
{
}

int VxSInNetworkDXTBatcher::pushPacket(struct ofpbuf *ob, const struct ofp_action_header *ah, int actions_len)
{
	return VxSInNetworkFlowBatcher::pushPacket( ob, ah, actions_len );
}

int VxSInNetworkDXTBatcher::sendToTaskQueue(struct ofpbuf *)
{
	return 0;
}

int VxSInNetworkDXTBatcher::recvFromTaskQueue(Datapath *)
{
	return 0;
}
CLICK_ENDDECLS
ELEMENT_PROVIDES(VxSInNetworkDXTBatcher)
