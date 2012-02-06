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
#include "VxSInNetworkBatchManager.hh"
#include "VxSInNetworkRawBatcher.hh"
#include "VxSInNetworkDXTBatcher.hh"

CLICK_DECLS

const char *media_type_name[VXS_END_OF_TYPE] = {
"VXS_MEDIA_TYPE_RAW",
"VXS_MEDIA_TYPE_DXT",
"VXS_MEDIA_TYPE_MPEG2",
"VXS_MEDIA_TYPE_H264"
};

/**
 * implementation of VxSInNetworkSegment
 */

VxSInNetworkSegment::VxSInNetworkSegment()
{
}

VxSInNetworkSegment::~VxSInNetworkSegment()
{
}

int VxSInNetworkSegment::setActionHeader( const uint8_t *d, uint32_t size )
{
	if( size >= VXS_MAX_ACTION_HEADER ) {
		click_chatter("Error: action length too long (limit=%d, action_len=%d)\n", 
				VXS_MAX_ACTION_HEADER, size);
		return -1;
	} 
	memcpy( _action_header, d, size );
	_action_len = size;
	_action_header_program_counter = NULL;
	return 0;
}

uint8_t * VxSInNetworkSegment::getNextActionHeader()
{
	if( _action_header_program_counter == NULL ) {
		_action_header_program_counter = _action_header;
	} else {
		/* first check */
		int diff = (int)(_action_header_program_counter - _action_header);

		if( diff >= _action_len ) {
			/* if we reach here, the actions are completely done */
			return NULL;
		}

		struct ofp_action_header *ah = (struct ofp_action_header *)_action_header_program_counter;
		int16_t length = htons(ah->len);

		if( length == 0 ) return NULL;		

		_action_header_program_counter = _action_header_program_counter + length;

		diff = (int)(_action_header_program_counter - _action_header);

		if( diff >= _action_len ) {
			/* if we reach here, the actions are completely done */
			return NULL;
		}
	}
	return _action_header_program_counter;
}

void VxSInNetworkSegment::print_to_chatter()
{
	click_chatter("Segment: action info\n");
	int total_len = 0;
	while( total_len < _action_len ) {
		struct ofp_action_header *ah = (struct ofp_action_header *)(_action_header + total_len);
		int16_t len = htons(ah->len);
		int16_t type = htons(ah->type);
		if( len == 0 ) return;
		total_len += len;
		click_chatter("[%d - %d] ", type, len );
	}
	click_chatter("\n");
}

/**
 * implementation of VxSInNetworkFlowBatcher
 */
VxSInNetworkFlowBatcher::VxSInNetworkFlowBatcher(const struct sw_flow_key *fid, 
	VxSInNetworkTaskQueue *tq_in, VxSInNetworkTaskQueue *tq_out )
{
	memcpy( &_flow_id, fid, sizeof(_flow_id) );
	_task_queue_incoming = tq_in;
	_task_queue_outgoing = tq_out;
}

VxSInNetworkFlowBatcher::~VxSInNetworkFlowBatcher()
{
}

/* Desc: flow comparison: by default, we ignore wildcard */
int VxSInNetworkFlowBatcher::pushPacket(struct ofpbuf *, const struct ofp_action_header *ah, int actions_len)
{
	_action_headers = ah;
	_action_len = actions_len;
	return 0;
}
int VxSInNetworkFlowBatcher::isTheSameFlow(const struct sw_flow_key *fid)
{
	return flow_compare( &_flow_id.flow, &fid->flow ) == 0;
}


/**
 * implementation of VxSInNetworkBatchManager
 */
VxSInNetworkBatchManager::VxSInNetworkBatchManager(VxSInNetworkTaskQueue *tq_in, VxSInNetworkTaskQueue *tq_out )
{
	_task_queue_incoming = tq_in;
	_task_queue_outgoing = tq_out;
}

VxSInNetworkBatchManager::~VxSInNetworkBatchManager()
{
	std::list<VxSInNetworkFlowBatcher *>::iterator ii;

	for( ii = _batchers.begin(); ii != _batchers.end(); ii ++ ) {
		VxSInNetworkFlowBatcher *b = *ii;
		delete b;
	}
}

VxSInNetworkFlowBatcher * VxSInNetworkBatchManager::createBatcher(int32_t media_type, const struct sw_flow_key *fid)
{
	VxSInNetworkFlowBatcher* re = NULL;
	switch( media_type ) {
	case VXS_MEDIA_TYPE_RAW:
		re = new VxSInNetworkRawBatcher( fid, _task_queue_incoming, _task_queue_outgoing );
	break;

	case VXS_MEDIA_TYPE_DXT:
		re = new VxSInNetworkDXTBatcher( fid, _task_queue_incoming, _task_queue_outgoing );
	break;
	
	default:
		click_chatter("Error: batcher creation request for unimplemented media type: %d (%s)\n", 
			media_type, media_type_name[media_type]);
		break;
	}

	if( re ) _batchers.push_back( re );

	return re;
}

VxSInNetworkFlowBatcher * VxSInNetworkBatchManager::searchBatcher(const struct sw_flow_key *fid) 
{
	std::list<VxSInNetworkFlowBatcher *>::iterator ii;

	for( ii = _batchers.begin(); ii != _batchers.end(); ii ++ ) {
		VxSInNetworkFlowBatcher *b = *ii;
		
		if( b->isTheSameFlow( fid ) ) {
			return b;
		}
	}
	return NULL;
}

int VxSInNetworkBatchManager::recvPacket(struct ofpbuf *ob, const struct sw_flow_key *fid, 
	const struct ofp_action_header *ah, int actions_len, int media_type)
{
	/* first, validate if the input params are ok */
	if( ob == NULL || fid == NULL || ah == NULL ) { 
		click_chatter("Error: invalid param for VxSInNetworkBatchManager::recvPacket\n");
		return -1;
	}

	VxSInNetworkFlowBatcher *b;

	/* search if we have the corresponding key for the flow */
	b = searchBatcher( fid );

	/* we don't have the bather for this particular flow @fid, 
	 * so we make a new batcher for this */
	if( b == NULL ) {
		b = createBatcher( media_type, fid );
	}

	/* if @b is still null, it means out-of-memory,
	 * halt immediately with warning! */
	if( b == NULL ) {
		click_chatter("Error: out of memory: VxSInNetworkBatchManager::recvPacket\n");
		return -1;
	}

	/* we pass the handle to the packet @ofpbuf to batcher */
	return b->pushPacket( ob, ah, actions_len );
}

int VxSInNetworkBatchManager::sendPacket(Datapath* dp)
{
	std::list<VxSInNetworkFlowBatcher *>::iterator ii;
	int residual;
	int total_residual = 0;
	for( ii = _batchers.begin(); ii != _batchers.end(); ii ++ ) {
		VxSInNetworkFlowBatcher *b = *ii;
		residual = b->recvFromTaskQueue( dp );
		/* 
		 * TODO: determines the next-call time based on the 
		 * residual number of tasks
		 */
		total_residual += residual;

	}
	return total_residual;
}

CLICK_ENDDECLS
ELEMENT_PROVIDES(VxSInNetworkBatchManager)
