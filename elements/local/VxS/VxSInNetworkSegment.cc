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
#include "VxSInNetworkSegment.hh"

CLICK_DECLS

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

CLICK_ENDDECLS
ELEMENT_PROVIDES(VxSInNetworkSegment)
