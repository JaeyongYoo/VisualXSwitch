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

#include "FlowBWShaperable.hh"
CLICK_DECLS
/*
 * jyyoo debugging : indenting stack depth 
 */
D_DEFINE_EXTERN;

void FlowBWShaperable::clear()
{

	memset( &si, 0, sizeof(si) );	
	bwshape_status = 0;
	lower_layer_flow = NULL;

	Flow::clear();
}

void FlowBWShaperable::toString( char* buf, int len )
{
	if( _vcShape ) {
		_vcShape->toString( this, buf, len );
	} else {
		sprintf( buf, "Shaper is NULL\n");
	}
}
CLICK_ENDDECLS
ELEMENT_PROVIDES(FlowBWShaperable)
