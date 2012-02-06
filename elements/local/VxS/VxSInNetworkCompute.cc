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
#include <unistd.h>
#include "VxSInNetworkCompute.hh"


CLICK_DECLS

VxSInNetworkCompute::VxSInNetworkCompute(const char *name)
{
	if( strlen( name ) > VSX_MAX_COMPUTE_NAME ) {
		click_chatter("Error: the name of compute (%s) is larger than max_name (%d)\n", 
			name, VSX_MAX_COMPUTE_NAME);
	}
	strncpy( _compute_name, name, VSX_MAX_COMPUTE_NAME );
}

VxSInNetworkCompute::~VxSInNetworkCompute()
{
}

int VxSInNetworkCompute::isThisCompute(const char *name)
{
	return strncmp(_compute_name, name, VSX_MAX_COMPUTE_NAME) == 0;
}


CLICK_ENDDECLS
ELEMENT_PROVIDES(VxSInNetworkCompute)
