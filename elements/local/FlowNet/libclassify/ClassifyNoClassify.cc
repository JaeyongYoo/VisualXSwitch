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

#include <arpa/inet.h>

#include "ClassifyNoClassify.hh"

CLICK_DECLS

/*
 * jyyoo debugging : indenting stack depth 
 */
D_DEFINE_EXTERN;

VcNoClassify::VcNoClassify()
{
	set_name("NoClassify");
}

VcNoClassify::~VcNoClassify()
{
}

int VcNoClassify::classify(const Packet* , struct FlowID* fid)
{
	D_START_FUNCTION;

	fid->len = 0;
	fid->classifier = this;	

	D_END_FUNCTION;
	return 0;
}

int VcNoClassify::to_string(const struct FlowID* , char* buf, int len)
{
	char str[512];

	sprintf(str, "NoClassifier" );

	if( len > (int)strlen(buf) ) {

		strncpy( buf, str, len );
		return 0;
	}

	return -1;
}

CLICK_ENDDECLS
ELEMENT_PROVIDES(VcNoClassify)
