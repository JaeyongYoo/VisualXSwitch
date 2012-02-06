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

#include "PFSinkFW.hh"

CLICK_DECLS


/*
 * jyyoo debugging : indenting stack depth 
 */
D_DEFINE_EXTERN;


void * PFSinkFW::cast(const char *name)
{
	if (strcmp(name, "PFSinkFW") == 0)
		return (void *)this;
	else
		return Element::cast(name);
}

PFSinkFW::PFSinkFW() 
{
	_god = NULL;
}

PFSinkFW::~PFSinkFW()
{
}

int PFSinkFW::configure(Vector<String> &conf, ErrorHandler *errh)
{
	D_START_FUNCTION;
	String classify_algorithm;	
        if (cp_va_kparse(conf, this, errh,
                                /* device parameter setting */
				"GOD", cpkP, cpElement, &_god,

                                cpEnd) < 0)
	{
		D_END_FUNCTION;
                return -1;
	}

	D_END_FUNCTION;
	return 0;
}

/*
 * call frequency: per-packet
 * desc: push
 */
void PFSinkFW::push(int i, Packet *p)
{
	D_START_FUNCTION;

	if( i == 0 ) {
		if( _god ) { 
			_god->papmo.do_monitor(
					COMPOSED_TRACE_TAG_NO_TAG ,
					COMPOSED_TRACE_POS_L4_IN,
					p,
					NULL,
					NULL,
					NULL,
					NULL );
		}

		output(0).push(p);
	} else {
		click_chatter("Error! Supposed not to receive this packet\n");
		p->kill();
	}
	D_END_FUNCTION;
}

#include <click/vector.hh>
CLICK_ENDDECLS
EXPORT_ELEMENT(PFSinkFW)

