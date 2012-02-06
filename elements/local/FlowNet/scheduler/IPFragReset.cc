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
#include <click/glue.hh>
#include <click/confparse.hh>
#include <click/error.hh>
#include <click/straccum.hh>
#ifdef CLICK_LINUXMODULE
# include <click/cxxprotect.h>
CLICK_CXX_PROTECT
# include <linux/sched.h>
CLICK_CXX_UNPROTECT
# include <click/cxxunprotect.h>
#endif

#include <clicknet/ether.h>

#include "IPFragReset.hh"


CLICK_DECLS

IPFragReset::IPFragReset()
{
}

IPFragReset::~IPFragReset()
{
}

int
IPFragReset::configure(Vector<String> &, ErrorHandler* )
{
  return 0;
}

Packet *
IPFragReset::simple_action(Packet *p)
{
	/* ip header mangling (put queue length) */
	struct click_ether* ether = (struct click_ether*)(p->data());
	struct click_ip* cip = (struct click_ip*)(p->data() + sizeof( struct click_ether ));
	unsigned hlen;
	hlen = cip->ip_hl << 2;
	
	if( ether->ether_type == ((ETHERTYPE_IP  & 0xFF00) >> 8) && cip->ip_v == 4 )
	{

		if( cip->ip_p == IP_PROTO_UDP ||
				cip->ip_p == IP_PROTO_TCP )
		{

			/* network byte order : 0100-000-000-000 (0x0040)*/
			/* second bit on: means no fragmentation */
			cip->ip_off =0x0040; 

			cip->ip_sum = 0;
			cip->ip_sum = click_in_cksum((unsigned char *)cip, hlen);
		}
	}
  return p;
}

CLICK_ENDDECLS
EXPORT_ELEMENT(IPFragReset)
ELEMENT_MT_SAFE(IPFragReset)
