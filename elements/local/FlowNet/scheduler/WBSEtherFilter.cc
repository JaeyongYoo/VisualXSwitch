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
CLICK_DECLS
#include <clicknet/ether.h>
#include "WBSEtherFilter.hh"

WBSEtherFilter::WBSEtherFilter()
{
}

WBSEtherFilter::~WBSEtherFilter()
{
}

int
WBSEtherFilter::configure(Vector<String> &conf, ErrorHandler *errh )
{
	int res;
/*	res = cp_va_parse(conf, this, errh, cpEtherAddress, "my EtherAddress", &myEther, 0 ); */
	res = cp_va_kparse(conf, this, errh, "my EtherAddress", cpkP, cpEtherAddress, &myEther, cpEnd ); 
	return 0;
}

void
WBSEtherFilter::push(int,  Packet *p)
{
	struct click_ether* ether = (struct click_ether*)(p->data());
	unsigned char broadcast[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

	/* to accept rssi-through monitoring, we just compare 5 bytes */ 
	if(	memcmp( ether->ether_dhost+1, myEther.data()+1, 5) == 0 ||
		memcmp( ether->ether_shost, myEther.data(), 6) == 0 ||
		memcmp( ether->ether_dhost+1, broadcast+1, 5) == 0 )
	{

		output(0).push(p);
	}
	else
	{
		 output(1).push(p);
	}

}

CLICK_ENDDECLS
EXPORT_ELEMENT(WBSEtherFilter)
ELEMENT_MT_SAFE(WBSEtherFilter)
