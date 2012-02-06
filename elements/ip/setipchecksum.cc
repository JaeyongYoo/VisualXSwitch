/*
 * setipchecksum.{cc,hh} -- element sets IP header checksum
 * Robert Morris
 *
 * Copyright (c) 1999-2000 Massachusetts Institute of Technology
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, subject to the conditions
 * listed in the Click LICENSE file. These conditions include: you must
 * preserve this copyright notice, and you cannot mention the copyright
 * holders in advertising related to the Software without their permission.
 * The Software is provided WITHOUT ANY WARRANTY, EXPRESS OR IMPLIED. This
 * notice is a summary of the Click LICENSE file; the license in that file is
 * legally binding.
 */

#include <click/config.h>
#include "setipchecksum.hh"
#include <click/glue.hh>
#include <clicknet/ip.h>
CLICK_DECLS

SetIPChecksum::SetIPChecksum()
{
}

SetIPChecksum::~SetIPChecksum()
{
}

Packet *
SetIPChecksum::simple_action(Packet *p_in)
{
    if (WritablePacket *p = p_in->uniqueify()) {
	click_ip *ip;
	unsigned plen, hlen;

	if (!p->has_network_header())
	    goto bad;
	plen = p->network_length();
	if (plen < sizeof(click_ip))
	    goto bad;
	ip = p->ip_header();
	hlen = ip->ip_hl << 2;
	if (hlen < sizeof(click_ip) || hlen > plen)
	    goto bad;

	ip->ip_sum = 0;
	ip->ip_sum = click_in_cksum((unsigned char *)ip, hlen);
	return p;

      bad:
	click_chatter("SetIPChecksum: bad lengths");
	p->kill();
    }
    return 0;
}

CLICK_ENDDECLS
EXPORT_ELEMENT(SetIPChecksum)
ELEMENT_MT_SAFE(SetIPChecksum)
