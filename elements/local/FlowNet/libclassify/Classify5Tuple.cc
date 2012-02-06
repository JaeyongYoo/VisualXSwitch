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

#include "Classify5Tuple.hh"


CLICK_DECLS

/*
 * jyyoo debugging : indenting stack depth 
 */
D_DEFINE_EXTERN;


Vc5TupleClassify::Vc5TupleClassify() 
{
	set_name("5TupleClassify");
}

Vc5TupleClassify::~Vc5TupleClassify() 
{
}

int Vc5TupleClassify::classify(const Packet* p, struct FlowID* fid)
{
	D_START_FUNCTION;

	const struct click_ether	*ether;
	const struct click_ip		*ip;	
	const struct click_udp		*udp;	

	if( p == NULL || fid == NULL ) {
		goto error;
	}

	ether 	= (struct click_ether*) p->data();
	ip 	= (struct click_ip*)(ether+1);
	udp	= (struct click_udp*)(ip+1);

	fid->len = 0;
	memcpy(fid->id+fid->len, &(ip->ip_src), sizeof(ip->ip_src));
	fid->len += sizeof(ip->ip_src);
	memcpy(fid->id+fid->len, &(ip->ip_dst), sizeof(ip->ip_dst));
	fid->len += sizeof(ip->ip_dst);
	memcpy(fid->id+fid->len, &(udp->uh_sport), sizeof(udp->uh_sport));
	fid->len += sizeof(udp->uh_sport);
	memcpy(fid->id+fid->len, &(udp->uh_dport), sizeof(udp->uh_dport));
	fid->len += sizeof(udp->uh_dport);
	memcpy(fid->id+fid->len, &(ip->ip_p), sizeof(ip->ip_p));

	fid->classifier = this;	

	if( fid->len >= MAX_FLOWID_LEN ) {
		goto error;
	}

	D_END_FUNCTION;
	return 0;
error:
	D_END_FUNCTION;
	return -1;
}

int Vc5TupleClassify::to_string(const struct FlowID* fid, char* buf, int len)
{
	char str[512];
	struct in_addr *ip_src = (struct in_addr*)fid->id;
	struct in_addr *ip_dst = (struct in_addr*)(ip_src+1);
	uint16_t *sport = (uint16_t*)(ip_dst+1);
	uint16_t *dport = (uint16_t*)(sport+1);
	uint8_t *proto = (uint8_t*)(dport+1);

	sprintf(str, "%s:%u->",        inet_ntoa(*ip_src), htons(*sport) );
	sprintf(str + strlen(str), "%s:%u / %s",  inet_ntoa(*ip_dst), htons(*dport), 
			*proto == 17 ? "udp" : *proto == 6 ? "tcp" : *proto == 1 ? "ping" : "others" );

	if( len > (int)strlen(buf) ) {

		strncpy( buf, str, len );
		return 0;
	}
	return -1;
}

CLICK_ENDDECLS
ELEMENT_PROVIDES(Vc5TupleClassify)
