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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "FlowNetUtils.hh"
CLICK_DECLS

void sprintf_mac( char* buf, uint8_t* macaddr)
{
	sprintf( buf, "%02x:%02x:%02x:%02x:%02x:%02x",
			macaddr[0],
			macaddr[1],
			macaddr[2],
			macaddr[3],
			macaddr[4],
			macaddr[5]);
}
void printf_ether( click_ether* e)
{
	char buf[1000];
	sprintf_ether( buf, e );
	printf("%s\n", buf);

}
void printf_ip( click_ip* i)
{
	char buf[1000];
	sprintf_ip( buf, i );
	printf("%s\n", buf);

}
void printf_udp( click_udp* u)
{
	char buf[1000];
	sprintf_udp( buf, u );
	printf("%s\n", buf);

}

void printf_ip( click_ip* );
void printf_udp( click_udp* );

void sprintf_ether( char* buf, click_ether* ethdr )
{
	sprintf( buf, "%02x:%02x:%02x:%02x:%02x:%02x ==> %02x:%02x:%02x:%02x:%02x:%02x [ %04x ]",
		ethdr->ether_shost[0],
		ethdr->ether_shost[1],
		ethdr->ether_shost[2],
		ethdr->ether_shost[3],
		ethdr->ether_shost[4],
		ethdr->ether_shost[5],

		ethdr->ether_dhost[0],
		ethdr->ether_dhost[1],
		ethdr->ether_dhost[2],
		ethdr->ether_dhost[3],
		ethdr->ether_dhost[4],
		ethdr->ether_dhost[5],
		ethdr->ether_type );
}

void sprintf_ip( char* buf, click_ip* ip )
{
	sprintf( buf, "v:%d l:%d tos:%d len:%d id:%d off:%d ttl:%d proto:%d %d.%d.%d.%d ==> %d.%d.%d.%d",
		ip->ip_v,
		ip->ip_hl,
		ip->ip_tos,
		ip->ip_len,
		ip->ip_id,
		ip->ip_off,
		ip->ip_ttl,
		ip->ip_p,
		*(((uint8_t*)(&ip->ip_src))+0),
		*(((uint8_t*)(&ip->ip_src))+1),
		*(((uint8_t*)(&ip->ip_src))+2),
		*(((uint8_t*)(&ip->ip_src))+3),

		*(((uint8_t*)(&ip->ip_dst))+0),
		*(((uint8_t*)(&ip->ip_dst))+1),
		*(((uint8_t*)(&ip->ip_dst))+2),
		*(((uint8_t*)(&ip->ip_dst))+3) );
}


void sprintf_udp( char* buf, click_udp* udp )
{
	sprintf( buf, "%d ==> %d",
		htons(udp->uh_sport),
		htons(udp->uh_dport) );
}


/* convert the timeval to printable format */
void sprintf_time( char* buf, struct timeval* tv )
{
	int temp, j;
	temp = tv->tv_usec;
	temp*=10;
	sprintf(buf, "%u.", (unsigned int)tv->tv_sec);
	for(j=0; j<6; j++)
	{
		temp /= 10;
		if( temp == 0 ) sprintf(buf+strlen(buf), "0");
	}
	sprintf(buf+strlen(buf), "%u ", (unsigned int)tv->tv_usec);
}

void print_now()
{
	char buf[100];
	struct timeval tv;
	gettimeofday( &tv, NULL );
	sprintf_time( buf, &tv );
	printf("%s\n", buf);
}
/* supposed to receive IP hdr including packet */
void checksumIP( Packet* p, int offset )
{
        click_ip* iphdr = (click_ip*)(p->data() + offset);
        unsigned hlen;
        iphdr->ip_len = htons(p->length() - offset);
        hlen = iphdr->ip_hl << 2;
        iphdr->ip_sum = 0;
        iphdr->ip_sum = click_in_cksum((unsigned char *)iphdr, hlen);
}

/* supposed to receive IP hdr including packet */
void checksumUDP( Packet* p, int offset )
{
        click_ip* iphdr = (click_ip*) (p->data() + offset);
        click_udp* udphdr = (click_udp*)(p->data() + sizeof(click_ip) + offset);
        unsigned csum;
        udphdr->uh_ulen = htons(p->length() - sizeof(click_ip) - offset);
        udphdr->uh_sum = 0;
        csum = click_in_cksum((unsigned char *)udphdr, ntohs(iphdr->ip_len) - sizeof(click_ip));
        udphdr->uh_sum = click_in_cksum_pseudohdr(csum, iphdr, ntohs(iphdr->ip_len) - sizeof(click_ip));
}

/* computes the difference of timeval */
int64_t timevaldiff( struct timeval* starttime, struct timeval *finishtime )
{
        int64_t usec;
        usec = ((int64_t)finishtime->tv_sec - starttime->tv_sec)*1000000l;
        usec += (finishtime->tv_usec - starttime->tv_usec);
        return usec;
}


void sprintf_flow( char* buf, uint32_t src, uint32_t dst, uint16_t sport, uint16_t dport )
{
	struct in_addr s, d;
	memcpy( &s, &src, sizeof(src) );
	memcpy( &d, &dst, sizeof(src) );
	sprintf_flow( buf, s, d, sport, dport );
}
void sprintf_flow( char* buf, struct in_addr src, struct in_addr dst, uint16_t sport, uint16_t dport )
{
	char sz_src[30];
	strcpy( sz_src, inet_ntoa(src) );
	sprintf( buf, "%s:%d ==> %s:%d", sz_src, htons(sport), inet_ntoa(dst), htons(dport) );
}
void sprintf_flow( char* buf, IPAddress src, IPAddress dst, uint16_t sport, uint16_t dport )
{
	sprintf_flow( buf, src.addr(), dst.addr(), sport, dport );
}

CLICK_ENDDECLS
ELEMENT_PROVIDES(DummyClass)

