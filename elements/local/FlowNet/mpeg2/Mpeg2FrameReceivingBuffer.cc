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
#include <clicknet/ether.h>
#include <clicknet/udp.h>

#include "Mpeg2FrameReceivingBuffer.hh"

CLICK_DECLS

/********************************************************************
 * class for FrameReceivingBuffer
 ********************************************************************/
FrameReceivingBuffer::FrameReceivingBuffer(int ft, int fi, int ppf)
{
	pHead = NULL;
	pNext = NULL;
	frametype = ft;
	frameindex = fi;
	pkts_per_frame = ppf;
	total_pkts = 0;

	/* timestamp at this moment */
	gettimeofday( &tv_first_packet_received, NULL );
}


void FrameReceivingBuffer::reset(uint16_t* , int *)
{
	/* kill all the packets */
	while( pHead )
	{
		Packet* tmp;
		tmp = pHead->next();
		pHead->kill();
		pHead = tmp;
	}
	pHead = NULL;
	frametype = 0;
	frameindex = 0;
	pkts_per_frame = 0;
	total_pkts = 0;

}

int FrameReceivingBuffer::enque(Packet* p)
{
	/* find the correct place according to pkts_index */
	click_ether* ether = (click_ether*)p->data();
        click_ip* iphdr = (click_ip*)(ether+1);
        click_udp* udphdr = (click_udp*)(iphdr+1);
        struct bpadapt_header* bphdr_in = (struct bpadapt_header*)(udphdr+1);
	struct bpadapt_header* bphdr_next;
	int pkts_index_in = bphdr_in->pkts_index;
	int pkts_index_next;
	Packet* pNext;

	pNext = pHead;

	if( pHead == NULL ) pHead = p;
	else
	{
		/* set the link in the order of pkts_index */
		/* XXX: Do we really need this ? */
		do {
			ether = (click_ether*)pNext->data();
			iphdr = (click_ip*)(ether+1);
			udphdr = (click_udp*)(iphdr+1);
			bphdr_next = (struct bpadapt_header*)(udphdr+1);
			pkts_index_next = bphdr_next->pkts_index;

			if( pkts_index_next > pkts_index_in )
			{
				/* link dually */
				if( pHead == pNext ) { /* if it is head */
					pHead = p;
					pHead->set_next( pNext );
					pNext->set_prev( pHead );
					break;
				} else { 
					/* it is in the middle */
					p->set_prev( pNext->prev() );
					p->set_next( pNext );

					pNext->prev()->set_next( p );
					pNext->set_prev( p );
					break;
				}
			}
			if( pNext->next() == NULL ) { /* then put the packet to the tail */
				pNext->set_next( p );
				p->set_prev( pNext );
				break;
			}
			pNext = pNext->next();
		} while( pNext );
	}
	total_pkts ++;
	return 0;
}

Packet* FrameReceivingBuffer::deque() 
{
	Packet* tmp = NULL;
	if( pHead )
	{
		tmp = pHead;
		pHead = pHead->next();
	}
	return tmp;
}
double FrameReceivingBuffer::received_ratio()
{
	return (double)total_pkts/(double)pkts_per_frame;
}
int FrameReceivingBuffer::missing_packets_count()
{
	return pkts_per_frame - total_pkts;
}
void FrameReceivingBuffer::dump_buffer(FILE* fp, char* buf )
{
	if( fp ) {
		fprintf(fp, "[%d] [%d] [%d] [%d] ", 
			frametype, 
			frameindex,
			pkts_per_frame,
			total_pkts );
	} else if( buf ) {

		sprintf( buf, "frametype=%u frameindex=%u pkts_per_frame=%u total_pkts=%u ratio=%f", 
			frametype,
			frameindex,
			pkts_per_frame,
			total_pkts,
			received_ratio() );
	}
}
int FrameReceivingBuffer::empty()
{
	return pHead == NULL;
}

int FrameReceivingBuffer::is_same_frameindex(Packet* p)
{
	click_ether* ether = (click_ether*)p->data();
        click_ip* iphdr = (click_ip*)(ether+1);
        click_udp* udphdr = (click_udp*)(iphdr+1);
        struct bpadapt_header* bphdr = (struct bpadapt_header*)(udphdr+1);
	
	return bphdr->frameindex == frameindex;
}

CLICK_ENDDECLS
ELEMENT_PROVIDES(Mpeg2FrameReceivingBuffer)
