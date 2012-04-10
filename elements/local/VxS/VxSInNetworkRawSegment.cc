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
#include <clicknet/ether.h>
#include <clicknet/ip.h>
#include <clicknet/udp.h>
#include <string.h>
#include <stdlib.h>
#include "VxSInNetworkRawSegment.hh"
#include "VxSInNetworkTaskQueue.hh"
#include "../OpenFlow/lib/ofpbuf.hh"
#include "../OpenFlow/datapath.hh"
#include "STIP.h"
        
CLICK_DECLS

/* XXX: if you include this into FlowNet build tree, delete the below two functions */
/* supposed to receive IP hdr including packet */
static void checksumIP( Packet* p, int offset )
{
        click_ip* iphdr = (click_ip*)(p->data() + offset);
        unsigned hlen;
        iphdr->ip_len = htons(p->length() - offset);
        hlen = iphdr->ip_hl << 2;
        iphdr->ip_sum = 0;
        iphdr->ip_sum = click_in_cksum((unsigned char *)iphdr, hlen);
}

/* supposed to receive IP hdr including packet */
static void checksumUDP( Packet* p, int offset )
{
        click_ip* iphdr = (click_ip*) (p->data() + offset);
        click_udp* udphdr = (click_udp*)(p->data() + sizeof(click_ip) + offset);
        unsigned csum;
        udphdr->uh_ulen = htons(p->length() - sizeof(click_ip) - offset);
        udphdr->uh_sum = 0;
        csum = click_in_cksum((unsigned char *)udphdr, ntohs(iphdr->ip_len) - sizeof(click_ip));
        udphdr->uh_sum = click_in_cksum_pseudohdr(csum, iphdr, ntohs(iphdr->ip_len) - sizeof(click_ip));
}


/**
 * implementation of VxSInNetworkRawSegment
 */
VxSInNetworkRawSegment::VxSInNetworkRawSegment(uint32_t seg_size)
{
	_segment_size = seg_size;
	_segment = (uint8_t *)malloc( seg_size );
	if( _segment == NULL ) {
		click_chatter("Error: out of memory: requesting size=%d\n", seg_size);
		exit(-1);
	}
#if JYD == 1 
	click_chatter("JYD =====> creating a segment: _segment = %p\n", _segment );
#endif

	_written_size = 0;
}
VxSInNetworkRawSegment::~VxSInNetworkRawSegment()
{
	if( _segment ) {
#if JYD == 1 
		click_chatter("JYD =====> freeing a segment: _segment = %p\n", _segment );
#endif
		free( _segment );
		_segment = NULL;
	}
}

uint32_t VxSInNetworkRawSegment::push(uint8_t *data, int size)
{
	int residual_size = _segment_size - _written_size;
	int copy_size = size < residual_size ? size : residual_size;

	memcpy( _segment + _written_size, data, copy_size );
	_written_size += copy_size;

	/* is it an error case? */
	if( _written_size % _Bpb ) {
		click_chatter("Warning: written size is not the division of src_Bpb\n");
	}

	return copy_size;
}

uint32_t VxSInNetworkRawSegment::prepareSegment( uint32_t required_size )
{
	if( _segment_size >= required_size ) return 0;
	/* if the size is smaller, then we free and re-allocate */
	free( _segment );
	_segment = (uint8_t *) malloc( required_size );
	_segment_size = required_size;
	if( _segment > 0 ) return 0;
	return 1;
}

uint32_t VxSInNetworkRawSegment::getNumberOfPackets(uint32_t packet_size) 
{
	uint32_t contents= int(packet_size / _Bpb) * _Bpb;
	return _written_size / contents + (_written_size%contents ? 1 : 0); 
}
Packet * VxSInNetworkRawSegment::packetize(uint32_t data_size, uint8_t *network_header, uint32_t network_header_len )
{
	Packet *p_head = NULL;
	Packet *p_tail = NULL;
	/* 
	 * network_header_len should include stip_transport_header 
	 */
	
	int dxt_send_unit = data_size;
	int packet_size = network_header_len + dxt_send_unit;
        uint32_t sent_size = 0;
        int sent_blocks = 0;
        int ip_mtu = 1500;
        int residual_packet_buffer = ip_mtu - 
			sizeof(click_ip) - sizeof(click_udp) - sizeof(struct stip_transport_header);
        int pixel_blocks_per_packet = residual_packet_buffer / _Bpb;

	if( packet_size >= ip_mtu ) {
		click_chatter("Error: packet size is larger than MTU %d\n", ip_mtu);
		return NULL;
	}

        while( sent_size != _written_size ) {
                if( sent_size > _written_size ) {
                        printf("Critical Error: size does not match (%d %d)\n", sent_size, _written_size );
                        break;
                }
                if( _written_size < (sent_size + data_size) ) {
                        /* we are reaching the last packet 
                         * and the last packet may not be the same number of pixel_blocks_per_packet */

                        /* count the last blocks */
                        int last_blocks = (_written_size - sent_size)/_Bpb;

                        /* change the variables accordingly */
                        pixel_blocks_per_packet = last_blocks;
                        dxt_send_unit = pixel_blocks_per_packet * _Bpb;
                        packet_size = network_header_len  + dxt_send_unit;
                }

                /* forging a packet */
                WritablePacket *p = Packet::make( 0, /* head room */
                        NULL, /* data */
                        packet_size, /* data size */
                        0); /* tailroom */

		if( p_head == NULL ) {
			p_head = p;
			p_tail = p;
		}
		p_tail->set_next(p);
		p_tail = p;

		uint32_t stip_offset = network_header_len - sizeof(struct stip_transport_header);
                struct stip_transport_header *sthdr = (struct stip_transport_header *)(p->data() + stip_offset);
                uint8_t *data = p->data() + network_header_len;

                /* fillup data and headers*/
		memcpy( p->data(), network_header, network_header_len );
		
                /* update stip transport header */
                sthdr->pblock_idx = sent_blocks;
                sthdr->pblock_count = pixel_blocks_per_packet;

                memcpy( data,
                        _segment + sent_size,
                        dxt_send_unit );


                checksumIP( p, sizeof(click_ether) );
                checksumUDP( p, sizeof(click_ether) );

                sent_blocks += pixel_blocks_per_packet;
                sent_size += dxt_send_unit;

	}

	p_tail->set_next(NULL);
	return p_head;
}

VxSInNetworkSegment * VxSInNetworkRawSegment::clone()
{
	VxSInNetworkRawSegment *rsaw = new VxSInNetworkRawSegment( _segment_size );
	rsaw->copy( this );
	return rsaw;
}

void VxSInNetworkRawSegment::copy(VxSInNetworkRawSegment *raw)
{
	memcpy( _segment, raw->getSegment(), _segment_size );
	_written_size = raw->getWrittenSize();
	_Bpb = raw->getBytePerPixelBlocks();
	_height = raw->getHeight();
	_width = raw->getWidth();

	/* 
	 * copy the action list 
	 * NOTE: notice that how we handle @_action_header_program_counter
	 * it is a bit tricky 
	 */
	_action_len = raw->getActionLen();
	memcpy( _action_header, raw->getActionHeader(), VXS_MAX_ACTION_HEADER );
	if( raw->getActionHeaderProgramCounter() == NULL ) {
		_action_header_program_counter = NULL;
	} else {
		_action_header_program_counter = _action_header + raw->getActionOffset();
	}
}

CLICK_ENDDECLS
ELEMENT_PROVIDES(VxSInNetworkRawSegment)
