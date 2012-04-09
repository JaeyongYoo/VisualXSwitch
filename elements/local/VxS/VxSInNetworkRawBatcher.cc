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
#include "VxSInNetworkRawBatcher.hh"
#include "VxSInNetworkTaskQueue.hh"
#include "../OpenFlow/lib/ofpbuf.hh"
#include "../OpenFlow/datapath.hh"
#include "STIP.h"
        
CLICK_DECLS

int g_delete_segment_cnt = 0;
int g_delete_task_cnt = 0;
int g_malloc_segment_cnt = 0;
int g_free_segment_cnt = 0;


int g_new_segment_cnt = 0;
int g_new_task_cnt = 0;

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
	g_malloc_segment_cnt ++;
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
		g_free_segment_cnt ++;
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


/**
 * implementation of VxSInNetworkRawBatcher
 */
VxSInNetworkRawBatcher::VxSInNetworkRawBatcher(const struct sw_flow_key *fid, 
	VxSInNetworkTaskQueue *tq_in, VxSInNetworkTaskQueue *tq_out) 
	: VxSInNetworkFlowBatcher( fid, tq_in, tq_out )
{
	_segment_size = 0;
	strncpy( _media_type_name, media_type_name[VXS_MEDIA_TYPE_RAW], VXS_MAX_FLOW_TYPE_NAME );
}

VxSInNetworkRawBatcher::~VxSInNetworkRawBatcher()
{
}
void VxSInNetworkRawBatcher::stip_initiation_packet_received(struct stip_initiation_header *sihdr)
{
	if( _initiated == true )
	{
		click_chatter("New flow arrived: re-initiated\n");
	}

	int block_width;
	int block_height;

	if( 1 ) {
		click_chatter("JYD: INIT ==================================\n");
		click_chatter("reading initiation header info.\n");
		click_chatter("\tsrc_video=%d\n", sihdr->src_video);
		click_chatter("\tproc_video=%d\n", sihdr->proc_video);
		click_chatter("\tsrc_px_width=%d\n", sihdr->src_px_width);
		click_chatter("\tsrc_px_height=%d\n", sihdr->src_px_height);
		click_chatter("\tsrc_bpb=%d\n", sihdr->src_Bpb);
		click_chatter("\tproc_px_width=%d\n", sihdr->proc_px_width);
		click_chatter("\tproc_px_height=%d\n", sihdr->proc_px_height);
		click_chatter("\tproc_bpb=%d\n", sihdr->proc_Bpb);
		click_chatter("\tframe_width=%d\n", sihdr->frame_width);
		click_chatter("\tframe_height=%d\n", sihdr->frame_height);
		click_chatter("\tmax_video_frame=%d\n", sihdr->max_video_frame);
		click_chatter("\tvideo_fps=%d\n", sihdr->video_fps); 
		click_chatter("JYD: INIT ==================================\n");
		click_chatter("\n\n");
	}
	_initiated = true;
	

	/* FIXME: at this moment, we assume YUV 2bytes per pixel */
	block_width = sihdr->frame_width/sihdr->src_px_width;
	block_height = sihdr->frame_height/sihdr->src_px_height;

	/* extract video-related info */
	_max_pixel_blocks = block_width * block_height;
	_src_Bpb = sihdr->src_Bpb;
	_proc_Bpb = sihdr->proc_Bpb;
	_frame_height = sihdr->frame_height;
	_frame_width = sihdr->frame_width;

	/* by default, we set the segment_size as the byte-size of a video frame */
	_segment_size = _max_pixel_blocks * _src_Bpb;
	click_chatter("[this=%p] _segment_size %d  = %d * %d\n", this, _segment_size, _max_pixel_blocks, _src_Bpb );

}

void VxSInNetworkRawBatcher::stip_data_packet_received(struct stip_transport_header *shdr)
{
	/* we can do the following because stip header size is constant */
	uint8_t *data = (uint8_t *)(shdr+1);
	uint32_t data_size = shdr->pblock_count * _src_Bpb; /* source Bytes per block (pixel) */

	VxSInNetworkRawSegment *s;
	/* 
	 * prepare a segment @s
	 */
	if( _segments.size() == 0 ) {
		/* we have no segment at all, create one */ 
		s = createNewSegment();
		_segments.push_back( s );
#if JYD == 1
		click_chatter("JYD ====> Push segment: %p\n", s );
		list_all_segments();
#endif

	} else {
		bool print_this = false;	
		s = _segments.back();
		if( s->isFull() ) {
			s = createNewSegment();
			print_this = true;

		} else {
			/* XXX: to make the consistency of "push_back" */
			_segments.pop_back();
		}
		_segments.push_back( s );

		if( print_this ) {
#if JYD == 1
			click_chatter("JYD ====> Push segment: %p\n", s );
			list_all_segments();
#endif
		}
	}


	/* insert data into segment */
	uint32_t len = s->push( data, data_size );

	if( len != data_size ) {
		/* not all data has been copied, 
		 * create new segment and complete 
		 * the copy */
		int residual = data_size - len;
		s = createNewSegment();
		s->push( data + len, residual );
		_segments.push_back( s );

#if JYD == 1
		click_chatter("JYD ====> Push segment: %p\n", s );
		list_all_segments();
#endif

	}


}

VxSInNetworkRawSegment * VxSInNetworkRawBatcher::createNewSegment() 
{

	g_new_segment_cnt ++;
	VxSInNetworkRawSegment *s = new VxSInNetworkRawSegment( _segment_size );
	s->setBytePerPixelBlocks( _src_Bpb );
	s->setWidthHeight( _frame_width, _frame_height );
	
	/* 
	 * for copying the action headers to a segment 
	 * FIXME: this would be better suitable in base-class!
	 */
	if( _action_headers )
		s->setActionHeader( (const uint8_t *)_action_headers, _action_len );
	else 
		click_chatter("Error: _action_header is NULL!\n");

	return s;
}

int VxSInNetworkRawBatcher::pushPacket(struct ofpbuf *ob, const struct ofp_action_header *ah, int actions_len)
{
	int re = VxSInNetworkFlowBatcher::pushPacket( ob, ah, actions_len );
	if( re ) return re;

	/* sanity check */
	if( ob->l2 == NULL || ob->l3 == NULL || ob->l4 == NULL || ob->l7 == NULL ) {
		click_chatter("Error: packet @ob has null for network-headers\n");
		return -1;
	}

	/* although @ob is openflow-defined packet structure, we use 
	 * click-network-headers for the convenience of writing codes */
//	click_ether *ether = (click_ether *)ob->l2; /* delete me if not using */
//	click_ip *ip = (click_ip *)ob->l3;
//	click_udp *udp = (click_udp *)ob->l4;

    struct  stip_common_header *schdr = (struct stip_common_header *)(ob->l7);

	/* STIP protocol version check */
	if( schdr->version != 0x01 ) {
		click_chatter("Error: unsupported stip version: %d\n", schdr->version );
		return -1;
	}

	/* if this packet is initiation packet */
	if( schdr->hdr_len == sizeof(struct stip_initiation_header) ) {
		stip_initiation_packet_received( (struct stip_initiation_header *)schdr );
	} else {
		if( _initiated == false ) {
			click_chatter("We may miss the initiation packet for DXT\n");
			return -1;
		}
		stip_data_packet_received( (struct stip_transport_header *)schdr );
	}

	/* at every packet arrival interval, we **try** to send a task */
	sendToInputTaskQueue(ob);

	return 0;
}

int VxSInNetworkRawBatcher::sendToInputTaskQueue(struct ofpbuf *ob)
{
	/* FIXME: here, we have to determine whether performing rate control */
	while( _segments.size() > 2  ) {

		/* create a task */
		
		g_new_task_cnt ++;
		VxSInNetworkTask *task = new VxSInNetworkTask();
		if( task == NULL ) {
			click_chatter("Error: out of memory while allocating VxSInNetworkTask\n");
			return -1;
		}
		int network_header_len = ((uint8_t *)ob->l7 - (uint8_t *)ob->l2) + sizeof(struct stip_transport_header);
		
		VxSInNetworkRawSegment *r_seg = _segments.front();


		task->set( r_seg, this, ob, network_header_len  );

#if JYD == 1
		click_chatter("JYD ====> before PoP segment: %p\n", r_seg );
		list_all_segments();
#endif
	
		/* remove the segment */
		_segments.pop_front();
		
#if JYD == 1
		click_chatter("JYD ====> PoP segment: %p\n", r_seg );
		list_all_segments();
#endif
	

//		task->print_to_chatter();

		/* 
		 * send to task queue 
		 * XXX: be aware that this is a blocking function 
		 */
		_task_queue_incoming->pushTask( task );
	}
	return 0;
}

/*
 * Desc: we pop a task from outgoing task queue 
 *       and packetize it and send to datapath @dp
 *
 * Return value is the number of residual tasks 
 */
int VxSInNetworkRawBatcher::recvFromTaskQueue(Datapath *dp)
{
	if( _task_queue_outgoing->size() == 0 ) return 0;
	/* here, we have to determine how many tasks do we send?
	 * note that the rate is really important here */
	VxSInNetworkTask *task = _task_queue_outgoing->popTask();
	VxSInNetworkRawSegment *rawSegment = (VxSInNetworkRawSegment *) task->getSegment();

	Packet *p = NULL;
	p = rawSegment->packetize(1400, task->getNetworkHeader(), task->getNetworkHeaderLen() );


	Packet *tmp;
	int iii = 0;
	while( p != NULL ) {
		tmp = p;
		p = p->next();
		tmp->set_next(NULL);
		dp->dp_output_port( tmp, task->getInPort(), task->getOutPort(), 0 );
		iii++;
	}

	// need to delete the task 
	/* FIXME: if I delete these two deletes, seg-fault happens !!! */
	/* FIXME: FIXME: FIXME: Jaeyong, please fix this immediately !!!! */
	/* XXX: I think it is fixed, delete all these comments!! */

#if JYD == 1
	click_chatter("JYD ===> delete a segment %p\n", rawSegment );
#endif
	g_delete_task_cnt ++;
	g_delete_segment_cnt ++;
	delete rawSegment;
	delete task;
	return _task_queue_outgoing->size();
}

/*
 * Debugging routines 
 */
void VxSInNetworkRawBatcher::list_all_segments()
{
	click_chatter("===> listing all segments\n");
	std::list<VxSInNetworkRawSegment *>::iterator ii;
        for( ii = _segments.begin(); ii != _segments.end(); ii ++ ) {
		VxSInNetworkRawSegment *s = *ii;
		click_chatter("\t%p\n", s );
	}
	click_chatter("\tEND\n\n");
}

CLICK_ENDDECLS
ELEMENT_PROVIDES(VxSInNetworkRawBatcher)
