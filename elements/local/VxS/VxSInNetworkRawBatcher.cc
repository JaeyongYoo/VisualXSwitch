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
#include "../OpenFlow/lib/dp_act.hh"
#include "../OpenFlow/datapath.hh"
#include "STIP.h"
        
CLICK_DECLS

/**
 * implementation of VxSInNetworkRawBatcher
 */
VxSInNetworkRawBatcher::VxSInNetworkRawBatcher(Datapath *dp, const struct sw_flow_key *fid, 
	VxSInNetworkTaskQueue *tq_in, VxSInNetworkTaskQueue *tq_out) 
	: VxSInNetworkFlowBatcher( dp, fid, tq_in, tq_out )
{
	_segment_size = 0;
	strncpy( _media_type_name, media_type_name[VXS_MEDIA_TYPE_RAW], VXS_MAX_FLOW_TYPE_NAME );
}

VxSInNetworkRawBatcher::~VxSInNetworkRawBatcher()
{
}

void VxSInNetworkRawBatcher::stip_initiation_packet_received(struct stip_initiation_header *sihdr, 
	struct ofpbuf *ob, const struct ofp_action_header *ah, int actions_len)
{
	if( _initiated == true ) {
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

	stip_process_initiation_packet( sihdr, ob, ah, actions_len );
}

void VxSInNetworkRawBatcher::stip_process_initiation_packet(struct stip_initiation_header *sihdr, 
	struct ofpbuf *ob, const struct ofp_action_header *actions, int actions_len)
{
        uint8_t *p = (uint8_t *)actions;
	while (actions_len > 0) {
		struct ofp_action_header *ah = (struct ofp_action_header *)p;
		size_t len = htons(ah->len);

		switch(ntohs(ah->type)) {
			case OFPAT_OUTPUT: 
			{
				struct ofp_action_output *oa = (struct ofp_action_output *)ah;
				int32_t out_port = ntohs(oa->port);
				uint16_t in_port = ntohs(_flow_id.flow.in_port);

				Packet* packet = ofpbuf_to_packet( ob );

				/* recompute checksum here */
				/* FIXME: do not check sum here, we should do at VXS FrameResize */
		                checksumIP_v2( packet, sizeof(click_ether) );
        		        checksumUDP_v2( packet, sizeof(click_ether) );

				_datapath->dp_output_port(packet, in_port, out_port, 0);

				break;
			}
			case OFPAT_VXS_DXTComp:
			{
				break;
			}
			case OFPAT_VXS_DXTDecomp:
			{
				break;
			}
			case OFPAT_VXS_FrameResize:
			{
				/* TODO: also re-do the UDP/IP checksum */
				
				/* TODO: frame resize should be paramatized */
				sihdr->frame_width /= 2;
				sihdr->frame_height /= 2;
				break;
			}
			case OFPAT_VXS_YUV2RGB:
			{
				break;
			}
			case OFPAT_SET_DL_DST:
			{
				set_dl_addr(NULL, ob, &_flow_id, ah, htons(ah->len));
				break;
			}
			case OFPAT_SET_NW_DST:
			{
				set_nw_addr(NULL, ob, &_flow_id, ah, htons(ah->len));
				break;
			}
			case OFPAT_SET_TP_DST:
			{
				set_tp_port(NULL, ob, &_flow_id, ah, htons(ah->len));
				break;
			}
			default:
				break;
		}
		p += len;
		actions_len -= len;
	}
}

void VxSInNetworkRawBatcher::stip_data_packet_received(struct stip_transport_header *shdr)
{
	if( shdr->pblock_idx == 0 ) { /* this is the first packet */
		struct timeval tv;
		gettimeofday( &tv, NULL );
		click_chatter("FRAME_FIRST_PACKET_IN %d.%06d %d\n", 
			tv.tv_sec, tv.tv_usec,
			shdr->frame_idx );
	}

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
	} else {
		s = _segments.back();
		if( s->isFull() ) {
			s = createNewSegment();
		} else {
			/* XXX: to make the consistency of "push_back" */
			_segments.pop_back();
		}
		_segments.push_back( s );
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
	}
}

VxSInNetworkRawSegment * VxSInNetworkRawBatcher::createNewSegment() 
{
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

/**
 * pushPacket function returns 0 when success (need to free packet by returning 1)
 *			returns 2 when success (need to forward the packet)
 * 			returns -1 when failure 
 * TODO: make 0, -1, 2 kind of constants as macros
 */ 
int VxSInNetworkRawBatcher::pushPacket(struct ofpbuf *ob, const struct ofp_action_header *ah, int actions_len)
{
	struct  stip_common_header *schdr;
	int re = VxSInNetworkFlowBatcher::pushPacket( ob, ah, actions_len );
	if( re ) return re;

	/* sanity check */
	if( ob->l2 == NULL || ob->l3 == NULL || ob->l4 == NULL || ob->l7 == NULL ) {
		click_chatter("Error: packet @ob has null for network-headers\n");
		return -1;
	}

	schdr = (struct stip_common_header *)(ob->l7);

	/* STIP protocol version check */
	if( schdr->version != 0x01 ) {
		click_chatter("Error: unsupported stip version: %d\n", schdr->version );
		return -1;
	}

	/* if this packet is initiation packet */
	if( schdr->hdr_len == sizeof(struct stip_initiation_header) ) {
		stip_initiation_packet_received( (struct stip_initiation_header *)schdr, ob, ah, actions_len);
		return 0; /* this is a nasty hardcode for handling initiation packet */
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
	/* TODO: here, we have to determine whether performing rate control */
	while( _segments.size() > 2  ) {

		/* create a task */
		VxSInNetworkTask *task = new VxSInNetworkTask();
		if( task == NULL ) {
			click_chatter("Error: out of memory while allocating VxSInNetworkTask\n");
			return -1;
		}
		int network_header_len = ((uint8_t *)ob->l7 - (uint8_t *)ob->l2) + sizeof(struct stip_transport_header);
		
		VxSInNetworkRawSegment *r_seg = _segments.front();

		task->set( r_seg, this, ob, network_header_len  );
	
		/* remove the segment */
		_segments.pop_front();
		
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
int VxSInNetworkRawBatcher::recvFromTaskQueue()
{
	if( _task_queue_outgoing->size() == 0 ) return 0;
	/* here, we have to determine how many tasks do we send?
	 * note that the rate is really important here */
	VxSInNetworkTask *task = _task_queue_outgoing->popTask();
	VxSInNetworkRawSegment *rawSegment = (VxSInNetworkRawSegment *) task->getSegment();

	Packet *p = NULL;
	p = rawSegment->packetize(1400, task->getNetworkHeaders(), task->getNetworkHeaderLen() );

	if( true ) { /* this is the first packet sent */

		struct stip_transport_header *shdr = (struct stip_transport_header *) 
			(task->getNetworkHeaders() + 14 /* ether */ + 20 /* ip */ + 8 /* udp */);
		struct timeval tv;
		gettimeofday( &tv, NULL );
		click_chatter("FRAME_FIRST_PACKET_OUT %d.%06d %d\n", 
			tv.tv_sec, tv.tv_usec,
			shdr->frame_idx );
	}

	Packet *tmp;
	int iii = 0;
	while( p != NULL ) {
		tmp = p;
		p = p->next();
		tmp->set_next(NULL);
		_datapath->dp_output_port( tmp, task->getInPort(), task->getOutPort(), 0 );
		iii++;
	}

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
