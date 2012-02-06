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

#include "FlowMpeg2AdaptDecap.hh"
#include "../utils/FlowNetUtils.hh"
CLICK_DECLS


/*
 * jyyoo debugging : indenting stack depth 
 */
D_DEFINE_EXTERN;


/********************************************************************
 * class for FlowMpeg2AdaptDecap 
 ********************************************************************/

void FlowMpeg2AdaptDecap::clear()
{
	frameBufferHead = NULL;
	frameBufferTail = NULL;
	buffer_size = 0;	
	
	memset( &stat, 0, sizeof(stat) );
	/* timestamp the starting point */
	stat.start_timestamp();

	/* stub policies */
	stub_policy_frame_delivery_inorder = 1;
	sp_frame_index = 0;

	/* initialize trace log */
	stat.fp_trace = NULL;


	killed_buffer = 0;
	total_killed_pkts = 0;
	video_started = false;
	adaptivePlayoutFactor = 0;

	last_frame_index = -1;
}

FlowMpeg2AdaptDecap::FlowMpeg2AdaptDecap() 
{

}

FlowMpeg2AdaptDecap::~FlowMpeg2AdaptDecap() 
{

}

void FlowMpeg2AdaptDecap::toString( char* str, int len )
{

	char buf[4096];

	/* get all dropped counts */
	int droped_counts = stat.total_packet_drop_count;
	double timediff;
	struct timeval tv_now;
	gettimeofday( &tv_now, NULL );
	timediff = timevaldiff( &stat.last_update_time, &tv_now);
	stat.last_update_time = tv_now;

	sprintf(buf, "\t[frame drp_ratio:%.2f%%]", 100.0*(double)stat.total_frame_drop_count/(double)stat.total_frames_sent);
	sprintf(buf + strlen(buf), "\t[packet drp_ratio:%.2f%%]", 100.0*(double)stat.total_packet_drop_count/(double)stat.total_packets_sent);
	sprintf(buf + strlen(buf), "\t[recv thr: %.2fMbps]\n\t[killed frames=%d]", 8.0*(double)stat.bytes_received / timediff, killed_buffer );
	sprintf(buf + strlen(buf), "\t[frame buffer size: %d] [adaptivePlayout: %d]\n", buffer_size, adaptivePlayoutFactor);

	/* reset statistics */
	stat.bytes_received = 0;

	FrameReceivingBuffer* next = frameBufferHead;
	while( next )
	{
		droped_counts += next->missing_packets_count();
		next = next->pNext;
	}

	
	/* print video frame status */
/*
	char b[4096];
	next = frameBufferHead;
	while( next ) {

		next->dump_buffer(NULL, b);
		sprintf(buf + strlen(buf),"[%s] ", b);
		next = next->pNext;
		sprintf(buf + strlen(buf), "\n");
	}

	// print the sent video frame index 
	stat.print_frame_index( NULL, b );
	sprintf(buf+strlen(buf), "\t%s\n", b );
*/
	int l = strlen(buf);
	if( len < l ) 
	{
		sprintf(str, "increase the char buffer: %d - %d\n", len, l );
	} else {
		sprintf(str, "%s", buf );
	}

	stat.flush_frame_index();

}

void FlowMpeg2AdaptDecap::print_stat()
{
}


FrameReceivingBuffer* FlowMpeg2AdaptDecap::search_frame_buffer( Packet* p )
{
	FrameReceivingBuffer* next;
	next = frameBufferHead;

	while( next )
	{
		if( next->is_same_frameindex( p ) ) break;
		next = next->pNext;
	}
	return next;
}

bool FlowMpeg2AdaptDecap::checkFrameIntegrity()
{
	FrameReceivingBuffer* next;
	next = frameBufferHead;
	int c=0;
	while( next )
	{
		c++;

		if( next->pNext )
		{
			if( next->pNext->pPrev != next )
			{
				fprintf(stderr, "Error!!!! integrity failed: pointing structure mismatch\n");
				return false;

			}
		}
		next = next->pNext;


		if( next == frameBufferTail ) {
			if( next->pNext != NULL ) {
				fprintf(stderr, "Error!!!! integrity failed: pNext is NULL\n");
				return false;
			}
			if( c != buffer_size-1 )	{
				fprintf(stderr, "Error!!!! integrity failed: buffer size mismatch [%d %d]\n", c, buffer_size - 1);
				return false;
			}
		}
	}
	return true;


}
FrameReceivingBuffer* FlowMpeg2AdaptDecap::create_new_buffer( int ftype, int findex, int pkts )
{
	FrameReceivingBuffer* targetBuffer;

	/* create a new Frame Buffer */
	targetBuffer = new FrameReceivingBuffer(
			ftype, 
			findex,
			pkts );

	buffer_size ++;
	targetBuffer->pNext = NULL;
	targetBuffer->pPrev = NULL;
	
	/* the frame buffer should be sorted by frame index */
	if( frameBufferHead ) { 
		FrameReceivingBuffer* next = frameBufferHead;

		/* find the right position */
		while( next )
		{
			if( (int) next->frameindex < findex )
				break;
			next = next->pNext;
		}

		if( next == NULL ) /* this one goes to tail */
		{
			frameBufferTail->pNext = targetBuffer;
			targetBuffer->pNext = NULL;
			targetBuffer->pPrev = frameBufferTail;
			frameBufferTail = targetBuffer;
		} else if( next == frameBufferHead ) { /* this one goes to the head */
			frameBufferHead->pPrev = targetBuffer;
			targetBuffer->pPrev = NULL;
			targetBuffer->pNext = frameBufferHead;
			frameBufferHead = targetBuffer;
		} else { /* in the middle */

			next->pPrev->pNext = targetBuffer;
			targetBuffer->pPrev = next->pPrev;

			next->pPrev = targetBuffer;
			targetBuffer->pNext = next;
		}
	} else {
		frameBufferHead = frameBufferTail = targetBuffer;
	}

	/* update the frame stat */
	stat.total_frames_sent ++;

	/* update the packet stat */
	stat.total_packets_sent += pkts;

	return targetBuffer;
}


void FlowMpeg2AdaptDecap::disconnect( FrameReceivingBuffer* buffer )
{
	D_START_FUNCTION;
	/* first Disconnect buffer from original doubly linked list structure */
	/* this usually consists of four parts: head & tail, head, tail, and middle */
	/* head & tail */
	if( buffer == frameBufferHead && buffer == frameBufferTail ) {
		frameBufferHead = frameBufferTail = NULL;
	}
	/* head */
	else if( buffer == frameBufferHead ) {
		frameBufferHead = frameBufferHead->pNext;
		frameBufferHead->pPrev = NULL;
	}
	/* tail */
	else if( buffer == frameBufferTail ) {
		frameBufferTail = frameBufferTail->pPrev;
		frameBufferTail->pNext = NULL;
	} 
	/* in the middle */
	else {
		buffer->pPrev->pNext = buffer->pNext;
		buffer->pNext->pPrev = buffer->pPrev;
	}

	D_END_FUNCTION;
}
bool FlowMpeg2AdaptDecap::isInitialFrameBufferReady()
{
	if( video_started ) return true;

	if(buffer_size < MAX_FRAME_BUFFER_SIZE * 0.5 ) return false;

	video_started = true;

	gettimeofday( &tv_video_play_start_time, NULL );

	return true;
}

uint64_t FlowMpeg2AdaptDecap::get_expected_playout_time( int frameindex )
{
	int timegap = 1000000 / MPEG2_FPS;
	return frameindex * timegap;
}

int FlowMpeg2AdaptDecap::flush_one_buffer(const Element* e, papmo* papmo)
{
	if( frameBufferTail )
	{	
		FrameReceivingBuffer* sent;
		sent = frameBufferTail;

		disconnect( sent );

		if( sent->received_ratio() == 1.0 )
		{

			sendFrameToUpperLayer( sent, e, papmo );

		} else {

/* TODO: Better use this one!
			Packet* p = sent->pHead;
			if( papmo ) papmo->do_monitor(
					COMPOSED_TRACE_TAG_MPEG_FRAME ,
					COMPOSED_TRACE_POS_L4_IN,
					p,
					this,
					NULL,
					NULL,
					NULL );
*/


			if( frameBufferTail->get_frametype() == 1 ) 
			{
				fprintf(stderr, "I frame loss\n");
			}

			deleteBuffer( sent );

		}
	}
	return 0;
}


int FlowMpeg2AdaptDecap::enque( Packet* p, const Element* e, papmo* papmo )
{

	click_ether* ether = (click_ether*)p->data();
	click_ip* iphdr = (click_ip*)(ether+1);
	click_udp* udphdr = (click_udp*)(iphdr+1);
	struct bpadapt_header* bphdr = (struct bpadapt_header*)(udphdr+1);

	/* if it is not video frame, send it immediately */
	if( bphdr->frametype == MPEG2_NON_VIDEO ) {
		WritablePacket* p_out;
		p_out = decapsulate_bpadapt( p );
		fprintf(stderr, "sending up control packet: \n");
		e->output(0).push( p_out );
		return 0;
	}

	FrameReceivingBuffer* targetBuffer = search_frame_buffer( p );

	if( targetBuffer == NULL && buffer_size == MAX_FRAME_BUFFER_SIZE ) { /* queue is full */
		/* even if the queue is full, if this packet is one of currently stored frame, then we can store it !!! */

		/* if the frame buffer is full, it probably is a missing frame */
		/* just truncate the oldest buffer and take the new one */
		flush_one_buffer(e, papmo);
		/* just for sanity check */
		assert( buffer_size != MAX_FRAME_BUFFER_SIZE );

//		fprintf(stderr, "Warning! receiving buffer full! You should increase your video receiving buffer\n");
	}


	/* new frame arrived */
	/* we need to add a new frame buffer */
	if( targetBuffer == NULL ) {
		targetBuffer = create_new_buffer( bphdr->frametype, bphdr->frameindex, bphdr->pkts_per_frame );
	}

	checkFrameIntegrity();
	/* now targetBuffer is the frame buffer that we should enqueue the packet */
	targetBuffer->enque( p );

	/* if something reaches threshold, signal to deque_and_send */
	return 0;
}


int FlowMpeg2AdaptDecap::deque_and_send( const Element* e, papmo* papmo )
{
	D_START_FUNCTION;

	if( buffer_size == 0 ) 
	{
		D_END_FUNCTION;
		return 0; /* queue is empty */
	}

	/* here, we use policy to determine which portion of frame is received, send up the frame to vlc 
	 * Issue 1. due to the un-ordered delivery, there might be a frame unordered delivery 
	 * Issue 2. the triggering of deque should be 
         *   (1) time limit of video playout and it's corresponding frame-received threshold
	 *   (2) what else should it be?
	 *   (3) has to implement flush-out partially missed frames
	 */
	
	/* we might need searching */
	/* searching frame tail since it is the oldest */

	if( isInitialFrameBufferReady() == true )
	{
		FrameReceivingBuffer* next;
		FrameReceivingBuffer* frameBufferReady;
		next = frameBufferTail;
		while( next && isThisFrameReady( next ) )
		{
	
			frameBufferReady = next;

			next = next->pPrev;

			disconnect( frameBufferReady );

			/* timestamp the time that sending to upper layer */
			gettimeofday( &tv_last_frame_sent, NULL );
			last_frame_index = frameBufferReady->get_frameindex();

			if( frameBufferReady->received_ratio() == 1.0 ) 
			{
	
				sendFrameToUpperLayer( frameBufferReady, e, papmo );
				

			} else {
				deleteBuffer( frameBufferReady );
			}
			
			checkFrameIntegrity();

		}
	}

	D_END_FUNCTION;
	return 0;
}
void FlowMpeg2AdaptDecap::sendFrameToUpperLayer( FrameReceivingBuffer* buffer, const Element* e, papmo* papmo )
{
	D_START_FUNCTION;

	fprintf(stderr, "sending frame = %d\t", buffer->get_frametype());

	/* send out the packets */
	Packet* p;
	WritablePacket* p_out;
	while( (p = buffer->deque()) )
	{

		if( papmo ) papmo->do_monitor(
				COMPOSED_TRACE_TAG_MPEG ,
				COMPOSED_TRACE_POS_L4_IN,
				p,
				this,
				NULL,
				NULL,
				NULL );

		p_out = decapsulate_bpadapt( p );
		e->output(0).push( p_out );
	}
	/* this drop variable is always zero... */

	delete buffer;
	buffer_size --;

	D_END_FUNCTION;
}
void FlowMpeg2AdaptDecap::deleteBuffer( FrameReceivingBuffer* buffer)
{

	buffer->reset(NULL, NULL);

	delete buffer;

	killed_buffer ++;
	buffer_size --;		
}


bool FlowMpeg2AdaptDecap::isThisFrameReady( FrameReceivingBuffer* buffer)
{

	/* start frame */
	if( last_frame_index == -1 ) return true;

	struct timeval tv;
	gettimeofday( &tv, NULL );

	int current_index = buffer->get_frameindex();

	int index_diff = current_index - last_frame_index;
	
	long timegap = timevaldiff( &tv_last_frame_sent, &tv );

	if( timegap > index_diff * (1000*1000 / MPEG2_FPS) + adaptivePlayoutFactor ) return true;

/*
	struct timeval tv;
	gettimeofday( &tv, NULL );

	long expected_time = get_expected_playout_time( buffer->get_frameindex() );

	long elapsed_time = timevaldiff( &tv_video_play_start_time, &tv );


//	printf("frameindex=%d, expected_time=%d, elapsed_time=%d\n", buffer->get_frameindex(),
//			expected_time, elapsed_time );

	if( expected_time < elapsed_time + MPEG2_PLAYOUT_TIME_BUDGET )
	{
		return true;
	}

*/
	return false;
}


WritablePacket* FlowMpeg2AdaptDecap::decapsulate_bpadapt( Packet* p )
{
        WritablePacket* p_out;
        struct bpadapt_header* bphdr;
        int sizeof_nh = sizeof(click_ether) + sizeof(click_ip) + sizeof(click_udp);
        p_out = Packet::make( sizeof_nh /* head room */
                                , NULL, /* data */
                                p->length() - sizeof(struct bpadapt_header), /* data size */
                                0); /* tailroom */

        /* targeting bphdr */
        bphdr = (struct bpadapt_header*)(p->data() + sizeof_nh);

        /* copy head */
        memcpy( p_out->data(), p->data(), sizeof_nh );

        /* copy the ts packets */
        memcpy( p_out->data() + sizeof_nh, bphdr+1, p->length() - sizeof_nh - sizeof(struct bpadapt_header));

	/* reset the IP Check sum: */
	checksumIP( p_out, sizeof(click_ether) );

        /* reset the UDP checksum: XXX do we have to?*/
        checksumUDP( p_out, sizeof(click_ether) );

        p->kill(); /* kill original packet */
        return p_out;

}

/************************************************************************
 * FlowMpeg2AdaptDecapStat Implementation 
 ************************************************************************/

void FlowMpeg2AdaptDecapStat::flush_frame_index()
{
	total_frame_index = 0;
}

void FlowMpeg2AdaptDecapStat::print_frame_index( FILE* fp, char* buf )
{
	buf[0]=NULL;
	int old=-1;
	int i;

	/* debugging messages */

	/* print drop statistics */
	if( fp ) fprintf(fp, "[%d: %d+%d] ", total_frame_drop_count,
						frame_drop_unordered,
						frame_drop_packet_loss );

	if( buf ) sprintf(buf+strlen(buf), "[%d: %d+%d] ", total_frame_drop_count,
						frame_drop_unordered,
						frame_drop_packet_loss );


	/* how many packets are unordered */
	for( i = 0; i<total_frame_index; i++ )
	{
		if( old == -1 ) {
			old = sent_frame_index[i];
		} else {
			if( sent_frame_index[i] - old != 1 )
			{
				if( fp )
					fprintf(fp, "*" );
				if( buf )
					sprintf(buf+strlen(buf), "*" );
			}
			old = sent_frame_index[i];
		}
	}
	if( fp ) fprintf(fp, "\n\t\t");
	if( buf ) sprintf( buf+strlen(buf), "\n\t\t" );

	/* print sent frame info */
	for( i = 0; i<total_frame_index; i++ )
	{
		if( fp )
		{
			if( drop_frame_indicator[i] ) fprintf(fp, "^");
			if( i > 0 ) fprintf(fp, "[%.3f]", double(timevaldiff( &sent_frame_time[i-1], &sent_frame_time[i])/1000)/1000.0 );
			fprintf(fp, "%d ", sent_frame_index[i] );
			
		}
		if( buf )
		{
			if( drop_frame_indicator[i] ) sprintf(buf+strlen(buf), "^");
			if( i > 0 ) sprintf(buf+strlen(buf), "[%.3f]", double(timevaldiff( &sent_frame_time[i-1], &sent_frame_time[i])/1000)/1000.0 );
			sprintf(buf+strlen(buf), "%d ", sent_frame_index[i] );
		}
	}
	if( fp ) fprintf( fp, "\n");
	if( buf ) sprintf( buf+strlen(buf), "\n");
}
void FlowMpeg2AdaptDecapStat::start_timestamp()
{
	gettimeofday( &start_time, NULL );
}

CLICK_ENDDECLS
ELEMENT_PROVIDES(FlowMpeg2AdaptDecap)
