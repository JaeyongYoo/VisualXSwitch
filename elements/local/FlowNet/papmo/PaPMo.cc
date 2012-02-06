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
#include <click/confparse.hh>
#include <click/error.hh>

#include <pthread.h>
#include <arpa/inet.h>
#include <unistd.h>

#include "../common/FlowCommon.hh"


#include "../scheduler/FlowSchedulable.hh" 
#include "../scheduler/CD_algorithms/CD_CORE.hh"
#include "../mpeg2/FlowMpeg2AdaptEncap.hh"
#include "PaPMo.hh"

CLICK_DECLS

/*
 * jyyoo debugging : indenting stack depth 
 */
D_DEFINE_EXTERN;

papmo::papmo()
{
	_thread_liveness=0;
}
papmo::~papmo()
{
}


/*
 * lock-free circular buffer implementation
 */

int lfc_buffer::create_buffer( int bs, int is )
{
	_buffer_size = bs;
	_item_size = is;
	_item = (void**)malloc(sizeof(void*) * bs);
	if( _item == NULL ) return -1;

	for( int i=0; i<bs; i++ )
	{
		_item[i] = (void*)malloc( is );
		if( _item[i] == NULL ) return -1;
	}
	_head = _tail = 0;
	return 0;
}

bool lfc_buffer::is_full()
{
	int tail_next = _tail + 1;
	if( tail_next == _buffer_size ) tail_next = 0;
	return tail_next == _head;
}
bool lfc_buffer::is_empty()
{
	return _head == _tail;
}
uint32_t lfc_buffer::size()
{
	int t = _tail;
	if( t < _head ) t += _buffer_size;
	return t - _head;
}
int lfc_buffer::insert( void* i )
{
	if( is_full() ) return -1;
	memcpy( _item[_tail], i, _item_size);
	_tail = _tail + 1;
	if( _tail == _buffer_size ) _tail = 0;
	return 0;
}

int lfc_buffer::pop(void* i)
{
	if( is_empty() ) return -1;
	memcpy(i, _item[_head], _item_size);
	_head = _head + 1;
	if( _head == _buffer_size ) _head = 0;
	return 0;
}

void* thread_papmo_send(void* arg) {
	struct papmo* papmo = (struct papmo*) arg;
	struct lfc_buffer* buffer = &(papmo->_buffer);

	while( papmo->_thread_liveness) 
	{
		if( buffer->size() > MAX_COMPOSEDTRACES_PER_PACKET ) {

			

			/* send out the composed trace packets until the buffer is empty */
			while( buffer->is_empty() == false )
			{
				uint8_t buf[1500];
				struct composed_trace *ct;
				uint32_t len=0;

				ct = (struct composed_trace*)buf;

				while( buffer->is_empty() == false && len < 1500 - sizeof(struct composed_trace) ) {
					buffer->pop( ct );
					ct = ct + 1;
					len += sizeof(struct composed_trace);
				}

				papmo->send_to_server( buf, len );


			}
			
		}
		
		usleep(100);
	}
	return NULL;
}

int papmo::init(int papmo_bs, IPAddress ipaddr )
{
	/* create buffer */
	_buffer.create_buffer( papmo_bs, sizeof(struct composed_trace) );

	/* connection create */

	/* make a udp socket for sending the monitored data */

	_monServerIP = ipaddr;

	_sockMonServer = socket(AF_INET, SOCK_DGRAM, 0);

	struct sockaddr_in sin;

	sin.sin_family = AF_INET;
	sin.sin_port = htons(9999);
	sin.sin_addr.s_addr = INADDR_ANY;

	if( bind(_sockMonServer, (struct sockaddr *)&sin, sizeof(sin)))
	{
		perror("bind()");
		return 1;
	}


	_sout.sin_family = AF_INET;
	_sout.sin_port = htons(SERVER_MONITOR_PORT);
	_sout.sin_addr = _monServerIP.in_addr();


	/* thread creation */
	_thread_liveness = 1;

	struct papmo_thread_arg pta;
	pta.p_thread_liveness = &_thread_liveness;
	pta.p_buffer = &_buffer;
	
	pthread_create( &_thread_send, NULL, thread_papmo_send, this );
	return 0;
}
int papmo::do_monitor(Packet* p, uint32_t tag, uint32_t qlen, uint32_t qlen_next)
{
	if( _thread_liveness == 0 ) return 0;

	struct timeval tv;
	struct composed_trace ct;

	if( p->length() < PAPMO_CAPTURE_HEAD_SIZE ) return -1;

	gettimeofday( &tv, NULL );
	ct.sec = tv.tv_sec;
	ct.usec = tv.tv_usec;
        ct.tag = tag;
        ct.qlen_self = qlen;
        ct.qlen_next = qlen_next;
	memcpy( ct.header, p->data(), PAPMO_CAPTURE_HEAD_SIZE );

	return this->do_monitor( &ct );

}
static int print_error = 0;
int papmo::do_monitor(struct composed_trace* ct)
{
	D_START_FUNCTION;
	if( _thread_liveness == 0 ) return 0;

	if( _buffer.insert( ct ) != 0 ) {
		if( print_error == 0 ) {
			printf("Error! papmo monitoring buffer overflow\n");
		}
		print_error = 1;
		D_END_FUNCTION;
		return -1;
	}
	D_END_FUNCTION;
	return 0;
}


int papmo::do_monitor( int tag, int pos, const Packet* p, const Flow* f, const VcSchedule* , const VcBWShape* , const VcCongestionDetection* cd )
{
	D_START_FUNCTION;

	if( p == NULL ) {
		printf("Error! papmo @ click receives NULL packet\n");
	}

	assert( p != NULL );

	if( _buffer.is_full() ) {
		if( print_error == 0 ) 
			printf("Error! papmo monitoring buffer overflow\n");

		return -1;
	}

	/* TODO: make it as callback handler */
	if( p->length() >= PAPMO_CAPTURE_HEAD_SIZE ) {

		struct composed_trace ct;
		memset( &ct, 0, sizeof(ct));

		struct timeval tv;


		gettimeofday( &tv, NULL );
		ct.sec = tv.tv_sec;
		ct.usec = tv.tv_usec;
		ct.tag = tag;
		ct.pos = pos;

		if( f && tag & COMPOSED_TRACE_TAG_FLOW )
		{
			const FlowSchedulable* flow = (const FlowSchedulable*) f;
			/* flow */
			ct.qdrop = flow->qdrop;
			ct.qlen_self = flow->queue_length();
			ct.qlen_next = flow->si.ni.queuelen;
		}
		if( cd && tag & COMPOSED_TRACE_TAG_CORE )
		{
			/* 
			 * TODO: Make this as the function of Algorithm
			 * CORE */
			if( strncmp( "CORE", cd->name(), 10 ) == 0 )
			{
				const VcCDCORE* vcCDCORE = (const VcCDCORE*) cd;
				ct.tag = ct.tag | COMPOSED_TRACE_TAG_CORE;
				ct.core = vcCDCORE->core_value(f);
				ct.slope = vcCDCORE->slope_value(f);
			}
		}
		if( tag & COMPOSED_TRACE_TAG_MPEG )
		{
			const struct bpadapt_header* bphdr = FlowMpeg2AdaptEncap::get_bpadapt_header_readonly( p );
			
			ct.frameindex = bphdr->frameindex;
			ct.frametype = bphdr->frametype;
			ct.pkts_per_frame = bphdr->pkts_per_frame;
			ct.pkts_index = bphdr->pkts_index;
			
		}

		memcpy( ct.header, p->data(), PAPMO_CAPTURE_HEAD_SIZE );

		papmo::do_monitor( &ct );
	}

	D_END_FUNCTION;
	return 0;
}

int papmo::send_to_server( uint8_t* buf, uint32_t len )
{
	/* send the composed trace packet to central monitoring node */
	int ret;
	ret = sendto( _sockMonServer, buf, len, 0, (struct sockaddr *)&_sout, sizeof(_sout) );
	if( ret < 0 )
	{
		perror("sendto() at papmo");
		return -1;
	}
	return 0;
}



CLICK_ENDDECLS
ELEMENT_PROVIDES(papmo)
