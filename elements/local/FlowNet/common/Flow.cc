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
#include <clicknet/ether.h>

#include <stdarg.h>

#include "FlowCommon.hh"
#include "Flow.hh"
CLICK_DECLS
/*
 * jyyoo debugging : indenting stack depth 
 */
D_DEFINE_EXTERN;


int Flow::allocate(struct Flow** f)
{
	D_START_FUNCTION;		
	*f = (struct Flow*) malloc( sizeof(struct Flow) );
	if( *f == NULL ) 
	{
		D_END_FUNCTION;		
		return ENOBUFS;
	}
	D_END_FUNCTION;		
	return 0;
}

void Flow::free(struct Flow* f)
{
	D_START_FUNCTION;
	f->destroy();
	free( f );
	D_END_FUNCTION;		
}


void Flow::destroy()
{
	q.destroy();
}
int Flow::setup(const struct FlowID* f)
{
	D_START_FUNCTION;		
	/* TODO: reset this flow */
	memcpy( &fid, f, sizeof(fid) );
	clear();
	D_END_FUNCTION;		
	return 0;
}

int Flow::does_it_expire()
{
	age ++;
	if( age >= FLOW_MAX_AGE && queue_empty() )
	{
		clear();
		return 0;
	}
	return -1;
}

int Flow::init(int max_queue_size)
{
	D_START_FUNCTION;
	q.init( max_queue_size );
	clear();

	D_END_FUNCTION;		
	return 0;
}

int Flow::enque(Packet* p)
{
	D_START_FUNCTION;		
	int re = q.push(p);
	if( re == 0 ) {
		queue_len ++;
		total_sent ++;
		sent_now ++;
	} else {
		qdrop ++;
		qdrop_now ++;
		p->kill();
	}
	D_END_FUNCTION;		
	return re;
}
Packet* Flow::deque()
{
	D_START_FUNCTION;

	Packet* p;
	if( !queue_empty() )
	{
		queue_len --;
	}
	p = q.pop();

	D_END_FUNCTION;		
	return p;
}



void Flow::clear()
{
	D_START_FUNCTION;		
	Packet* p;
	age = 0;
	queue_len = 0;
	qdrop = 0;
	qdrop_now = 0;
	sent_now = 0;
	total_sent = 0;
	while( (p = q.pop()) )
	{
		p->kill();
	}
	D_END_FUNCTION;		
}

void Flow::toString(char* buf, int )
{
	sprintf(buf, "Information Not Supported\n");
}

CLICK_ENDDECLS
ELEMENT_PROVIDES(Flow)
