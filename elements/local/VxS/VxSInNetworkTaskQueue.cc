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
#include "VxSInNetworkTaskQueue.hh" 
#include "../OpenFlow/lib/ofpbuf.hh"

CLICK_DECLS

/**
 * implementation of VxSInNetworkTaskQueue
 */
void VxSInNetworkTask::set( VxSInNetworkSegment *s, VxSInNetworkFlowBatcher *f, struct ofpbuf *ob, int len )
{
	segment = s;
	flow = f;

	_task_done = false;
	_return_value = 0;

	memcpy( network_headers, ob->data, len );
	network_header_len = len;

}
struct ofp_action_header * VxSInNetworkTask::getNextActionHeader() 
{
	if( segment == NULL ) return NULL;

	return (struct ofp_action_header *)segment->getNextActionHeader();
}

void VxSInNetworkTask::print_to_chatter()
{
	click_chatter("TaskInfo: [%p]\n", this );
	if( segment ) {
		segment->print_to_chatter();
	}
	click_chatter("\n");
}

/**
 * implementation of VxSInNetworkTaskQueue
 */
VxSInNetworkTaskQueue::VxSInNetworkTaskQueue()
{
	sem_init(&_sem_tasks, 0, 1);
}

VxSInNetworkTaskQueue::~VxSInNetworkTaskQueue()
{
	sem_destroy(&_sem_tasks);
}

int VxSInNetworkTaskQueue::pushTask( VxSInNetworkTask *task )
{
	sem_wait( &_sem_tasks );
	_tasks.push_back( task );
	sem_post( &_sem_tasks );
	return 0;
}

VxSInNetworkTask * VxSInNetworkTaskQueue::popTask()
{
	VxSInNetworkTask *t;
	sem_wait( &_sem_tasks );
	if( _tasks.size() == 0 ) t = NULL; 
	else {
		t = _tasks.front();
		_tasks.pop_front();
	}
	sem_post( &_sem_tasks );
	return t;
}

CLICK_ENDDECLS
ELEMENT_PROVIDES(VxSInNetworkTaskQueue)
