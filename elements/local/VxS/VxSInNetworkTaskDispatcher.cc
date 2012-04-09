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
#include <unistd.h>
#include "VxSInNetworkTaskDispatcher.hh"
#include "VxSInNetworkTaskQueue.hh" 

#include "VxSInNetworkCompute.hh"
#include "VxSInNetworkComputeDXTC.hh"
#include "VxSInNetworkComputeDXTD.hh"
#include "VxSInNetworkComputeFrameResize.hh"
#include "VxSInNetworkComputeYUV2_to_RGB4.hh"

#include "VxSInNetworkRawBatcher.hh" /* for debugging */

CLICK_DECLS

/**
 * a thread for working as a dispatcher 
 * 
 * FIXME: by spreading the usleep over multiple threads,
 * increase the responsiveness
 */
void *dispatcher(void *d) 
{
	VxSInNetworkTaskDispatcher *dispatcher = (VxSInNetworkTaskDispatcher *)d;
	while( dispatcher->__dispatch() ) usleep(VXS_DISPATCH_INTERVAL);

	return NULL;
}

/**
 * implementation of VxSInNetworkTaskQueue
 */
VxSInNetworkTaskDispatcher::VxSInNetworkTaskDispatcher(VxSInNetworkTaskQueue *t, VxSInNetworkTaskQueue *t2)
{
	_task_queue_incoming = t;
	_task_queue_outgoing = t2;
	_num_of_live_threads = 0;
	_on_the_go = 0;
}

VxSInNetworkTaskDispatcher::~VxSInNetworkTaskDispatcher()
{
	int status;

	_on_the_go = 0;

	/* delete all the registered computes */
	std::list<VxSInNetworkCompute *>::iterator ii;
	for( ii = _list_computes.begin(); ii != _list_computes.end(); ii ++ ) {
		VxSInNetworkCompute *c = *ii;
		delete c;
	}

	/* kick up the threads here */

	/* now, join all the threads */
	for( int i = 0; i<_num_of_live_threads; i++ ) {
		pthread_join( _thread_handles[i], (void**)&status );
	}
}

void VxSInNetworkTaskDispatcher::init_computes()
{
	/* create compute objects */
	VxSInNetworkComputeDXTC 	*cuda_dxtc = new VxSInNetworkComputeDXTC("CUDA_DXTC");
	VxSInNetworkComputeDXTD 	*cuda_dxtd = new VxSInNetworkComputeDXTD("CUDA_DXTD");
	VxSInNetworkComputeYUV2_to_RGB4 *y2r = new VxSInNetworkComputeYUV2_to_RGB4("YUV2_TO_RGB4");
	VxSInNetworkComputeFrameResize  *rs  = new VxSInNetworkComputeFrameResize("FRAME_RESIZE");

	/* register compute objects */
	_list_computes.push_back( cuda_dxtc );
	_list_computes.push_back( cuda_dxtd );
	_list_computes.push_back( y2r );
	_list_computes.push_back( rs );
}

int VxSInNetworkTaskDispatcher::startDispatching( int thread_num )
{
	if( thread_num > VXS_MAX_THREADS ) {
		click_chatter("Error: request too many threads: %d the limit is %d\n", 
			thread_num, VXS_MAX_THREADS );
		return -1;
	}

	init_computes();

	_on_the_go = 1;
	_num_of_live_threads = thread_num;

	for( int i = 0; i<_num_of_live_threads; i++ ) {
		if( pthread_create( &(_thread_handles[i]), NULL, dispatcher, (void *)this) < 0 ) {
			click_chatter("Error: while creating threads\n");
			return -1;
		}
	}	

	return 0;
}

int VxSInNetworkTaskDispatcher::run_action_on_task( VxSInNetworkTask *task, struct ofp_action_header *ah )
{
	int type = htons(ah->type);
	VxSInNetworkRawSegment *s = NULL; /* only for debugging */
	switch( type ) {

		/* FIXME: need a mapping between OFPAT_* and computes (in the list) */
		case OFPAT_VXS_DXTComp:
		{
			VxSInNetworkCompute *c = lookupCompute("CUDA_DXTC");
			if( c == NULL ) {
				click_chatter("Error: CUDA_DXTC not found\n");
			} else { /* task is done */
				int re;

				/* do we need explicit type-casting ? */
				re = ((VxSInNetworkComputeDXTC *)c)->compute( task->getSegment() );

				task->taskDone();
				task->setReturnValue( re );

				s = (VxSInNetworkRawSegment *)task->getSegment();
			}
			break;
		}

		case OFPAT_VXS_DXTDecomp:
		{
			VxSInNetworkCompute *c = lookupCompute("CUDA_DXTD");
			if( c == NULL ) {
				click_chatter("Error: CUDA_DXTD not found\n");
			} else { /* task is done */
				int re;

				/* do we need explicit type-casting ? */
				re = ((VxSInNetworkComputeDXTC *)c)->compute( task->getSegment() );

				task->taskDone();
				task->setReturnValue( re );
				s = (VxSInNetworkRawSegment *)task->getSegment();
			}
			break;
		}

		case OFPAT_VXS_FrameResize:
		{
			VxSInNetworkCompute *c = lookupCompute("FRAME_RESIZE");
			if( c == NULL ) {
				click_chatter("Error: CUDA_DXTC not found\n");
			} else { /* task is done */
				int re;

				/* do we need explicit type-casting ? */
				re = ((VxSInNetworkComputeDXTC *)c)->compute( task->getSegment() );

				task->taskDone();
				task->setReturnValue( re );
				s = (VxSInNetworkRawSegment *)task->getSegment();
			}
			break;
		}

		case OFPAT_VXS_YUV2RGB:
		{
			VxSInNetworkCompute *c = lookupCompute("YUV2_TO_RGB4");
			if( c == NULL ) {
				click_chatter("Error: CUDA_DXTC not found\n");
			} else { /* task is done */
				int re;

				/* do we need explicit type-casting ? */
				re = ((VxSInNetworkComputeDXTC *)c)->compute( task->getSegment() );

				task->taskDone();
				task->setReturnValue( re );
				s = (VxSInNetworkRawSegment *)task->getSegment();
			}
			break;
		}

		case OFPAT_OUTPUT:
		{
			sendToOutputTaskQueue(task, ah);
			break;
		}
	
		default:
			click_chatter("Unimplemented of action: %d\n", ah->type );
			break;
	}

	return 0;
}

void VxSInNetworkTaskDispatcher::sendToOutputTaskQueue( VxSInNetworkTask *task, struct ofp_action_header *ah)
{
	VxSInNetworkFlowBatcher *flow = task->getFlow();
	struct sw_flow_key *fid = flow->getFlowId();
	struct ofp_action_output *oa = (struct ofp_action_output *)ah;
	int32_t out_port = ntohs(oa->port);
	uint16_t in_port = ntohs(fid->flow.in_port);
		
	task->set_prev_in_port( in_port, out_port );

	if( task->isTaskDone() ) {
		_task_queue_outgoing->pushTask( task );
	} else {
		click_chatter("Error: Unfinished task try to do output: %p\n", task);
	}
}

VxSInNetworkCompute * VxSInNetworkTaskDispatcher::lookupCompute(const char *name)
{
	std::list<VxSInNetworkCompute *>::iterator ii;
	for( ii = _list_computes.begin(); ii != _list_computes.end(); ii ++ ) {
		VxSInNetworkCompute *c = *ii;
		if( c->isThisCompute( name ) )
			return c;
	}
	return NULL;
}

/* 
 * this function is periodically called by threads 
 * in order to specify this (thread thingy), we use
 * double underbar
 */
int VxSInNetworkTaskDispatcher::__dispatch()
{
	VxSInNetworkTask *task =  _task_queue_incoming->popTask();

	if( task ) {
		
		struct ofp_action_header *ah;
		while( (ah = task->getNextActionHeader()) != NULL ) {
			run_action_on_task( task, ah );
		}
	}

	return _on_the_go;
}

CLICK_ENDDECLS
ELEMENT_PROVIDES(VxSInNetworkTaskDispatcher)
