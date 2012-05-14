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
#include "../OpenFlow/lib/packets.hh"

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
	sem_init(&_sem_GPU, 0, 1);
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

	sem_destroy(&_sem_GPU);
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
		case OFPAT_VXS_COPY_BRANCH:
		{
			struct ofp_action_vxs_copy_branch *vcb = (struct ofp_action_vxs_copy_branch *)ah;
			uint32_t iterations=ntohl(vcb->num_copy);
			uint32_t steps1 = ntohl(vcb->branch_pos1);
			uint32_t steps2 = ntohl(vcb->branch_pos2);
			uint32_t i;
			if( iterations >= 1 ) {
				VxSInNetworkTask *copied_task = task->clone();
			
				for( i = 0; i<steps1; i++ ) {
					/* just remove the actions */
					if( copied_task->getNextActionHeader() == NULL ) {
						click_chatter("OOPS: ACTION HEADER GETS NULL!?\n");
						break;
					}

				}
				_task_queue_incoming->pushTask( copied_task );
			}
			if( iterations >= 2 ) {
				VxSInNetworkTask *copied_task = task->clone();
			
				for( i = 0; i<steps2; i++ ) {
					/* just remove the actions */
					if( copied_task->getNextActionHeader() == NULL ) {
						click_chatter("OOPS: ACTION HEADER GETS NULL!?\n");
						break;
					}
				}
				_task_queue_incoming->pushTask( copied_task );
			}
			break;
		}

		case OFPAT_VXS_YUV2RGB_DXTC:
		{
			VxSInNetworkCompute *c = lookupCompute("CUDA_DXTC");
			if( c == NULL ) {
				click_chatter("Error: CUDA_DXTC not found\n");
			} else { /* task is done */
				int re;

				sem_wait( &_sem_GPU );

				/* TODO: make this input mode thing as parametric form of OFPAT_VXS_DXTComp */
				((VxSInNetworkComputeDXTC *)c)->set_input_mode( 1 ); /* rgb4 */

				/* do we need explicit type-casting ? */
				re = ((VxSInNetworkComputeDXTC *)c)->compute( task->getSegment() );

				sem_post( &_sem_GPU );

				task->taskDone();
				task->setReturnValue( re );

				s = (VxSInNetworkRawSegment *)task->getSegment();
			}
			break;
		}

		case OFPAT_VXS_DXTComp:
		{
			VxSInNetworkCompute *c = lookupCompute("CUDA_DXTC");
			if( c == NULL ) {
				click_chatter("Error: CUDA_DXTC not found\n");
			} else { /* task is done */
				int re;

				sem_wait( &_sem_GPU );

				((VxSInNetworkComputeDXTC *)c)->set_input_mode( 0 ); /* rgb4 */

				/* do we need explicit type-casting ? */
				re = ((VxSInNetworkComputeDXTC *)c)->compute( task->getSegment() );

				sem_post( &_sem_GPU );

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

				sem_wait( &_sem_GPU );

				/* do we need explicit type-casting ? */
				re = ((VxSInNetworkComputeDXTC *)c)->compute( task->getSegment() );

				sem_post( &_sem_GPU );

				task->taskDone();
				task->setReturnValue( re );
				s = (VxSInNetworkRawSegment *)task->getSegment();
			}
			break;
		}

		case OFPAT_VXS_FrameResize:
		{
			/* TODO: frame resize should be paramatized */
			VxSInNetworkCompute *c = lookupCompute("FRAME_RESIZE");
			if( c == NULL ) {
				click_chatter("Error: CUDA_DXTC not found\n");
			} else { /* task is done */
				int re;

				sem_wait( &_sem_GPU );

				/* do we need explicit type-casting ? */
				re = ((VxSInNetworkComputeDXTC *)c)->compute( task->getSegment() );

				sem_post( &_sem_GPU );

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

				sem_wait( &_sem_GPU );

				/* do we need explicit type-casting ? */
				re = ((VxSInNetworkComputeDXTC *)c)->compute( task->getSegment() );

				sem_post( &_sem_GPU );

				task->taskDone();
				task->setReturnValue( re );
				s = (VxSInNetworkRawSegment *)task->getSegment();
			}
			break;
		}

		case OFPAT_OUTPUT:
		{
			sendToOutputTaskQueue(task, ah);
			return 1;
		}

		case OFPAT_SET_DL_DST:
		{
			struct ofp_action_dl_addr *da = (struct ofp_action_dl_addr *)ah;
			struct eth_header *eh = (struct eth_header*) task->getNetworkHeaders();
			memcpy(eh->eth_dst, da->dl_addr, sizeof eh->eth_dst);
			break;
		}

		case OFPAT_SET_NW_DST:
		{
			struct ofp_action_nw_addr *da = (struct ofp_action_nw_addr *)ah;
			struct eth_header *eh = (struct eth_header*) task->getNetworkHeaders();
			struct ip_header *nh = (struct ip_header*) (eh+1);

			if( ntohs(eh->eth_type) == ETH_TYPE_IP ) {
				nh->ip_dst = da->nw_addr;
			}
			break;
		}

		case OFPAT_SET_TP_DST:
		{
			struct ofp_action_tp_port *ta = (struct ofp_action_tp_port *)ah;
			struct eth_header *eh = (struct eth_header*) task->getNetworkHeaders();
			struct ip_header *nh = (struct ip_header*) (eh+1);
			/**
			 * NOTE: tcp also has the same structure upto 4 byte 
			 * and we only use upto 4 byte here
			 */
            		struct udp_header *th = (struct udp_header*) (nh+1); 

			if( ntohs(eh->eth_type) == ETH_TYPE_IP ) {
				th->udp_dst = ta->tp_port;
			}
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
			if( run_action_on_task( task, ah ) ) break;
		}
	}

	return _on_the_go;
}

CLICK_ENDDECLS
ELEMENT_PROVIDES(VxSInNetworkTaskDispatcher)
