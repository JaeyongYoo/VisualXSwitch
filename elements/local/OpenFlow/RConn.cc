// -*- c-basic-offset: 4 -*-
/*
 * 
 * Jae-Yong Yoo
 *
 * Copyright (c) 2011 Gwangju Institute of Science and Technology
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
#include <click/glue.hh>
#include <click/error.hh>
#include <click/confparse.hh>
#include <click/straccum.hh>
#include <click/standard/alignmentinfo.hh>

#include "include/config.h"
#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include "include/openflow/openflow.hh"
#include "lib/ofpbuf.hh"
#include "lib/timeval.hh"
#include "lib/vconn.hh"
#include "lib/vconn-provider.hh"
#include "lib/poll-loop.hh"
#include "lib/ofp-print.hh"
#include "RConn.hh"
#include "datapath.hh"


CLICK_DECLS

/* jyyoo TODO: consider the properness of this semaphore */
extern sem_t sem_waiters;

void handle_remote(class RConn *rconnElement);
void dp_wait(struct RConn *rc);

/**************************************************************
 * a thread that communicates to nox
 * this blocks until there are some messages from nox 
 * or passive information reader (such as dpctl)
 **************************************************************/
void* thread_nox_commun(void* param)
{
        struct rnc_param* rnc = (struct rnc_param*) param;
        RConn* rc = rnc->rconnElement;

        while( rnc->liveness )
        {
		sem_wait( &rc->mutex_rconn );
                handle_remote(rc);
                dp_wait( rc );
		sem_post( &rc->mutex_rconn );
                poll_block();

        }
        return NULL;
}

/**************************************************************************
 * Inner functions to RConn element 
 *************************************************************************/
static void remote_wait(struct rconn_remote *r)
{
        rconn_run_wait(r->rconn);
        rconn_recv_wait(r->rconn);
}
static void remote_destroy(struct rconn_remote *r)
{
        if (r) {
                if (r->cb_dump && r->cb_done) {
                        r->cb_done(r->cb_aux);
                }
                list_remove(&r->node);
                rconn_destroy(r->rconn);
                free(r);
        }
}

static struct rconn_remote * remote_create(class RConn *rconnElement, struct rconn *rconn)
{
        struct rconn_remote *remote = (struct rconn_remote*)xmalloc(sizeof *remote);
        list_push_back(&rconnElement->remotes, &remote->node);
        remote->rconn = rconn;
        remote->cb_dump = NULL;
        remote->n_txq = 0;

        return remote;
}



static void remote_run(class RConn *rconnElement, struct rconn_remote *r)
{
	DEBUG_TRACE_FUNCTION_CALL;
	int i;

        rconn_run(r->rconn);

        /* Do some remote processing, but cap it at a reasonable amount so that
         * other processing doesn't starve. */
	for (i = 0; i < 50; i++) {
		struct ofpbuf *buffer;
		struct ofp_header *oh;

		buffer = rconn_recv(r->rconn);
		if (!buffer) {
			break;
		}

		if (buffer->size >= sizeof *oh) {

			oh = (struct ofp_header *)buffer->data;

			rconnElement->parse_control_input( r, oh->xid, buffer->data, buffer->size);

		} else {
			click_chatter("Error: receive too short Datapath message\n");
		}
		ofpbuf_delete(buffer);
	}

        if (!rconn_is_alive(r->rconn)) {
                remote_destroy(r);
        }
        DEBUG_TRACE_FUNCTION_END
}

/*********************************************************************
 * basic communication function to nox 
 ********************************************************************/
static void *make_openflow_reply(size_t openflow_len, uint8_t type,
                struct rconn_remote *sender, uint32_t xid, struct ofpbuf **bufferp)
{
        return make_openflow_xid(openflow_len, type, sender ? xid : 0,
                        bufferp);
}

static int send_openflow_buffer_to_remote(struct ofpbuf *buffer, struct rconn_remote *remote)
{

	DEBUG_TRACE_FUNCTION_CALL;

        int retval = rconn_send_with_limit(remote->rconn, buffer, &remote->n_txq,
                        TXQ_LIMIT);
        if (retval) {
		click_chatter("Error: send to \"%s\" failed: %s %s\n",
                                rconn_get_name(remote->rconn), 
				strerror(retval),
				retval == EAGAIN ? "(too much control messages)" : ""  );
        } else {
	}
        return retval;
}


static int send_openflow_buffer(class RConn *rconnElement, struct ofpbuf *buffer, struct rconn_remote *sender, uint32_t xid UNUSED)
{

	DEBUG_TRACE_FUNCTION_CALL;
	update_openflow_length(buffer);

	/* jyyoo debug */
#if JD_DEBUG_PARSE_CONTROL_PACKET
	click_chatter("===========> control out <================\n");
	click_chatter("%s\n", ofp_to_string( buffer->data, buffer->size, 1 ) );
#endif

	if (sender) {
		return send_openflow_buffer_to_remote(buffer, sender);
	} else {
		/* Broadcast to all remotes. */
		struct rconn_remote *r, *prev = NULL;
		LIST_FOR_EACH (r, struct rconn_remote, node, &rconnElement->remotes) {
			if (prev) {
				send_openflow_buffer_to_remote(ofpbuf_clone(buffer), prev);
			}
			prev = r;
		}
		if (prev) {
			send_openflow_buffer_to_remote(buffer, prev);
		} else {
			ofpbuf_delete(buffer);
		}
		return 0;
	}
}

static void rconn_send_error_msg(class RConn* rconnElement, struct rconn_remote *sender, uint32_t xid,
		uint16_t type, uint16_t code, const void *data, size_t len)
{
	DEBUG_TRACE_FUNCTION_CALL;
	struct ofpbuf *buffer;
	struct ofp_error_msg *oem;


	oem = (struct ofp_error_msg*)make_openflow_reply(sizeof(*oem)+len, OFPT_ERROR, sender, xid, &buffer);
	oem->type = htons(type);
	oem->code = htons(code);
	memcpy(oem->data, data, len);
	send_openflow_buffer(rconnElement, buffer, sender, xid);
}
/*********************************************************************
 * RConn replay functions
 ********************************************************************/
static struct ofpbuf * make_barrier_reply(const struct ofp_header *req)
{
	size_t size = ntohs(req->length);
	struct ofpbuf *buf = (struct ofpbuf*) ofpbuf_new(size);
	struct ofp_header *reply = (struct ofp_header*)ofpbuf_put(buf, req, size);

	reply->type = OFPT_BARRIER_REPLY;
	return buf;
}
static int recv_barrier_request(class RConn *rconnElement, struct rconn_remote *sender, uint32_t xid, const void *ofph)
{
	return send_openflow_buffer(rconnElement, make_barrier_reply((struct ofp_header*)ofph), sender, xid);
}

static int recv_echo_request(class RConn *rconnElement, struct rconn_remote *sender, uint32_t xid, const void *oh)
{
        return send_openflow_buffer(rconnElement, make_echo_reply((const struct ofp_header*)oh), sender, xid);
}

static int recv_echo_reply(class RConn *rconn UNUSED, struct rconn_remote *sender UNUSED, uint32_t xid UNUSED,
                const void *oh UNUSED)
{
        return 0;
}




/**************************************************************************
 * Interface functions to RConn element 
 *************************************************************************/
void dp_wait(struct RConn* rconn)
{
	DEBUG_TRACE_FUNCTION_CALL;
        struct rconn_remote *r;
        size_t i;

        LIST_FOR_EACH (r, struct rconn_remote, node, &rconn->remotes) {
                remote_wait(r);
        }
        for (i = 0; i < rconn->n_listeners; i++) {
                pvconn_wait(rconn->listeners[i]);
        }
        DEBUG_TRACE_FUNCTION_END
}

void handle_remote(class RConn *rconnElement)
{
	DEBUG_TRACE_FUNCTION_CALL;

        struct rconn_remote *r, *rn;
        LIST_FOR_EACH_SAFE (r, rn, struct rconn_remote, node, &rconnElement->remotes) {
                remote_run(rconnElement, r);
        }

        for (uint16_t i = 0; i < rconnElement->n_listeners; ) {
                struct pvconn *pvconn = rconnElement->listeners[i];
                struct vconn *new_vconn;

                int retval = pvconn_accept(pvconn, OFP_VERSION, &new_vconn);

                if (!retval) 
		{
                        remote_create(rconnElement, rconn_new_from_vconn("passive", new_vconn));

                } else if (retval != EAGAIN) {
                        printf("Error! accept failed (%s)", strerror(retval));
                        rconnElement->listeners[i] = rconnElement->listeners[--rconnElement->n_listeners];
                        continue;
                }
                i++;
        }

        DEBUG_TRACE_FUNCTION_END
}



/*********************************************************************
 * click RConn element packaging 
 ********************************************************************/
RConn::RConn() : send_timer(this), report_timer(this)
{
	packetQueue.init( 1000 );
	strcpy( _pvconn_name, "punix:/var/run/dp0.sock");
}

RConn::~RConn()
{
	_datapath = NULL;
	packetQueue.destroy();
}

void RConn::cleanup(CleanupStage stage UNUSED)
{
	sem_destroy(&sem_waiters); /* destroy semaphore */
	sem_destroy(&mutex_rconn); /* destroy semaphore */
}


int RConn::initialize(ErrorHandler*)
{
	send_timer.initialize(this);
        if( RCONN_SEND_TIMER_CLOCK ) send_timer.schedule_after_msec(RCONN_SEND_TIMER_CLOCK);
	report_timer.initialize(this);
        if( RCONN_REPORT_TIMER_CLOCK ) report_timer.schedule_after_msec(RCONN_REPORT_TIMER_CLOCK);

        sem_init(&sem_waiters, 0, 1);      /* initialize mutex to 1 - binary semaphore */
	sem_init(&mutex_rconn, 0, 1);

	memset( &rstat, 0, sizeof(rstat) );

	return 0;
}

int RConn::configure(Vector<String> &, ErrorHandler *)
{
	DEBUG_TRACE_FUNCTION_CALL;

	int retval;

	time_init();


	list_init(&remotes);
	listeners = NULL;
	n_listeners = 0;

	_pvconn=NULL;

	retval = pvconn_open(_pvconn_name, &_pvconn);

	Timestamp now = Timestamp::now();
	click_chatter("%{timestamp} %{element}: pvconn_open [name=%s : %x]\n", &now, this, _pvconn_name, _pvconn);

	if (!retval || retval == EAGAIN) {

		add_pvconn( _pvconn );
	} else {
		Timestamp now = Timestamp::now();
		click_chatter("%{timestamp} %{element}: Error! opening %s\n", &now, this, _pvconn_name);

		return -1;
	}

	if (!n_listeners) {
		Timestamp now = Timestamp::now();
		click_chatter("%{timestamp} %{element}: Error! could not listen for any connections\n", &now, this);

		return -1;
	}

	create_thread_nox_commun();

	return 0;
}

void RConn::run_timer(Timer* t)
{
        if( t == &send_timer) { /* send the control packets */

		send_timer.schedule_after_msec(RCONN_SEND_TIMER_CLOCK);

		send_control_packets_to_click();
	} else if( t == &report_timer) { /* report status timer */ 
		report_timer.schedule_after_msec(RCONN_REPORT_TIMER_CLOCK);

		report_stat();
	}
}

void RConn::add_handlers()
{
}

/* TODO: this push is deprecated, we use direct communication to datapath using _datapath */
void RConn::push(int input UNUSED, Packet *p)
{
	DEBUG_TRACE_FUNCTION_CALL;
	p->kill();
}

void RConn::send_openflow_buffer(struct ofpbuf *buffer, struct rconn_remote *rconn_sender, uint32_t xid)
{
	sem_wait( &mutex_rconn );

	/* update rconn state */
	rstat.stat_total_control_out ++;
	rstat.stat_control_out ++;

	::send_openflow_buffer( this, buffer, rconn_sender, xid );

	sem_post( &mutex_rconn );
}

static int RConn_report_stat_counter = 0;
void RConn::report_stat()
{
	RConn_report_stat_counter ++;
	
	click_chatter("[%c] Remote Connection Status\n",
		RConn_report_stat_counter % 4 == 0 ? '-' :
		RConn_report_stat_counter % 4 == 2 ? '/' :
		RConn_report_stat_counter % 4 == 3 ? '|' : '\\' );

	click_chatter("\tControl In\t\tTotal Control In\tControl Out\t\tTotal Control Out\n");
	click_chatter("\t\t%d\t\t%d\t\t\t%d\t\t\t%d\n", 
				rstat.stat_control_in, 
				rstat.stat_total_control_in, 
				rstat.stat_control_out, 
				rstat.stat_total_control_out );
	
	click_chatter("\tQueue Len\t\tQueue Len Avg\t\tQueue Len Dev\t\tQueue Drop\t\tTotal Queue Drop\n" );
	click_chatter("\t\t%d\t\t%f\t\t%f\t\t%d\t\t\t%d\n", 
				rstat.stat_queue_len,
				rstat.stat_ewma_avg_queue_len,
				rstat.stat_ewma_dev_queue_len,
				rstat.stat_queue_drop,
				rstat.stat_total_queue_drop );

	click_chatter("\tTask in-queue len\tTask out-queue len\n" );
	click_chatter("\t\t%d\t\t%d\n", 
				_datapath->taskQueueIncoming.size(),
				_datapath->taskQueueOutgoing.size() );

	rstat.stat_control_in = 0;
	rstat.stat_control_out = 0;
	rstat.stat_queue_drop = 0;

}

void RConn::add_pvconn(struct pvconn *pvconn)
{
	listeners = (struct pvconn **)xrealloc(listeners,
			sizeof *listeners * (n_listeners + 1));
	listeners[n_listeners++] = pvconn;
}

/*********************************************************************************
 * jyyoo XXX NOTE: 
 * this function is called from native rconn
 * caution should be made while handling this function 
 * since this function does not come from Click thread context
 * it is another thread that is created by RConn element 
 ********************************************************************************/
int RConn::parse_control_input(struct rconn_remote *sender, uint32_t xid, const void *msg_in, size_t length)
{
	DEBUG_TRACE_FUNCTION_CALL;

	int (*handler)(class RConn *, struct rconn_remote *, uint32_t xid, const void *) = NULL;
	struct ofp_header *oh;
	size_t min_size;

	int ofpe_type = OFPE_RCONN;

	/* FIXME: RConn should maintain the map from which click elements are under which type of control
 	 * this information should be configured by RConn configure function
	 * and according to this information parse_control_input should forward the control message 
	 * to the proper output ports */

	/* FIXME: copy the message to avoid double free at remote_run */
	void* msg = malloc( length );
	memcpy( msg, msg_in, length );
	
	/* Check encapsulated length. */
	oh = (struct ofp_header *) msg;
	if (ntohs(oh->length) > length) {
		return -EINVAL;
	}
	assert(oh->version == OFP_VERSION);

	/* Figure out how to handle it. */
	switch (oh->type) {

		/* to RConn itself */
		case OFPT_BARRIER_REQUEST:
			min_size = sizeof(struct ofp_header);
			handler = recv_barrier_request;
			ofpe_type = OFPE_RCONN;
			break;
		case OFPT_ECHO_REQUEST:
			min_size = sizeof(struct ofp_header);
			handler = recv_echo_request;
			ofpe_type = OFPE_RCONN;
			break;
		case OFPT_ECHO_REPLY:
			min_size = sizeof(struct ofp_header);
			handler = recv_echo_reply;
			ofpe_type = OFPE_RCONN;
			break;
		/* to datapath element */


                case OFPT_FEATURES_REQUEST:
                        min_size = sizeof(struct ofp_header);
			ofpe_type = OFPE_DATAPATH;
			break;
                case OFPT_GET_CONFIG_REQUEST:
			min_size = sizeof(struct ofp_header);
			ofpe_type = OFPE_DATAPATH;
			break;
		case OFPT_SET_CONFIG:
			min_size = sizeof(struct ofp_switch_config);
			ofpe_type = OFPE_DATAPATH;
			break;
		case OFPT_PACKET_OUT:
			min_size = sizeof(struct ofp_packet_out);
			ofpe_type = OFPE_DATAPATH;
			break;
		case OFPT_FLOW_MOD:
			min_size = sizeof(struct ofp_flow_mod);
			ofpe_type = OFPE_DATAPATH;
			break;
		case OFPT_PORT_MOD:
			min_size = sizeof(struct ofp_port_mod);
			ofpe_type = OFPE_DATAPATH;
			break;
		case OFPT_STATS_REQUEST:
			min_size = sizeof(struct ofp_stats_request);
			ofpe_type = OFPE_DATAPATH;
			break;
		case OFPT_QUEUE_GET_CONFIG_REQUEST:
			min_size = sizeof(struct ofp_header);
			ofpe_type = OFPE_DATAPATH;
			break;
		case OFPT_VENDOR:
			min_size = sizeof(struct ofp_vendor_header);
			ofpe_type = OFPE_DATAPATH;
			break;

		/* to shaper element */
	        case OFPT_RATE_SHAPE:
                        min_size = sizeof(struct netopen_rate_shape);
			ofpe_type = OFPE_BWSHAPER;
			break;
            break;
		default:
			fprintf(stderr, "Unrecognized type:%d\n", oh->type );

			rconn_send_error_msg(this, sender, xid, OFPET_BAD_REQUEST, OFPBRC_BAD_TYPE,
					msg, length);
			return -EINVAL;
	}

	if (length < min_size)
		return -EFAULT;

	if( handler ) {
		/* Handle it. */
		return handler(this, sender, xid, msg);
	} else {
		buffer_control_packet( ofpe_type, msg, length, sender, xid );
		return 0;
	}
}

void RConn::buffer_control_packet( int ofpe_type UNUSED, void *msg, int length, struct rconn_remote *sender, uint32_t xid )
{
	DEBUG_TRACE_FUNCTION_CALL;
	/* jyyoo NOTE: how to convert this control packet into click packet has the following issues
	 * 	1) how to minimize the memory copy
	 *	2) how can the other parts (datapath, or shaper) can compatibly interpret the packet 
	 *
	 *	Now, we just encapsulate this as null packet for the simplicity of implementation.
	 * 	I believe that click does not check header types while passing between elements.
	 */

	struct buf_rconn buf_rconn;
	buf_rconn.msg = msg;
	buf_rconn.length = length;
	buf_rconn.sender = sender;
	buf_rconn.xid = xid;

	rstat.stat_total_control_in ++;
	rstat.stat_control_in ++;


	/* put this packet into shared buffer between click thread */
	if( packetQueue.push( &buf_rconn ) == -1 ) /* queue is full */
	{
		rstat.stat_queue_drop ++;
		rstat.stat_total_queue_drop ++;

		/* jyyoo TODO: formalize this error message */
		fprintf( stderr, "Error! RConn packet queue is full [Formalize me]\n");
	} 
	rstat.update_queue_len( packetQueue.length() );

}

void RConn::send_control_packets_to_click()
{
	struct buf_rconn br;
	while (true) 
	{
		br = packetQueue.pop();
		if( br.msg == NULL ) break;

		if( _datapath ) {
			_datapath->fwd_control_input( br.msg, br.length, br.sender, br.xid );
		}
		rstat.update_queue_len( packetQueue.length() );
	}
}

void RConn::create_thread_nox_commun()
{
        rnc.liveness = true;
        rnc.rconnElement = this;
        pthread_create( &t_nox_commun, NULL, thread_nox_commun, &rnc );

}

void rconn_stat::update_queue_len(uint32_t qlen)
{
	stat_queue_len = qlen;

	stat_ewma_avg_queue_len = stat_ewma_avg_queue_len * 0.95 + 0.05 * (double) qlen;
	stat_ewma_dev_queue_len = stat_ewma_avg_queue_len * 0.95 + 0.05 * (double) qlen;
}


CLICK_ENDDECLS
EXPORT_ELEMENT(RConn)
