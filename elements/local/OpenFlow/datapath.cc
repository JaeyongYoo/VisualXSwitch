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
#include <click/confparse.hh>
#include <click/error.hh>
#include <click/straccum.hh>
#include <click/router.hh>
#include <click/routervisitor.hh>
#include <linux/if_ether.h>
#include <linux/if_packet.h>
#include <linux/sockios.h>
#include <sys/ioctl.h>
#include <arpa/inet.h>
#include <assert.h>
#include <errno.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../../userlevel/fromdevice.hh"
#include "datapath.hh"
#include "include/config.h"
#include "include/openflow/openflow.hh"
#include "lib/poll-loop.hh"
#include "lib/queue.hh"
#include "lib/util.hh"
#include "lib/rconn.hh"
#include "lib/timeval.hh"
#include "lib/vconn.hh"
#include "lib/vconn-ssl.hh"

#include "lib/chain.hh"
#include "lib/csum.hh"
#include "lib/flow.hh"
#include "lib/ofpbuf.hh"
#include "include/openflow/openflow.hh"
#include "include/openflow/nicira-ext.hh"
#include "include/openflow/private-ext.hh"
#include "include/openflow/openflow-ext.hh"
#include "lib/stp.hh"
#include "lib/switch-flow.hh"
#include "lib/table.hh"
#include "lib/vconn.hh"
#include "lib/xtoxll.hh"
#include "lib/private-msg.hh"
#include "lib/of_ext_msg.hh"
#include "lib/dp_act.hh"
#include "lib/ofp-print.hh"

#include "RConn.hh"


//#include "lib/vlog-socket.hh"



#ifdef CLICK_LINUXMODULE
# include <click/cxxprotect.h>
CLICK_CXX_PROTECT
# include <linux/sched.h>
CLICK_CXX_UNPROTECT
# include <click/cxxunprotect.h>
#endif

#define THIS_MODULE VLM_udatapath
#include "lib/vlog.hh"

CLICK_DECLS

extern char mfr_desc;
extern char hw_desc;
extern char sw_desc;
extern char serial_num;

char g_ofpt_name[100][100];



/* should go to static function in class */

static void * make_openflow_reply(size_t openflow_len, uint8_t type,
                struct rconn_remote *rconn_sender, uint32_t xid, struct ofpbuf **bufferp)
{
        return make_openflow_xid(openflow_len, type, rconn_sender ? xid : 0,
                        bufferp);
}




static void fill_queue_desc(struct ofpbuf *buffer, struct sw_queue *q,
		struct ofp_packet_queue *desc)
{
	struct ofp_queue_prop_min_rate *mr;
	int len;

	len = sizeof(struct ofp_packet_queue) +
		sizeof(struct ofp_queue_prop_min_rate);
	desc->queue_id = htonl(q->queue_id);
	desc->len = htons(len);

	/* Property list */
	mr = (struct ofp_queue_prop_min_rate*)ofpbuf_put_zeros(buffer, sizeof *mr);
	mr->prop_header.property = htons(OFPQT_MIN_RATE);
	len = sizeof(struct ofp_queue_prop_min_rate);
	mr->prop_header.len = htons(len);
	mr->rate = htons(q->min_rate);
}
struct sw_queue * dp_lookup_queue(struct click_port *p, uint32_t queue_id)
{
	struct sw_queue *q;

	LIST_FOR_EACH(q, struct sw_queue, node, &p->queue_list) {
		if (q->queue_id == queue_id) {
			return q;
		}
	}
	return NULL;
}
/* Generates and returns a random datapath id. */
static uint64_t gen_datapath_id(void)
{
	uint8_t ea[ETH_ADDR_LEN];
	eth_addr_random(ea);
	ea[0] = 0x00;               /* Set Nicira OUI. */
	ea[1] = 0x23;
	ea[2] = 0x20;
	return eth_addr_to_uint64(ea);
}
static void fill_flow_stats(struct ofpbuf *buffer, struct sw_flow *flow,
		int table_idx, uint64_t now)
{
	struct ofp_flow_stats *ofs;
	int length = sizeof *ofs + flow->sf_acts->actions_len;
	uint64_t tdiff = now - flow->created;
	uint32_t sec = tdiff / 1000;
	ofs = (struct ofp_flow_stats*) ofpbuf_put_uninit(buffer, length);
	ofs->length          = htons(length);
	ofs->table_id        = table_idx;
	ofs->pad             = 0;
	ofs->match.wildcards = htonl(flow->key.wildcards);
	ofs->match.in_port   = flow->key.flow.in_port;
	memcpy(ofs->match.dl_src, flow->key.flow.dl_src, ETH_ADDR_LEN);
	memcpy(ofs->match.dl_dst, flow->key.flow.dl_dst, ETH_ADDR_LEN);
	ofs->match.dl_vlan   = flow->key.flow.dl_vlan;
	ofs->match.dl_type   = flow->key.flow.dl_type;
	ofs->match.nw_tos    = flow->key.flow.nw_tos;
	ofs->match.nw_src    = flow->key.flow.nw_src;
	ofs->match.nw_dst    = flow->key.flow.nw_dst;
	ofs->match.nw_proto  = flow->key.flow.nw_proto;
	ofs->match.dl_vlan_pcp = flow->key.flow.dl_vlan_pcp;
	ofs->match.tp_src    = flow->key.flow.tp_src;
	ofs->match.tp_dst    = flow->key.flow.tp_dst;
	ofs->duration_sec    = htonl(sec);
	ofs->duration_nsec   = htonl((tdiff - (sec * 1000)) * 1000000);
	ofs->cookie          = htonll(flow->cookie);
	ofs->priority        = htons(flow->priority);
	ofs->idle_timeout    = htons(flow->idle_timeout);
	ofs->hard_timeout    = htons(flow->hard_timeout);
	memset(&ofs->pad2, 0, sizeof ofs->pad2);
	ofs->packet_count    = htonll(flow->packet_count);
	ofs->byte_count      = htonll(flow->byte_count);
	memcpy(ofs->actions, flow->sf_acts->actions, flow->sf_acts->actions_len);
}
struct flow_stats_state {
	int table_idx;
	struct sw_table_position position;
	struct ofp_flow_stats_request rq;
	uint64_t now;                  /* Current time in milliseconds */

	struct ofpbuf *buffer;
};



static int desc_stats_dump(class Datapath *dp UNUSED, void *state UNUSED,
		struct ofpbuf *buffer)
{
	struct ofp_desc_stats *ods = (struct ofp_desc_stats*) ofpbuf_put_uninit(buffer, sizeof *ods);
	/* jyyoo do not use this at the moment */
	/*
	   strncpy(ods->mfr_desc, &mfr_desc, sizeof ods->mfr_desc);
	   strncpy(ods->hw_desc, &hw_desc, sizeof ods->hw_desc);
	   strncpy(ods->sw_desc, &sw_desc, sizeof ods->sw_desc);
	   strncpy(ods->dp_desc, dp->dp_desc, sizeof ods->dp_desc);
	   strncpy(ods->serial_num, &serial_num, sizeof ods->serial_num);
	 */
	return 0;
}


#define MAX_FLOW_STATS_BYTES 4096
#define EMERG_TABLE_ID_FOR_STATS 0xfe

static int flow_stats_init(const void *body, int body_len UNUSED, void **state)
{
	DEBUG_TRACE_FUNCTION_CALL
		const struct ofp_flow_stats_request *fsr = (const struct ofp_flow_stats_request*) body;
	struct flow_stats_state *s = (struct flow_stats_state*) xmalloc(sizeof *s);
	s->table_idx = fsr->table_id == 0xff ? 0 : fsr->table_id;
	memset(&s->position, 0, sizeof s->position);
	s->rq = *fsr;
	*state = s;
	return 0;
}

static int flow_stats_dump_callback(struct sw_flow *flow, void *Private)
{
	DEBUG_TRACE_FUNCTION_CALL
		struct flow_stats_state *s = (struct flow_stats_state*)Private;
	fill_flow_stats(s->buffer, flow, s->table_idx, s->now);
	return s->buffer->size >= MAX_FLOW_STATS_BYTES;
}

static int flow_stats_dump(class Datapath *dp, void *state,
		struct ofpbuf *buffer)
{
	DEBUG_TRACE_FUNCTION_CALL
		struct flow_stats_state *s = (struct flow_stats_state*)state;
	struct sw_flow_key match_key;

	flow_extract_match(&match_key, &s->rq.match);
	s->buffer = buffer;
	s->now = time_msec();

	if (s->rq.table_id == EMERG_TABLE_ID_FOR_STATS) {
		struct sw_table *table = (struct sw_table*) dp->get_chain()->emerg_table;

		table->iterate(table, &match_key, s->rq.out_port,
				&s->position, flow_stats_dump_callback, s);
	} else {
		while (s->table_idx < dp->get_chain()->n_tables
				&& (s->rq.table_id == 0xff || s->rq.table_id == s->table_idx))
		{
			struct sw_table *table = (struct sw_table*) dp->get_chain()->tables[s->table_idx];

			if (table->iterate(table, &match_key, s->rq.out_port,
						&s->position, flow_stats_dump_callback, s))
				break;

			s->table_idx++;
			memset(&s->position, 0, sizeof s->position);
		}
	}
	return s->buffer->size >= MAX_FLOW_STATS_BYTES;
}

static void flow_stats_done(void *state)
{
	DEBUG_TRACE_FUNCTION_CALL
		free(state);
}

struct aggregate_stats_state {
	struct ofp_aggregate_stats_request rq;
};

static int aggregate_stats_init(const void *body, int body_len UNUSED, void **state)
{
	DEBUG_TRACE_FUNCTION_CALL
		const struct ofp_aggregate_stats_request *rq = (const struct ofp_aggregate_stats_request*)body;
	struct aggregate_stats_state *s = (struct aggregate_stats_state*) xmalloc(sizeof *s);
	s->rq = *rq;
	*state = s;
	return 0;
}

static int aggregate_stats_dump_callback(struct sw_flow *flow, void *Private)
{
	DEBUG_TRACE_FUNCTION_CALL;
	struct ofp_aggregate_stats_reply *rpy = (struct ofp_aggregate_stats_reply*) Private;
	rpy->packet_count += flow->packet_count;
	rpy->byte_count += flow->byte_count;
	rpy->flow_count++;
	return 0;
}

static int aggregate_stats_dump(class Datapath *dp, void *state, struct ofpbuf *buffer)
{
	DEBUG_TRACE_FUNCTION_CALL;
	struct aggregate_stats_state *s = (struct aggregate_stats_state*) state;
	struct ofp_aggregate_stats_request *rq = (struct ofp_aggregate_stats_request*) &s->rq;
	struct ofp_aggregate_stats_reply *rpy;
	struct sw_table_position position;
	struct sw_flow_key match_key;
	int table_idx;
	int error;

	rpy = (struct ofp_aggregate_stats_reply*) ofpbuf_put_uninit(buffer, sizeof *rpy);
	memset(rpy, 0, sizeof *rpy);

	flow_extract_match(&match_key, &rq->match);
	table_idx = rq->table_id == 0xff ? 0 : rq->table_id;
	memset(&position, 0, sizeof position);

	if (rq->table_id == EMERG_TABLE_ID_FOR_STATS) {
		struct sw_table *table = (struct sw_table*) dp->get_chain()->emerg_table;

		error = table->iterate(table, &match_key, rq->out_port, &position,
				aggregate_stats_dump_callback, rpy);
		if (error)
			return error;
	} else {
		while (table_idx < dp->get_chain()->n_tables
				&& (rq->table_id == 0xff || rq->table_id == table_idx))
		{
			struct sw_table *table = (struct sw_table*) dp->get_chain()->tables[table_idx];

			error = table->iterate(table, &match_key, rq->out_port, &position,
					aggregate_stats_dump_callback, rpy);
			if (error)
				return error;

			table_idx++;
			memset(&position, 0, sizeof position);
		}
	}

	rpy->packet_count = htonll(rpy->packet_count);
	rpy->byte_count = htonll(rpy->byte_count);
	rpy->flow_count = htonl(rpy->flow_count);
	return 0;
}

static void aggregate_stats_done(void *state)
{
	DEBUG_TRACE_FUNCTION_CALL
		free(state);
}

static int table_stats_dump(class Datapath *dp, void *state UNUSED,
		struct ofpbuf *buffer)
{
	DEBUG_TRACE_FUNCTION_CALL
		int i;
	for (i = 0; i < dp->get_chain()->n_tables; i++) {
		struct ofp_table_stats *ots = (struct ofp_table_stats*) ofpbuf_put_uninit(buffer, sizeof *ots);
		struct sw_table_stats stats;
		dp->get_chain()->tables[i]->stats(dp->get_chain()->tables[i], &stats);
		strncpy(ots->name, stats.name, sizeof ots->name);
		ots->table_id = i;
		ots->wildcards = htonl(stats.wildcards);
		memset(ots->pad, 0, sizeof ots->pad);
		ots->max_entries = htonl(stats.max_flows);
		ots->active_count = htonl(stats.n_flows);
		ots->lookup_count = htonll(stats.n_lookup);
		ots->matched_count = htonll(stats.n_matched);
	}
	return 0;
}

struct port_stats_state {
	int start_port;	/* port to start dumping from */
	int port_no;	/* from ofp_stats_request */
};

struct queue_stats_state {
	uint16_t port;
	uint32_t queue_id;
};

static int port_stats_init(const void *body, int /*body_len*/, void **state)
{
	struct port_stats_state *s = (struct port_stats_state*) xmalloc(sizeof *s);
	struct ofp_port_stats_request *psr = (struct ofp_port_stats_request*)body;

	s->start_port = 0;
	s->port_no = ntohs(psr->port_no);
	*state = s;
	return 0;
}

static void dump_port_stats(struct click_port *port, struct ofpbuf *buffer)
{
	struct ofp_port_stats *ops = (struct ofp_port_stats*) ofpbuf_put_uninit(buffer, sizeof *ops);
	ops->port_no = htons(port->port_no);
	memset(ops->pad, 0, sizeof ops->pad);
	ops->rx_packets   = htonll(port->rx_packets);
	ops->tx_packets   = htonll(port->tx_packets);
	ops->rx_bytes     = htonll(port->rx_bytes);
	ops->tx_bytes     = htonll(port->tx_bytes);
	ops->rx_dropped   = htonll(-1);
	ops->tx_dropped   = htonll(port->tx_dropped);
	ops->rx_errors    = htonll(-1);
	ops->tx_errors    = htonll(-1);
	ops->rx_frame_err = htonll(-1);
	ops->rx_over_err  = htonll(-1);
	ops->rx_crc_err   = htonll(-1);
	ops->collisions   = htonll(-1);
}

static int port_stats_dump(class Datapath *dp, void *state,
		struct ofpbuf *buffer)
{
	struct port_stats_state *s = (struct port_stats_state*) state;
	struct click_port *p = NULL;
	int i = 0;

	if (s->port_no == OFPP_NONE) {
		/* Dump statistics for all ports */
		for (i = s->start_port; i < DP_MAX_PORTS; i++) {
			p = dp->dp_lookup_port(i);
			if ( p && p->port_on_use ) {
				dump_port_stats(p, buffer);
			}
		}
	} else {
		/* Dump statistics for a single port */
		p = dp->dp_lookup_port(s->port_no);
		if (p) {
			dump_port_stats(p, buffer);
		}
	}

	return 0;
}

static void port_stats_done(void *state)
{
	free(state);
}

static int queue_stats_init(const void *body, int body_len UNUSED, void **state)
{
	const struct ofp_queue_stats_request *qsr = (const struct ofp_queue_stats_request*) body;
	struct queue_stats_state *s = (struct queue_stats_state*) xmalloc(sizeof *s);
	s->port = ntohs(qsr->port_no);
	s->queue_id = ntohl(qsr->queue_id);
	*state = s;
	return 0;
}

static void dump_queue_stats(struct sw_queue *q, struct ofpbuf *buffer)
{
	struct ofp_queue_stats *oqs = (struct ofp_queue_stats*) ofpbuf_put_uninit(buffer, sizeof *oqs);
	oqs->port_no = htons(q->port->port_no);
	oqs->queue_id = htonl(q->queue_id);
	oqs->tx_bytes = htonll(q->tx_bytes);
	oqs->tx_packets = htonll(q->tx_packets);
	oqs->tx_errors = htonll(q->tx_errors);
}

static int queue_stats_dump(class Datapath *dp, void *state,
		struct ofpbuf *buffer)
{
	struct queue_stats_state *s = (struct queue_stats_state*) state;
	struct sw_queue *q;
	struct click_port *p;


	if (s->port == OFPP_ALL) {
		LIST_FOR_EACH(p, struct click_port, node, dp->get_port_list()) {
			if (p->port_no < OFPP_MAX) {
				if (s->queue_id == OFPQ_ALL) {
					LIST_FOR_EACH(q, struct sw_queue, node, &p->queue_list) {
						dump_queue_stats(q,buffer);
					}
				}
				else {
					q = (struct sw_queue*) dp_lookup_queue(p, s->queue_id);
					if (q) {
						dump_queue_stats(q, buffer);
					}
				}
			}
		}
	}
	else {
		p = dp->dp_lookup_port(s->port);
		if (p) {
			if (s->queue_id == OFPQ_ALL) {
				LIST_FOR_EACH(q, struct sw_queue, node, &p->queue_list) {
					dump_queue_stats(q,buffer);
				}
			}
			else {
				q = dp_lookup_queue(p, s->queue_id);
				if (q) {
					dump_queue_stats(q, buffer);
				}
			}
		}
	}
	return 0;
}

static void queue_stats_done(void *state)
{
	free(state);
}

/*
 * We don't define any vendor_stats_state, we let the actual
 * vendor implementation do that.
 * The only requirement is that the first member of that object
 * should be the vendor id.
 * Jean II
 *
 * Basically, it would look like :
 * struct acme_stats_state {
 *   uint32_t              vendor;         // ACME_VENDOR_ID.
 * <...>                                  // Other stuff.
 * };
 */
static int vendor_stats_init(const void *body, int body_len UNUSED,
		void **state UNUSED)
{
	/* min_body was checked, this should be safe */
	const uint32_t vendor = ntohl(*((uint32_t *)body));
	int err;

	switch (vendor) {
		default:
			err = -EINVAL;
	}

	return err;
}

static int vendor_stats_dump(class Datapath *dp UNUSED, void *state,
		struct ofpbuf *buffer UNUSED)
{
	const uint32_t vendor = *((uint32_t *)state);
	int err;

	switch (vendor) {
		default:
			/* Should never happen */
			err = 0;
	}

	return err;
}

static void vendor_stats_done(void *state)
{
	const uint32_t vendor = *((uint32_t *) state);

	switch (vendor) {
		default:
			/* Should never happen */
			free(state);
	}

	return;
}

struct stats_type {
	/* Value for 'type' member of struct ofp_stats_request. */
	int type;

	/* Minimum and maximum acceptable number of bytes in body member of
	 * struct ofp_stats_request. */
	size_t min_body, max_body;

	/* Prepares to dump some kind of datapath statistics.  'body' and
	 * 'body_len' are the 'body' member of the struct ofp_stats_request.
	 * Returns zero if successful, otherwise a negative error code.
	 * May initialize '*state' to state information.  May be null if no
	 * initialization is required.*/
	int (*init)(const void *body, int body_len, void **state);

	/* Appends statistics for 'dp' to 'buffer', which initially contains a
	 * struct ofp_stats_reply.  On success, it should return 1 if it should be
	 * called again later with another buffer, 0 if it is done, or a negative
	 * errno value on failure. */
	int (*dump)(class Datapath *dp, void *state, struct ofpbuf *buffer);

	/* Cleans any state created by the init or dump functions.  May be null
	 * if no cleanup is required. */
	void (*done)(void *state);
};

static const struct stats_type stats[] = {
	{
		OFPST_DESC,
		0,
		0,
		NULL,
		desc_stats_dump,
		NULL
	},
	{
		OFPST_FLOW,
		sizeof(struct ofp_flow_stats_request),
		sizeof(struct ofp_flow_stats_request),
		flow_stats_init,
		flow_stats_dump,
		flow_stats_done
	},
	{
		OFPST_AGGREGATE,
		sizeof(struct ofp_aggregate_stats_request),
		sizeof(struct ofp_aggregate_stats_request),
		aggregate_stats_init,
		aggregate_stats_dump,
		aggregate_stats_done
	},
	{
		OFPST_TABLE,
		0,
		0,
		NULL,
		table_stats_dump,
		NULL
	},
	{
		OFPST_PORT,
		sizeof(struct ofp_port_stats_request),
		sizeof(struct ofp_port_stats_request),
		port_stats_init,
		port_stats_dump,
		port_stats_done
	},
	{
		OFPST_QUEUE,
		sizeof(struct ofp_queue_stats_request),
		sizeof(struct ofp_queue_stats_request),
		queue_stats_init,
		queue_stats_dump,
		queue_stats_done
	},
	{
		OFPST_VENDOR,
		8,             /* vendor + subtype */
		32,            /* whatever */
		vendor_stats_init,
		vendor_stats_dump,
		vendor_stats_done
	},
};

struct stats_dump_cb {
	bool done;
	struct ofp_stats_request *rq;
	struct rconn_remote *sender;
	uint32_t xid;
	const struct stats_type *s;
	void *state;
};

static int stats_dump(class Datapath *dp, void *cb_)
{
	struct stats_dump_cb *cb = (struct stats_dump_cb*)cb_;
	struct ofp_stats_reply *osr;
	struct ofpbuf *buffer;
	int err;

	if (cb->done) {
		return 0;
	}

	DEBUG_TRACE_FUNCTION_CALL;
	osr = (struct ofp_stats_reply*) make_openflow_reply(sizeof *osr, OFPT_STATS_REPLY, cb->sender, cb->xid, &buffer);
	osr->type = htons(cb->s->type);
	osr->flags = 0;

	err = cb->s->dump(dp, cb->state, buffer);
	if (err >= 0) {
		int err2;
		if (!err) {
			cb->done = true;
		} else {
			/* Buffer might have been reallocated, so find our data again. */
			osr = (struct ofp_stats_reply*)ofpbuf_at_assert(buffer, 0, sizeof *osr);
			osr->flags = ntohs(OFPSF_REPLY_MORE);
		}
		err2 = dp->send_openflow_buffer(buffer, cb->sender, cb->xid);
		if (err2) {
			err = err2;
		}
	}

	return err;
}

static void stats_done(void *cb_)
{
	struct stats_dump_cb *cb = (struct stats_dump_cb*) cb_;
	if (cb) {
		if (cb->s->done) {
			cb->s->done(cb->state);
		}
		if (cb->rq) {
			free(cb->rq);
		}
		free(cb);
	}
}

/* Capabilities supported by this implementation. */
#define OFP_SUPPORTED_CAPABILITIES ( OFPC_FLOW_STATS        \
		| OFPC_TABLE_STATS        \
		| OFPC_PORT_STATS        \
		| OFPC_QUEUE_STATS       \
		| OFPC_ARP_MATCH_IP )

/* Actions supported by this implementation. */
#define OFP_SUPPORTED_ACTIONS ( (1 << OFPAT_OUTPUT)         \
		| (1 << OFPAT_SET_VLAN_VID) \
		| (1 << OFPAT_SET_VLAN_PCP) \
		| (1 << OFPAT_STRIP_VLAN)   \
		| (1 << OFPAT_SET_DL_SRC)   \
		| (1 << OFPAT_SET_DL_DST)   \
		| (1 << OFPAT_SET_NW_SRC)   \
		| (1 << OFPAT_SET_NW_DST)   \
		| (1 << OFPAT_SET_TP_SRC)   \
		| (1 << OFPAT_SET_TP_DST)   \
		| (1 << OFPAT_ENQUEUE))


Datapath::Datapath(): base_timer(this), vxs_timer(this),
	vxsManager(&taskQueueIncoming,&taskQueueOutgoing), 
	vxsDispatcher(&taskQueueIncoming, &taskQueueOutgoing)
{
	strcpy(_mfr_desc, "Stanford University");
	strcpy(_hw_desc, "Reference Userspace Switch");
	strcpy(_sw_desc, "Datapath Element");
	strcpy(_dp_desc, "");
	strcpy(_serial_num, "None");

	_dp = NULL;
	_rconn = NULL;
	_dpid = 0xffffffffffffffffLL;
	strcpy(_str_port_list, "ath0"); // ath0,ath1,ath2
	strcpy(_str_local_port, "tap0:");
	_num_queues = NETDEV_MAX_QUEUES;

	/*Immutablemessages.*/
	strcpy(g_ofpt_name[OFPT_HELLO],"OFPT_HELLO");
	strcpy(g_ofpt_name[OFPT_ERROR],"OFPT_ERROR");
	strcpy(g_ofpt_name[OFPT_ECHO_REQUEST],"OFPT_ECHO_REQUEST");
	strcpy(g_ofpt_name[OFPT_ECHO_REPLY],"OFPT_ECHO_REPLY");
	strcpy(g_ofpt_name[OFPT_VENDOR],"OFPT_VENDOR");
	strcpy(g_ofpt_name[OFPT_FEATURES_REQUEST],"OFPT_FEATURES_REQUEST");
	strcpy(g_ofpt_name[OFPT_FEATURES_REPLY],"OFPT_FEATURES_REPLY");
	strcpy(g_ofpt_name[OFPT_GET_CONFIG_REQUEST],"OFPT_GET_CONFIG_REQUEST");
	strcpy(g_ofpt_name[OFPT_GET_CONFIG_REPLY],"OFPT_GET_CONFIG_REPLY");
	strcpy(g_ofpt_name[OFPT_SET_CONFIG],"OFPT_SET_CONFIG");
	strcpy(g_ofpt_name[OFPT_PACKET_IN],"OFPT_PACKET_IN");
	strcpy(g_ofpt_name[OFPT_FLOW_REMOVED],"OFPT_FLOW_REMOVED");
	strcpy(g_ofpt_name[OFPT_PORT_STATUS],"OFPT_PORT_STATUS");
	strcpy(g_ofpt_name[OFPT_PACKET_OUT],"OFPT_PACKET_OUT");
	strcpy(g_ofpt_name[OFPT_FLOW_MOD],"OFPT_FLOW_MOD");
	strcpy(g_ofpt_name[OFPT_PORT_MOD],"OFPT_PORT_MOD");
	strcpy(g_ofpt_name[OFPT_STATS_REQUEST],"OFPT_STATS_REQUEST");
	strcpy(g_ofpt_name[OFPT_STATS_REPLY],"OFPT_STATS_REPLY");
	strcpy(g_ofpt_name[OFPT_BARRIER_REQUEST],"OFPT_BARRIER_REQUEST");
	strcpy(g_ofpt_name[OFPT_BARRIER_REPLY],"OFPT_BARRIER_REPLY");
	strcpy(g_ofpt_name[OFPT_QUEUE_GET_CONFIG_REQUEST],"OFPT_QUEUE_GET_CONFIG_REQUEST");
	strcpy(g_ofpt_name[OFPT_QUEUE_GET_CONFIG_REPLY],"OFPT_QUEUE_GET_CONFIG_REPLY");
}


Datapath::~Datapath()
{
}

void Datapath::cleanup(CleanupStage stage UNUSED)
{
}

int Datapath::send_openflow_buffer(struct ofpbuf *buffer, struct rconn_remote *rconn_sender, uint32_t xid )
{
	if( _rconn ) {
		_rconn->send_openflow_buffer( buffer, rconn_sender, xid );
	}

	return 0;
}


int Datapath::dp_add_port(const char *devname, const uint8_t *macaddr, uint16_t num_queues, uint16_t click_port_num)
{
	int port_no;
	for (port_no = 0; port_no < DP_MAX_PORTS; port_no++) {
		struct click_port *port = &_ports[port_no];
		if( port->devname[0] == 0 )
		return new_port(port, port_no, devname, macaddr, num_queues, click_port_num);
	}

	return EXFULL;
}

int Datapath::dp_add_local_port(const char *devname, uint16_t num_queues)
{
	uint8_t ea[ETH_ADDR_LEN];
	int error;
	struct click_port *port = &_ports[DP_MAX_PORTS-1];
	eth_addr_from_uint64(_id, ea);
	error = new_port( port, OFPP_LOCAL, devname, ea, num_queues, CLICK_PORT_NUM_TAP );
	if (!error) {
		_local_port = port;
	}
	return error;
}

/*
 * ngkim: add datapath
 */
void Datapath::send_port_status(struct click_port *p, uint8_t status)
{
	struct ofpbuf *buffer;
	struct ofp_port_status *ops;

	DEBUG_TRACE_FUNCTION_CALL;

	ops = (struct ofp_port_status*)make_openflow_xid(sizeof *ops, OFPT_PORT_STATUS, 0, &buffer);
	ops->reason = status;
	memset(ops->pad, 0, sizeof ops->pad);
	// ngkim: modify - add dp
	fill_port_desc(p, &ops->desc);

	

	send_openflow_buffer(buffer, NULL, 0);
}

void Datapath::fill_port_desc(struct click_port *p, struct ofp_phy_port *desc, struct ofpbuf* buffer UNUSED)
{
	memset( desc, 0xff, sizeof( *desc ) );
	desc->port_no = htons(p->port_no);

	strncpy((char *) desc->name, p->devname, sizeof desc->name);
	desc->name[sizeof desc->name - 1] = '\0';
	/* I don't know how to deal with this at the moment */
	desc->curr = 0;
	desc->supported = 0;
	desc->advertised = 0;
	memcpy(desc->hw_addr, p->macaddr, ETH_ADDR_LEN);
	desc->config = htonl(p->config);
	desc->state = htonl(p->state);

	/* TODO: make a note for this confusion
	 * jyyoo note: we should use click_port_num */
	/* ngkim: call Datapath element to get port descriptions */
	port_description(desc, p->click_port_num);

	/* change to network-order */
	desc->peer = htonl(desc->peer);
	desc->curr = htonl(desc->curr);
	desc->supported  = htonl(desc->supported);
	desc->advertised  = htonl(desc->advertised);

}


int Datapath::configure(Vector<String> &conf, ErrorHandler* errh)
{
	DEBUG_TRACE_FUNCTION_CALL;

  	bool have_dpid;

	for( int i = 0; i<DP_MAX_PORTS; i++ )
	{
		_have_port_macaddr[i] = false;
		_have_port_devname[i] = false;
	}


	/* PORT0 is registered for tun/tap device */

	uint32_t dpid;
	/* read port configuration */
	if (cp_va_kparse(conf, this, errh,
				/* device parameter setting */
				"HOST_ADDR", cpkP, cpEthernetAddress, &hostMacAddr,
				"PORT1_ADDR", cpkC, &(_have_port_macaddr[0]), cpEthernetAddress, &_port_macaddr[0],
				"PORT2_ADDR", cpkC, &_have_port_macaddr[1], cpEthernetAddress, &_port_macaddr[1],
				"PORT3_ADDR", cpkC, &_have_port_macaddr[2], cpEthernetAddress, &_port_macaddr[2],
				"PORT4_ADDR", cpkC, &_have_port_macaddr[3], cpEthernetAddress, &_port_macaddr[3],
				"PORT5_ADDR", cpkC, &_have_port_macaddr[4], cpEthernetAddress, &_port_macaddr[4],

				"PORT1_NAME", cpkC, &_have_port_devname[0], cpString, &_port_devname[0],
				"PORT2_NAME", cpkC, &_have_port_devname[1], cpString, &_port_devname[1],
				"PORT3_NAME", cpkC, &_have_port_devname[2], cpString, &_port_devname[2],
				"PORT4_NAME", cpkC, &_have_port_devname[3], cpString, &_port_devname[3],
				"PORT5_NAME", cpkC, &_have_port_devname[4], cpString, &_port_devname[4],
        	               
	        	        "DPID", cpkC, &have_dpid, cpUnsigned, &dpid, 

				cpEnd) < 0 )
	{
		return -1;
	}

	time_init();

	if( have_dpid ) {
		_dpid = dpid;
	}

	/* jyyoo add to init of_actions */
	init_of_actions();

	/* dp_new */
	_id = _dpid <= 0xffffffffffffffffLL ? _dpid : gen_datapath_id();
	_chain = chain_create(this);
	if (_chain == NULL) {
		Timestamp now = Timestamp::now();
		click_chatter("%{timestamp} %{element}: Datapath Error: could not create chain\n", &now, this);

		return ENOMEM;
	}

	list_init(&_port_list);
	_flags = 0;
	_miss_send_len = OFP_DEFAULT_MISS_SEND_LEN;

	char hostnametmp[DESC_STR_LEN];
	gethostname(hostnametmp,sizeof hostnametmp);
	snprintf(_dp_desc, sizeof _dp_desc, "%s pid=%u",hostnametmp, getpid());

	/* we connect to rconn directly here */
	Router* router = this->router();
    	_rconn = (RConn*)router->find("rc", errh );
	if( _rconn == NULL ) {
		click_chatter("You need to use RConn with the name rc element\n");
		return -1;
	}
	_rconn->_datapath = this;

	vxsDispatcher.startDispatching( 4 ); /* generate 4 threads */

	return 0;
}


int Datapath::initialize(ErrorHandler*)
{
        int res=0;
        base_timer.initialize(this);
        vxs_timer.initialize(this);
        if( BASE_TIMER_INTERVAL ) base_timer.schedule_after_msec(BASE_TIMER_INTERVAL);
        if( VXS_TIMER_INTERVAL ) vxs_timer.schedule_after_msec(VXS_TIMER_INTERVAL);
	memset( &fs, 0, sizeof(fs) );


	memset( _ports, 0, sizeof(_ports) );

	/*
	 * jyyoo NOTE: rather than adding port here, it must be done by Click
	 * TODO: Make a clear note about this click port number
	 */
	int click_port_num=CLICK_PORT_START_NUM_ITF;
	for( int i = 0; i<DP_MAX_PORTS; i++ )
	{
		if( _have_port_macaddr[i] == true && _have_port_devname[i] == true )
		{
			dp_add_port( _port_devname[i].c_str(), _port_macaddr[i].data(), _num_queues, click_port_num );
			click_port_num++;
		} else if( _have_port_macaddr[i] == false && _have_port_devname[i] == false )
		{/* do nothing */
		} else {
			Timestamp now = Timestamp::now();
			click_chatter("%{timestamp} %{element}: Error! only specify either of" 
					"macaddr or devname for ports: %d \n", &now, this, i);

			return -1;
		}
	}

	dp_add_local_port( _str_local_port, 0);


        return res;
}


void Datapath::push(int input, Packet* p)
{
	click_ether *ethdr = (click_ether*) p->data();

	if( input == 0 ) {/* designated port where control (nox) packets are incoming  */
		/* this path is deprecated */
	}
	else {
	
		/* bypass configuration packets */
		uint16_t type = htons(ethdr->ether_type);
		if(	type != ETHERTYPE_IP &&
				type != ETHERTYPE_ARP &&
				type != ETHERTYPE_TRAIL &&
				type != ETHERTYPE_8021Q &&
				//		type != ETHERTYPE_IP6 &&
				type != ETHERTYPE_MACCONTROL &&
				type != ETHERTYPE_LLDP &&
				type != ETHERTYPE_GRID  ) 
		{
			p->kill();
			return;
		}


		if( type == ETHERTYPE_IP ) {

			click_ip *iphdr = (click_ip*) (p->data() + sizeof(click_ether));
			IPAddress ip( iphdr->ip_dst );

			if( ip.is_multicast() ) 
			{
				//			printf("discard multicast packets\n" );
				p->kill();
				return;
			}
		}



		//first find out which sw_port belongs to this input
		for( uint32_t i = 0; i<DP_MAX_PORTS; i++ )
		{
			if( _ports[i].click_port_num == input ) 
			{		

#if JD_DEBUG == 1
				printf("==> (PUSH) packet in [size:%d]", p->length());
				if( type == ETHERTYPE_IP ) printf(" [IP] ");
				if( type == ETHERTYPE_ARP ) printf(" [ARP] ");
				if( type == ETHERTYPE_IP6 ) printf(" [IP6] ");
				/* debugging */
				for( uint32_t j = 0; j< ( p->length() < 80U ? p->length() : 80 ) ;j ++ )
				{
					if( j % 16 == 0 ) printf("\n\t");
					printf("%x ", *((uint8_t*)p->data() + j));

				}
				printf("\n");
#endif
				// TODO: this conversion is just to avoid compile error
				struct ofpbuf *buffer = packet_to_ofpbuf_with_headroom( p, 
						offsetof(struct ofp_packet_in, data) );

				PRINT_OFP( buffer );

				_ports[i].rx_packets ++;
				_ports[i].rx_bytes += p->length();
				fwd_port_input( buffer, &(_ports[i]) );

				p->kill();
				return;
			}
		}
		Timestamp now = Timestamp::now();
		click_chatter("%{timestamp} %{element}: Error! unspecified input port receives a packet: input port=%d\n", &now, this, input);

	}
}


void Datapath::add_handlers()
{
}

/* timer based report function */
void Datapath::run_timer(Timer* t)
{
        static int c=0;
        c++;
        if( t == &base_timer ) {
		memset( &fs, 0, sizeof(fs) );
                t->schedule_after_msec( BASE_TIMER_INTERVAL );
		/* TODO: report interval-based status */

        } else if( t == &vxs_timer ) {
		vxsManager.sendPacket( this );
                t->schedule_after_msec( VXS_TIMER_INTERVAL );
	}
}

/* 
 * jyyoo add
 * This should be adopted to fill the port information of Click ports 
 */
void Datapath::dp_send_features_reply(struct rconn_remote *rconn_sender, uint32_t xid)
{
	DEBUG_TRACE_FUNCTION_CALL;
	struct ofpbuf *buffer;
	struct ofp_switch_features *ofr;
	struct click_port *p;


	ofr = (struct ofp_switch_features*) make_openflow_reply(sizeof *ofr, OFPT_FEATURES_REPLY,
			rconn_sender, xid, &buffer);
	ofr->datapath_id  = htonll(_id);
	ofr->n_tables     = _chain->n_tables;
	ofr->n_buffers    = htonl(N_PKT_BUFFERS);
	ofr->capabilities = htonl(OFP_SUPPORTED_CAPABILITIES);
	ofr->actions      = htonl(OFP_SUPPORTED_ACTIONS);

	LIST_FOR_EACH (p, struct click_port, node, &_port_list) {
		struct ofp_phy_port *opp = (struct ofp_phy_port*)ofpbuf_put_uninit(buffer, sizeof *opp);
		memset(opp, 0, sizeof *opp);
		// ngkim: modify calling fill_port_desc 
		fill_port_desc(p, opp, buffer);
	}

	PRINT_OFP(buffer)
	send_openflow_buffer(buffer, rconn_sender, xid);
}

int Datapath::new_port(struct click_port *port, uint16_t port_no,
		const char *netdev_name, const uint8_t *macaddr, uint16_t num_queues, uint16_t click_port_num)
{
	DEBUG_TRACE_FUNCTION_CALL

	memset(port, '\0', sizeof *port);

	list_init(&port->queue_list);
	memset( port->devname, 0, sizeof(port->devname) );

	strcpy(port->devname, netdev_name);
	memcpy(port->macaddr, macaddr, ETH_ADDR_LEN);
	port->datapath = this;
	port->port_no = port_no;
	port->num_queues = num_queues;
	port->click_port_num = click_port_num;
	port->port_on_use = 1;

	list_push_back(&_port_list, &port->node);


	/* 
	 * ngkim: add dp 
 	 * Notify the ctlpath that this port has been added 
	 */
	send_port_status(port, OFPPR_ADD);

	return 0;
}



struct click_port * Datapath::dp_lookup_port(uint16_t port_no)
{
	return (port_no < DP_MAX_PORTS ? &_ports[port_no] : NULL);
}

void Datapath::update_port_flags(const struct ofp_port_mod *opm)
{
	struct click_port *p = (struct click_port*) dp_lookup_port(ntohs(opm->port_no));

	/* Make sure the port id hasn't changed since this was sent */
	if (!p || memcmp(opm->hw_addr, p->macaddr,
				ETH_ADDR_LEN) != 0) {
		return;
	}


	if (opm->mask) {
		uint32_t config_mask = ntohl(opm->mask);
		p->config &= ~config_mask;
		p->config |= ntohl(opm->config) & config_mask;
	}
}

/* Send packets out all the ports except the originating one.  If the
 * "flood" argument is set, don't send out ports with flooding disabled.
 */
int Datapath::output_all(struct ofpbuf *buffer, int in_port, int flood)
{
	DEBUG_TRACE_FUNCTION_CALL;
	struct click_port *p;
	int prev_port;

	prev_port = -1;
	LIST_FOR_EACH (p, struct click_port, node, &_port_list) {
		if (p->port_no == in_port) {
			continue;
		}
		if (flood && p->config & OFPPC_NO_FLOOD) {
			continue;
		}
		if (prev_port != -1) {
			dp_output_port(ofpbuf_clone(buffer), in_port, prev_port,
					0,false);
		}
		prev_port = p->port_no;
	}
	if (prev_port != -1)
		dp_output_port( buffer, in_port, prev_port, 0, false);
	else
		ofpbuf_delete(buffer);

	return 0;
}

void Datapath::output_packet(struct ofpbuf *buffer, uint16_t out_port, uint32_t queue_id)
{
	DEBUG_TRACE_FUNCTION_CALL;
	Packet* packet = ofpbuf_to_packet( buffer );

#ifdef JY_DEBUG_NO
	if( packet->length() == 128 ) {
		printf("=> Packet OutBuffer [size:%d sizeof: %d]", 
				buffer->allocated, sizeof(buffer->base) );  
		int i;
		for(i=0; i < (buffer->allocated < 80 ? buffer->allocated : 80); i++) { 
			if( i%16==0 ) printf("\n\t");  
			printf("%x ", *( ((uint8_t*)buffer->base)+i) );  
		} 
		printf("\n");
	}
#endif
	ofpbuf_delete(buffer);
	return output_packet( packet, out_port, queue_id );
}

void Datapath::output_packet(Packet *packet, uint16_t out_port, uint32_t queue_id)
{
	DEBUG_TRACE_FUNCTION_CALL;

	struct sw_queue * q;
	struct click_port *port;

	q = NULL;
	port = dp_lookup_port(out_port);
	if( port == NULL) {
		goto error;

	}

	if( !(port->config & OFPPC_PORT_DOWN) ) {
		int result=0;
		/* avoid the queue lookup for best-effort traffic */
		if (queue_id == 0) {
		}
		else {
			/* silently drop the packet if queue doesn't exist */
			q = dp_lookup_queue(port, queue_id);
			if (q) {
			}
			else {
				goto error;
			}
		}

		if( port->click_port_num == 1 /* 1 is for tun/tap */ )
		{/* this one goes to host device */
			printf("go to host device\n");

		} else {

			fs.output_packet ++;
			if( port->click_port_num == 0 ) {
				printf("Error: datapacket is sent to RConn, may be collapse!\n");
			}
			output(port->click_port_num).push( packet );
		}

		// TODO: jyyoo we should check the result of this push
		if( result == 0 ) {
			port->tx_packets++;
			port->tx_bytes += packet->length();
			if (q) {
				q->tx_packets++;
				q->tx_bytes += packet->length();
			}
		} else {
			port->tx_dropped++;
		}
	}
	return;
error:
	packet->kill();
	Timestamp now = Timestamp::now();
	click_chatter("%{timestamp} %{element}: can't forward to bad port:queue(%d:%d)\n", 
			&now, this, out_port, queue_id);
}

/*
 * Takes ownership of 'buffer' and transmits it to 'out_port' on 'dp'.
 */
void Datapath::dp_output_port(Packet *packet, int in_port, int out_port, uint32_t queue_id)
{
	DEBUG_TRACE_FUNCTION_CALL;

	assert(packet);
	switch (out_port) {
		case OFPP_IN_PORT:
			output_packet(packet, in_port, queue_id);
			break;

		case OFPP_TABLE: 
		case OFPP_FLOOD:
		case OFPP_ALL:
		case OFPP_CONTROLLER:
			click_chatter("Warning: not supporting for this operation: %d\n", out_port);
			break;

		case OFPP_LOCAL:
		default:
			if (in_port == out_port) {
			        Timestamp now = Timestamp::now();
			        click_chatter("%{timestamp} %{element}: can't directly forward to input port\n", 
						&now, this);
				return;
			}
			output_packet(packet, out_port, queue_id);
			break;
	}
}

/*
 * Takes ownership of 'buffer' and transmits it to 'out_port' on 'dp'.
 */
void Datapath::dp_output_port(struct ofpbuf *buffer, int in_port, int out_port, uint32_t queue_id, 
				bool ignore_no_fwd UNUSED )
{
	DEBUG_TRACE_FUNCTION_CALL;

	assert(buffer);
	switch (out_port) {
		case OFPP_IN_PORT:
			output_packet(buffer, in_port, queue_id);
			break;

		case OFPP_TABLE: 
			{
				struct click_port *p = dp_lookup_port(in_port);
				if (run_flow_through_tables(buffer, p)) {
					ofpbuf_delete(buffer);
				}
				break;
			}

		case OFPP_FLOOD:
			output_all(buffer, in_port, 1);
			break;

		case OFPP_ALL:
			output_all(buffer, in_port, 0);
			break;

		case OFPP_CONTROLLER:
			dp_output_control(buffer, in_port, UINT16_MAX, OFPR_ACTION);
			break;

		case OFPP_LOCAL:
		default:
			if (in_port == out_port) {
			        Timestamp now = Timestamp::now();
			        click_chatter("%{timestamp} %{element}: can't directly forward to input port\n", 
						&now, this);
				return;
			}
			output_packet(buffer, out_port, queue_id);
			break;
	}
}

/* Takes ownership of 'buffer' and transmits it to 'dp''s controller.  If the
 * packet can be saved in a buffer, then only the first max_len bytes of
 * 'buffer' are sent; otherwise, all of 'buffer' is sent.  'reason' indicates
 * why 'buffer' is being sent. 'max_len' sets the maximum number of bytes that
 * the caller wants to be sent. */
void Datapath::dp_output_control(struct ofpbuf *buffer, int in_port, size_t max_len, int reason)
{
	DEBUG_TRACE_FUNCTION_CALL;

	struct ofp_packet_in *opi;
	size_t total_len;
	uint32_t buffer_id;


	buffer_id = packetbuffer.save_buffer(buffer);
	total_len = buffer->size;
	if (buffer_id != 0xffffffffL && buffer->size > max_len) {
		buffer->size = max_len;
	}

	opi = (struct ofp_packet_in*)ofpbuf_push_uninit(buffer, offsetof(struct ofp_packet_in, data));
	
	opi->header.version = OFP_VERSION;
	opi->header.type    = OFPT_PACKET_IN;
	opi->header.length  = htons(buffer->size);
	opi->header.xid     = htonl(0);
	opi->buffer_id      = htonl(buffer_id);
	opi->total_len      = htons(total_len);
	opi->in_port        = htons(in_port);
	opi->reason         = reason;
	opi->pad            = 0;

	send_openflow_buffer( buffer, NULL, 0);
}

void Datapath::dp_send_flow_end(struct sw_flow *flow, enum ofp_flow_removed_reason reason)
{
	struct ofpbuf *buffer;
	struct ofp_flow_removed *ofr;
	uint64_t tdiff = time_msec() - flow->created;
	uint32_t sec = tdiff / 1000;

	DEBUG_TRACE_FUNCTION_CALL;

	if (!flow->send_flow_rem) {
		return;
	}

	if (flow->emerg_flow) {
		return;
	}

	ofr = (struct ofp_flow_removed*)make_openflow_xid(sizeof *ofr, OFPT_FLOW_REMOVED, 0, &buffer);
	if (!ofr) {
		return;
	}

	flow_fill_match(&ofr->match, &flow->key.flow, flow->key.wildcards);

	ofr->cookie = htonll(flow->cookie);
	ofr->priority = htons(flow->priority);
	ofr->reason = reason;

	ofr->duration_sec = htonl(sec);
	ofr->duration_nsec = htonl((tdiff - (sec * 1000)) * 1000000);
	ofr->idle_timeout = htons(flow->idle_timeout);

	ofr->packet_count = htonll(flow->packet_count);
	ofr->byte_count   = htonll(flow->byte_count);

	send_openflow_buffer( buffer, NULL, 0);
}

/* 'buffer' was received on 'p', which may be a a physical switch port or a
 * null pointer.  Process it according to 'dp''s flow table.  Returns 0 if
 * successful, in which case 'buffer' is destroyed, or -ESRCH if there is no
 * matching flow, in which case 'buffer' still belongs to the caller. */
int Datapath::run_flow_through_tables(struct ofpbuf *buffer, struct click_port *p)
{
	struct sw_flow_key key;
	struct sw_flow *flow;

	key.wildcards = 0;
	if (flow_extract(buffer, p ? p->port_no : (int) OFPP_NONE, &key.flow)
			&& (_flags & OFPC_FRAG_MASK) == OFPC_FRAG_DROP) {
		/* Drop fragment. */
		ofpbuf_delete(buffer);
		return 0;
	}

	if (p && p->config & (OFPPC_NO_RECV | OFPPC_NO_RECV_STP)
			&& p->config & (!eth_addr_equals(key.flow.dl_dst, stp_eth_addr)
				? OFPPC_NO_RECV : OFPPC_NO_RECV_STP)) {
		ofpbuf_delete(buffer);
		return 0;
	}

	flow = chain_lookup(_chain, &key, 0);
	if (flow != NULL) {
		flow_used(flow, buffer);
		execute_actions(this, buffer, &key, flow->sf_acts->actions,
				flow->sf_acts->actions_len, false);
		return 0;
	} else {
		return -ESRCH;
	}
}

/* 'buffer' was received on 'p', which may be a a physical switch port or a
 * null pointer.  Process it according to 'dp''s flow table, sending it up to
 * the controller if no flow matches.  Takes ownership of 'buffer'. */
void Datapath::fwd_port_input(struct ofpbuf *buffer, struct click_port *p)
{
	DEBUG_TRACE_FUNCTION_CALL;

	PRINT_OFP(buffer);
	if (run_flow_through_tables(buffer, p)) {
		dp_output_control(buffer, p->port_no, _miss_send_len, OFPR_NO_MATCH);
	}
}

int Datapath::recv_features_request(struct rconn_remote *rconn_sender, uint32_t xid, const void *msg UNUSED)
{
	dp_send_features_reply(rconn_sender, xid);
	return 0;
}

int Datapath::recv_get_config_request(struct rconn_remote *rconn_sender, uint32_t xid,	const void *msg UNUSED)
{
	struct ofpbuf *buffer;
	struct ofp_switch_config *osc;

	osc = (struct ofp_switch_config*) make_openflow_reply(sizeof *osc, OFPT_GET_CONFIG_REPLY,
			rconn_sender, xid, &buffer);

	osc->flags = htons(_flags);
	osc->miss_send_len = htons(_miss_send_len);

	return send_openflow_buffer(buffer, rconn_sender, xid);
}

int Datapath::recv_set_config(struct rconn_remote *rconn_sender UNUSED, uint32_t xid UNUSED, const void *msg)
{
	DEBUG_TRACE_FUNCTION_CALL;
	const struct ofp_switch_config *osc = (const struct ofp_switch_config*)msg;
	int flags;

	flags = ntohs(osc->flags) & OFPC_FRAG_MASK;
	if ((flags & OFPC_FRAG_MASK) != OFPC_FRAG_NORMAL
			&& (flags & OFPC_FRAG_MASK) != OFPC_FRAG_DROP) {
		flags = (flags & ~OFPC_FRAG_MASK) | OFPC_FRAG_DROP;
	}
	_flags = flags;
	_miss_send_len = ntohs(osc->miss_send_len);

	return 0;
}

int Datapath::recv_packet_out(struct rconn_remote *rconn_sender, uint32_t xid UNUSED, const void *msg)
{
	DEBUG_TRACE_FUNCTION_CALL
	const struct ofp_packet_out *opo = (const struct ofp_packet_out*)msg;
	struct sw_flow_key key;
	uint16_t v_code;
	struct ofpbuf *buffer;
	size_t actions_len = ntohs(opo->actions_len);

	if (actions_len > (ntohs(opo->header.length) - sizeof *opo)) {
		Timestamp now = Timestamp::now();
		click_chatter("%{timestamp} %{element}: message too short for number of actions \n", &now, this);

		return -EINVAL;
	}

	if (ntohl(opo->buffer_id) == (uint32_t) -1) {
		/* FIXME: can we avoid copying data here? */
		int data_len = ntohs(opo->header.length) - sizeof *opo - actions_len;
		buffer = (struct ofpbuf*) ofpbuf_new(data_len);
		ofpbuf_put(buffer, (uint8_t *)opo->actions + actions_len, data_len);
	} else {
		buffer = packetbuffer.retrieve_buffer(ntohl(opo->buffer_id));
		if (!buffer) {

			return -ESRCH;
		}
	}
	flow_extract(buffer, ntohs(opo->in_port), &key.flow);

	v_code = validate_actions(this, &key, opo->actions, actions_len);
	if (v_code != ACT_VALIDATION_OK) {
		dp_send_error_msg(rconn_sender, xid, OFPET_BAD_ACTION, v_code,
				msg, ntohs(opo->header.length));
		goto error;
	}
	execute_actions(this, buffer, &key, opo->actions, actions_len, true);

	return 0;

error:
	ofpbuf_delete(buffer);
	return -EINVAL;
}

int Datapath::recv_port_mod(struct rconn_remote *rconn_sender UNUSED, uint32_t xid UNUSED, const void *msg)
{
	DEBUG_TRACE_FUNCTION_CALL;
	const struct ofp_port_mod *opm = (const struct ofp_port_mod*)msg;

	update_port_flags(opm);

	return 0;
}

int Datapath::add_flow(struct rconn_remote *rconn_sender, uint32_t xid UNUSED, const struct ofp_flow_mod *ofm)
{
	DEBUG_TRACE_FUNCTION_CALL;
	int error = -ENOMEM;
	uint16_t v_code;
	struct sw_flow *flow;
	size_t actions_len = ntohs(ofm->header.length) - sizeof *ofm;
	int overlap;

	/* Allocate memory. */
	flow =(struct sw_flow*) flow_alloc(actions_len);
	if (flow == NULL)
	{

		goto error;
	}

	flow_extract_match(&flow->key, &ofm->match);

	v_code = validate_actions(this, &flow->key, ofm->actions, actions_len);
	if (v_code != ACT_VALIDATION_OK) {
		dp_send_error_msg(rconn_sender, xid, OFPET_BAD_ACTION, v_code,
				ofm, ntohs(ofm->header.length));
		goto error_free_flow;
	}

	flow->priority = flow->key.wildcards ? ntohs(ofm->priority) : -1;

	if (ntohs(ofm->flags) & OFPFF_CHECK_OVERLAP) {
		/* check whether there is any conflict */
		overlap = chain_has_conflict(_chain, &flow->key, flow->priority, false);
		if (overlap){
			dp_send_error_msg(rconn_sender, xid, OFPET_FLOW_MOD_FAILED,
					OFPFMFC_OVERLAP, ofm, ntohs(ofm->header.length));
			goto error_free_flow;
		}
	}

	if (ntohs(ofm->flags) & OFPFF_EMERG) {
		if (ntohs(ofm->idle_timeout) != OFP_FLOW_PERMANENT
				|| ntohs(ofm->hard_timeout) != OFP_FLOW_PERMANENT) {
			dp_send_error_msg(rconn_sender, xid, OFPET_FLOW_MOD_FAILED,
					OFPFMFC_BAD_EMERG_TIMEOUT, ofm,
					ntohs(ofm->header.length));
			goto error_free_flow;
		}
	}

	/* Fill out flow. */
	flow->cookie = ntohll(ofm->cookie);
	flow->idle_timeout = ntohs(ofm->idle_timeout);
	flow->hard_timeout = ntohs(ofm->hard_timeout);
	flow->send_flow_rem = (ntohs(ofm->flags) & OFPFF_SEND_FLOW_REM) ? 1 : 0;
	flow->emerg_flow = (ntohs(ofm->flags) & OFPFF_EMERG) ? 1 : 0;
	flow_setup_actions(flow, ofm->actions, actions_len);

	/* Act. */
	error = chain_insert(_chain, flow,
			(ntohs(ofm->flags) & OFPFF_EMERG) ? 1 : 0);
	if (error == -ENOBUFS) {
		dp_send_error_msg(rconn_sender, xid, OFPET_FLOW_MOD_FAILED,
				OFPFMFC_ALL_TABLES_FULL, ofm, ntohs(ofm->header.length));
		goto error_free_flow;
	} else if (error) {

		goto error_free_flow;
	}

	error = 0;
	if ((uint32_t) ntohl(ofm->buffer_id) != 0xffffffffL) {
		struct ofpbuf *buffer = (struct ofpbuf*)packetbuffer.retrieve_buffer(ntohl(ofm->buffer_id));
		if (buffer) {
			struct sw_flow_key key;
			uint16_t in_port = ntohs(ofm->match.in_port);
			flow_extract(buffer, in_port, &key.flow);
			flow_used(flow, buffer);
			execute_actions(this, buffer, &key,
					ofm->actions, actions_len, false);
		} else {
			error = -ESRCH;
		}
	}

	return error;

error_free_flow:
	flow_free(flow);
error:
	if (ntohl(ofm->buffer_id) != (uint32_t) -1)
		packetbuffer.discard_buffer(ntohl(ofm->buffer_id));
	return error;
}

int Datapath::mod_flow(struct rconn_remote *rconn_sender, uint32_t xid UNUSED, const struct ofp_flow_mod *ofm)
{
	DEBUG_TRACE_FUNCTION_CALL;
	int error = -ENOMEM;
	uint16_t v_code;
	size_t actions_len = ntohs(ofm->header.length) - sizeof *ofm;
	struct sw_flow *flow;
	int strict;

	/* Allocate memory. */
	flow = (struct sw_flow*) flow_alloc(actions_len);
	if (flow == NULL)
	{
		goto error;
	}

	flow_extract_match(&flow->key, &ofm->match);

	v_code = validate_actions(this, &flow->key, ofm->actions, actions_len);
	if (v_code != ACT_VALIDATION_OK) {
		dp_send_error_msg(rconn_sender, xid, OFPET_BAD_ACTION, v_code,
				ofm, ntohs(ofm->header.length));
		goto error_free_flow;
	}

	flow->priority = flow->key.wildcards ? ntohs(ofm->priority) : -1;
	strict = (ofm->command == htons(OFPFC_MODIFY_STRICT)) ? 1 : 0;

	/* First try to modify existing flows if any */
	/* if there is no matching flow, add it */
	if (!chain_modify(_chain, &flow->key, flow->priority,
				strict, ofm->actions, actions_len,
				(ntohs(ofm->flags) & OFPFF_EMERG) ? 1 : 0)) {
		/* Fill out flow. */
		flow->cookie = ntohll(ofm->cookie);
		flow->idle_timeout = ntohs(ofm->idle_timeout);
		flow->hard_timeout = ntohs(ofm->hard_timeout);
		flow->send_flow_rem = (ntohs(ofm->flags) & OFPFF_SEND_FLOW_REM) ? 1 : 0;
		flow->emerg_flow = (ntohs(ofm->flags) & OFPFF_EMERG) ? 1 : 0;
		flow_setup_actions(flow, ofm->actions, actions_len);
		error = chain_insert(_chain, flow,
				(ntohs(ofm->flags) & OFPFF_EMERG) ? 1 : 0);
		if (error == -ENOBUFS) {
			dp_send_error_msg(rconn_sender, xid, OFPET_FLOW_MOD_FAILED,
					OFPFMFC_ALL_TABLES_FULL, ofm,
					ntohs(ofm->header.length));
			goto error_free_flow;
		} else if (error) {
			goto error_free_flow;
		}
	}

	error = 0;
	if ((uint32_t)ntohl(ofm->buffer_id) != 0xffffffffL) {
		struct ofpbuf *buffer = (struct ofpbuf*) packetbuffer.retrieve_buffer(ntohl(ofm->buffer_id));
		if (buffer) {
			struct sw_flow_key skb_key;
			uint16_t in_port = ntohs(ofm->match.in_port);
			flow_extract(buffer, in_port, &skb_key.flow);
			execute_actions(this, buffer, &skb_key,
					ofm->actions, actions_len, false);
		} else {
			error = -ESRCH;
		}
	}

	return error;

error_free_flow:
	flow_free(flow);
error:
	if (ntohl(ofm->buffer_id) != (uint32_t) -1)
		packetbuffer.discard_buffer(ntohl(ofm->buffer_id));
	return error;
}


int Datapath::recv_flow(struct rconn_remote *rconn_sender, uint32_t xid, const void *msg)
{
	DEBUG_TRACE_FUNCTION_CALL;
	const struct ofp_flow_mod *ofm = (const struct ofp_flow_mod*) msg;
	uint16_t command = ntohs(ofm->command);

	if (command == OFPFC_ADD) {
		return add_flow(rconn_sender, xid, ofm);
	} else if ((command == OFPFC_MODIFY) || (command == OFPFC_MODIFY_STRICT)) {
		return mod_flow(rconn_sender, xid, ofm);
	}  else if (command == OFPFC_DELETE) {
		struct sw_flow_key key;
		flow_extract_match(&key, &ofm->match);

		return chain_delete(_chain, &key, ofm->out_port, 0, 0,
				(ntohs(ofm->flags) & OFPFF_EMERG) ? 1 : 0)
			? 0 : -ESRCH;
	} else if (command == OFPFC_DELETE_STRICT) {
		struct sw_flow_key key;
		uint16_t priority;
		flow_extract_match(&key, &ofm->match);
		priority = key.wildcards ? ntohs(ofm->priority) : -1;

		return chain_delete(_chain, &key, ofm->out_port, priority, 1,
				(ntohs(ofm->flags) & OFPFF_EMERG) ? 1 : 0)
			? 0 : -ESRCH;
	} else {

		return -ENODEV;
	}
}

int Datapath::recv_stats_request(struct rconn_remote *rconn_sender, uint32_t xid, const void *oh)
{
	const struct ofp_stats_request *rq = (const struct ofp_stats_request*) oh;
	size_t rq_len = ntohs(rq->header.length);
	const struct stats_type *st;
	struct stats_dump_cb *cb;
	int type, body_len;
	int err;

	type = ntohs(rq->type);
	for (st = stats; ; st++) {
		if (st >= &stats[ARRAY_SIZE(stats)]) {
			Timestamp now = Timestamp::now();
			click_chatter("%{timestamp} %{element}: received stats request of unknown type %d \n", &now, this, type);

			return -EINVAL;
		} else if (type == st->type) {
			break;
		}
	}

	cb = (struct stats_dump_cb*)xmalloc(sizeof *cb);
	cb->done = false;
	cb->rq = (struct ofp_stats_request*)xmemdup(rq, rq_len);
	cb->sender = rconn_sender;
	cb->xid = xid;
	cb->s = st;
	cb->state = NULL;

	body_len = rq_len - offsetof(struct ofp_stats_request, body);
	if ((int)body_len < (int)cb->s->min_body || (int)body_len > (int)cb->s->max_body) {
		    Timestamp now = Timestamp::now();
		    click_chatter("%{timestamp} %{element}: stats request type %d with bad body length %d \n", &now, this, 
				type, body_len);
		err = -EINVAL;
		goto error;
	}

	if (cb->s->init) {
		err = cb->s->init(rq->body, body_len, &cb->state);
		if (err) {
			Timestamp now = Timestamp::now();
			click_chatter("%{timestamp} %{element}: failed initialization of stats request type %d: %s \n", &now, this, 
					type, strerror(-err));
			goto error;
		}
	}

	// TODO: rate limitation is required
	// if (r->n_txq < TXQ_LIMIT) {
	err = stats_dump(this, cb);

	if (err <= 0) {
		if (err) {
			Timestamp now = Timestamp::now();
			click_chatter("%{timestamp} %{element}: stats_dump error %s \n", &now, this, 
					strerror(-err));

		}
		stats_done(cb);
	}

	return 0;

error:
	free(cb->rq);
	free(cb);
	return err;
}

int Datapath::recv_vendor(struct rconn_remote *rconn_sender, uint32_t xid UNUSED, const void *oh)
{
	DEBUG_TRACE_FUNCTION_CALL;
	const struct ofp_vendor_header *ovh = (const struct ofp_vendor_header*) oh;

	switch (ntohl(ovh->vendor))
	{
		case PRIVATE_VENDOR_ID:

			return private_recv_msg(this, NULL /*rconn_sender*/, oh);

		case OPENFLOW_VENDOR_ID:

			return of_ext_recv_msg(this, NULL /*rconn_sender*/, oh);

		default:
			Timestamp now = Timestamp::now();
			click_chatter("%{timestamp} %{element}: unknown vendor: 0x%x\n", &now, this, ntohl(ovh->vendor));

			dp_send_error_msg(rconn_sender, xid, OFPET_BAD_REQUEST,
					OFPBRC_BAD_VENDOR, oh, ntohs(ovh->header.length));
			return -EINVAL;
	}
	return 0;
}

int Datapath::recv_queue_get_config_request(struct rconn_remote *rconn_sender, uint32_t xid, const void *oh)
{
	DEBUG_TRACE_FUNCTION_CALL;
	struct ofpbuf *buffer;
	struct ofp_queue_get_config_reply *ofq_reply;
	const struct ofp_queue_get_config_request *ofq_request;
	struct click_port *p;
	struct sw_queue *q;
	uint16_t port_no;


	ofq_request = (struct ofp_queue_get_config_request *)oh;
	port_no = ntohs(ofq_request->port);

	if (port_no < OFPP_MAX) {
		/* Find port under query */
		p = dp_lookup_port(port_no);

		/* if the port under query doesn't exist, send an error */
		if (!p ||  (p->port_no != port_no)) {
			dp_send_error_msg( rconn_sender, xid, OFPET_QUEUE_OP_FAILED, OFPQOFC_BAD_PORT,
					oh, ntohs(ofq_request->header.length));
			goto error;
		}
		ofq_reply = (struct ofp_queue_get_config_reply*) make_openflow_reply(sizeof *ofq_reply, OFPT_QUEUE_GET_CONFIG_REPLY,
				rconn_sender, xid, &buffer);
		ofq_reply->port = htons(port_no);
		LIST_FOR_EACH(q, struct sw_queue, node, &p->queue_list) {
			struct ofp_packet_queue * opq = (struct ofp_packet_queue*) ofpbuf_put_zeros(buffer, sizeof *opq);
			fill_queue_desc(buffer, q, opq);
		}
		send_openflow_buffer(buffer, rconn_sender, xid);
	}
	else {
		dp_send_error_msg(rconn_sender, xid, OFPET_QUEUE_OP_FAILED, OFPQOFC_BAD_PORT,
				oh, ntohs(ofq_request->header.length));
	}
error:
	return 0;
}


/*
 * jyyoo: no copying at all
 * now, we use direct calling by ``knowing'' the pointer to the Datapath/RConn objects.
 */
/* 'msg', which is 'length' bytes long, was received from the control path.
 * Apply it to 'chain'. */
int Datapath::fwd_control_input(void *msg, int length, struct rconn_remote *rsender, uint32_t xid)
{
	DEBUG_TRACE_FUNCTION_CALL;

	int ret;
	struct ofp_header *oh;
	size_t min_size;

	/* Check encapsulated length. */
	oh = (struct ofp_header *) msg;
	if (ntohs(oh->length) > length) {
		return -EINVAL;
	}
	assert(oh->version == OFP_VERSION);
#if JD_DEBUG_PARSE_CONTROL_PACKET
	click_chatter("===========> control in <================\n");
	click_chatter("%s\n", ofp_to_string( oh, length, 1 ) );
#endif

	/* Figure out how to handle it. */
	switch (oh->type) {
		case OFPT_FEATURES_REQUEST:
			min_size = sizeof(struct ofp_header);
			ret = recv_features_request( rsender, xid, msg );
			break;
		case OFPT_GET_CONFIG_REQUEST:
			min_size = sizeof(struct ofp_header);
			ret = recv_get_config_request( rsender, xid, msg );
			break;
		case OFPT_SET_CONFIG:
			min_size = sizeof(struct ofp_switch_config);
			ret = recv_set_config( rsender, xid, msg );
			break;
		case OFPT_PACKET_OUT:
			min_size = sizeof(struct ofp_packet_out);
			ret = recv_packet_out( rsender, xid, msg );
			break;
		case OFPT_FLOW_MOD:
			min_size = sizeof(struct ofp_flow_mod);
			ret = recv_flow( rsender, xid, msg );
			break;
		case OFPT_PORT_MOD:
			min_size = sizeof(struct ofp_port_mod);
			ret = recv_port_mod( rsender, xid, msg );
			break;
		case OFPT_STATS_REQUEST:
			min_size = sizeof(struct ofp_stats_request);
			ret = recv_stats_request( rsender, xid, msg );
			break;
		case OFPT_QUEUE_GET_CONFIG_REQUEST:
			min_size = sizeof(struct ofp_header);
			ret = recv_queue_get_config_request( rsender, xid, msg );
			break;
		case OFPT_VENDOR:
			min_size = sizeof(struct ofp_vendor_header);
			ret = recv_vendor( rsender, xid, msg );
			break;
            break;
		default:
		    Timestamp now = Timestamp::now();
		    click_chatter("%{timestamp} %{element}: Unrecognized type: %d \n", &now, this, oh->type);


		    dp_send_error_msg(rsender, xid, OFPET_BAD_REQUEST, OFPBRC_BAD_TYPE, msg, length);
	    return -EINVAL;
	}

	free(msg);

	return ret;
}


void Datapath::dp_send_error_msg(struct rconn_remote *rconn_sender, uint32_t xid, uint16_t type, uint16_t code, const void *data, size_t len)
{
	DEBUG_TRACE_FUNCTION_CALL;
	struct ofpbuf *buffer;
	struct ofp_error_msg *oem;


	oem = (struct ofp_error_msg*)make_openflow_reply(sizeof(*oem)+len, OFPT_ERROR, rconn_sender, xid, &buffer);
	oem->type = htons(type);
	oem->code = htons(code);
	memcpy(oem->data, data, len);
	send_openflow_buffer(buffer, rconn_sender, xid);
}


/***************************************************************************
 * ngkim: get_port_desc 
 * @param: fd socket descriptor
 **************************************************************************/
void Datapath::get_port_description(struct ofp_phy_port* desc, String ifname, int fd)
{
	struct ifreq ifr; 
	struct ethtool_cmd ecmd; 

	memset(&ifr, 0, sizeof ifr);
	strncpy(ifr.ifr_name, ifname.c_str(), sizeof ifr.ifr_name);
	ifr.ifr_data = (caddr_t) &ecmd;

	desc->peer = 0;

	memset(&ecmd, 0, sizeof ecmd);
	ecmd.cmd = ETHTOOL_GSET; 
	if (ioctl(fd, SIOCETHTOOL, &ifr) == 0) {
		if (ecmd.supported & SUPPORTED_10baseT_Half) {
			desc->supported |= OFPPF_10MB_HD;
		}
		if (ecmd.supported & SUPPORTED_10baseT_Full) {
			desc->supported |= OFPPF_10MB_FD;
		}
		if (ecmd.supported & SUPPORTED_100baseT_Half)  {
			desc->supported |= OFPPF_100MB_HD;
		}
		if (ecmd.supported & SUPPORTED_100baseT_Full) {
			desc->supported |= OFPPF_100MB_FD;
		}
		if (ecmd.supported & SUPPORTED_1000baseT_Half) {
			desc->supported |= OFPPF_1GB_HD;
		}
		if (ecmd.supported & SUPPORTED_1000baseT_Full) {
			desc->supported |= OFPPF_1GB_FD;
		}
		if (ecmd.supported & SUPPORTED_10000baseT_Full) {
			desc->supported |= OFPPF_10GB_FD;
		}
		if (ecmd.supported & SUPPORTED_TP) {
			desc->supported |= OFPPF_COPPER;
		}
		if (ecmd.supported & SUPPORTED_FIBRE) {
			desc->supported |= OFPPF_FIBER;
		}
		if (ecmd.supported & SUPPORTED_Autoneg) {
			desc->supported |= OFPPF_AUTONEG;
		}
		/* Set the advertised features */
		if (ecmd.advertising & ADVERTISED_10baseT_Half) {
			desc->advertised |= OFPPF_10MB_HD;
		}
		if (ecmd.advertising & ADVERTISED_10baseT_Full) {
			desc->advertised |= OFPPF_10MB_FD;
		}
		if (ecmd.advertising & ADVERTISED_100baseT_Half) {
			desc->advertised |= OFPPF_100MB_HD;
		}
		if (ecmd.advertising & ADVERTISED_100baseT_Full) {
			desc->advertised |= OFPPF_100MB_FD;
		}
		if (ecmd.advertising & ADVERTISED_1000baseT_Half) {
			desc->advertised |= OFPPF_1GB_HD;
		}
		if (ecmd.advertising & ADVERTISED_1000baseT_Full) {
			desc->advertised |= OFPPF_1GB_FD;
		}
		if (ecmd.advertising & ADVERTISED_10000baseT_Full) {
			desc->advertised |= OFPPF_10GB_FD;
		}
		if (ecmd.advertising & ADVERTISED_TP) {
			desc->advertised |= OFPPF_COPPER;
		}
		if (ecmd.advertising & ADVERTISED_FIBRE) {
			desc->advertised |= OFPPF_FIBER;
		}
		if (ecmd.advertising & ADVERTISED_Autoneg) {
			desc->advertised |= OFPPF_AUTONEG;
		}

		/* Set the current features */
		if (ecmd.speed == SPEED_10) {
			desc->curr = (ecmd.duplex) ? OFPPF_10MB_FD : OFPPF_10MB_HD;
		}
		else if (ecmd.speed == SPEED_100) {
			desc->curr = (ecmd.duplex) ? OFPPF_100MB_FD : OFPPF_100MB_HD;
		}
		else if (ecmd.speed == SPEED_1000) {
			desc->curr = (ecmd.duplex) ? OFPPF_1GB_FD : OFPPF_1GB_HD;
		}
		else if (ecmd.speed == SPEED_10000) {
			desc->curr = OFPPF_10GB_FD;
		}

		if (ecmd.port == PORT_TP) {
			desc->curr |= OFPPF_COPPER;
		}
		else if (ecmd.port == PORT_FIBRE) {
			desc->curr |= OFPPF_FIBER;
		}

		if (ecmd.autoneg) {
			desc->curr |= OFPPF_AUTONEG;
		}

	} else {
		Timestamp now = Timestamp::now();
		click_chatter("%{timestamp} %{element}: ioctl(SIOCETHTOOL) failed: %s \n", &now, this, strerror(errno));
	}

}

/*
 * jyyoo Datapath + Click Interface 
 */
void Datapath::flowtable_timer() 
{
	struct list deleted = LIST_INITIALIZER(&deleted);
	struct sw_flow *f, *n;

	chain_timeout(_chain, &deleted);
	LIST_FOR_EACH_SAFE (f, n, struct sw_flow, node, &deleted) {
		dp_send_flow_end(f, (enum ofp_flow_removed_reason)f->reason);
		list_remove(&f->node);
		flow_free(f);
	}
}


/***************************************************************************
 * ngkim: port_desc
 * @brief for datapath to get port descriptions 
 * @param port_no port number
 **************************************************************************/
void Datapath::port_description(struct ofp_phy_port* desc, uint16_t port_no) 
{
	Router* router = this->router();
	ElementNeighborhoodTracker tracker(router);
	router->visit_upstream(this, port_no, &tracker);
	Vector<Element *> _elements = tracker.elements();

	for (Vector<Element *>::iterator it = _elements.begin(); it != _elements.end(); ++it) {
		// For FromDevice elements
		if (strcmp((*it)->class_name(), "FromDevice") == 0) {
			FromDevice* elm = (FromDevice *)(*it);
			get_port_description(desc, elm->ifname(), elm->fd());
		}
	}
}


DatapathPacketBuffer::DatapathPacketBuffer()
{
	_buffer_idx = 0;
	memset( _buffers, 0, sizeof(_buffers) );
}
DatapathPacketBuffer::~DatapathPacketBuffer()
{
}

uint32_t DatapathPacketBuffer::save_buffer(struct ofpbuf *buffer)
{
	struct dp_packet_buffer *p;
	uint32_t id;

	_buffer_idx = (_buffer_idx + 1) & PKT_BUFFER_MASK;

	p = &_buffers[_buffer_idx];

	if (p->buffer) {
		/* Don't buffer packet if existing entry is less than
		 * OVERWRITE_SECS old. */

		struct timeval tv;
		gettimeofday(&tv, NULL);
		
		if (time_now() < p->timeout) { /* FIXME */
			return (uint32_t)-1;
		} else {
			ofpbuf_delete(p->buffer);
		}
	}
	/* Don't use maximum cookie value since the all-bits-1 id is
	 * special. */
	if (++p->cookie >= (1u << PKT_COOKIE_BITS) - 1)
		p->cookie = 0;
	p->buffer = ofpbuf_clone(buffer);      /* FIXME */
	p->timeout = time_now() + OVERWRITE_SECS; /* FIXME */
	id = _buffer_idx | (p->cookie << PKT_BUFFER_BITS);

	return id;
}

struct ofpbuf *DatapathPacketBuffer::retrieve_buffer(uint32_t id)
{
	struct ofpbuf *buffer = NULL;
	struct dp_packet_buffer *p;

	p = &_buffers[id & PKT_BUFFER_MASK];
	if (p->cookie == id >> PKT_BUFFER_BITS) {
		buffer = p->buffer;
		p->buffer = NULL;
	} else {
		Timestamp now = Timestamp::now();
		click_chatter("%{timestamp} %{element}: cookie mismatch: %x != %x \n", &now, this, 
				id >> PKT_BUFFER_BITS, p->cookie);

	}

	return buffer;
}

void DatapathPacketBuffer::discard_buffer(uint32_t id)
{
	struct dp_packet_buffer *p;

	p = &_buffers[id & PKT_BUFFER_MASK];
	if (p->cookie == id >> PKT_BUFFER_BITS) {
		ofpbuf_delete(p->buffer);
		p->buffer = NULL;
	}
}



CLICK_ENDDECLS
EXPORT_ELEMENT(Datapath)
