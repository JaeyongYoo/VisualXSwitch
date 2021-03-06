/*
 * udprewriter.{cc,hh} -- rewrites packet source and destination
 * Max Poletto, Eddie Kohler
 *
 * Copyright (c) 2000 Massachusetts Institute of Technology
 * Copyright (c) 2008-2010 Meraki, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, subject to the conditions
 * listed in the Click LICENSE file. These conditions include: you must
 * preserve this copyright notice, and you cannot mention the copyright
 * holders in advertising related to the Software without their permission.
 * The Software is provided WITHOUT ANY WARRANTY, EXPRESS OR IMPLIED. This
 * notice is a summary of the Click LICENSE file; the license in that file is
 * legally binding.
 */

#include <click/config.h>
#include "udprewriter.hh"
#include <click/args.hh>
#include <click/straccum.hh>
#include <click/error.hh>
#include <click/timer.hh>
CLICK_DECLS

UDPRewriter::UDPRewriter()
{
}

UDPRewriter::~UDPRewriter()
{
}

void *
UDPRewriter::cast(const char *n)
{
    if (strcmp(n, "IPRewriterBase") == 0)
	return (IPRewriterBase *)this;
    else if (strcmp(n, "UDPRewriter") == 0)
	return (UDPRewriter *)this;
    else
	return 0;
}

int
UDPRewriter::configure(Vector<String> &conf, ErrorHandler *errh)
{
    bool dst_anno = true, has_reply_anno = false;
    int reply_anno;
    _timeouts[0] = 300;		// 5 minutes

    if (Args(this, errh).bind(conf)
	.read("DST_ANNO", dst_anno)
	.read("REPLY_ANNO", AnnoArg(1), reply_anno).read_status(has_reply_anno)
	.read("UDP_TIMEOUT", SecondsArg(), _timeouts[0])
	.read("UDP_GUARANTEE", SecondsArg(), _timeouts[1])
	.consume() < 0)
	return -1;

    _annos = (dst_anno ? 1 : 0) + (has_reply_anno ? 2 + (reply_anno << 2) : 0);
    return IPRewriterBase::configure(conf, errh);
}

IPRewriterEntry *
UDPRewriter::add_flow(int ip_p, const IPFlowID &flowid,
		      const IPFlowID &rewritten_flowid, int input)
{
    void *data;
    if (!(data = _allocator.allocate()))
	return 0;

    IPRewriterFlow *flow = new(data) IPRewriterFlow
	(flowid, _input_specs[input].foutput,
	 rewritten_flowid, _input_specs[input].routput, ip_p,
	 !!_timeouts[1], click_jiffies() + relevant_timeout(_timeouts),
	 this, input);

    return store_flow(flow, input, _map);
}

void
UDPRewriter::push(int port, Packet *p_in)
{
    WritablePacket *p = p_in->uniqueify();
    if (!p)
	return;
    click_ip *iph = p->ip_header();

    // handle non-TCP and non-first fragments
    int ip_p = iph->ip_p;
    if ((ip_p != IP_PROTO_TCP && ip_p != IP_PROTO_UDP && ip_p != IP_PROTO_DCCP)
	|| !IP_FIRSTFRAG(iph)
	|| p->transport_length() < 8) {
	const IPRewriterInput &is = _input_specs[port];
	if (is.kind == IPRewriterInput::i_nochange)
	    output(is.foutput).push(p);
	else
	    p->kill();
	return;
    }

    IPFlowID flowid(p);
    IPRewriterEntry *m = _map.get(flowid);

    if (!m) {			// create new mapping
	IPRewriterInput &is = _input_specs.at_u(port);
	IPFlowID rewritten_flowid = IPFlowID::uninitialized_t();
	int result = is.rewrite_flowid(flowid, rewritten_flowid, p);
	if (result == rw_addmap)
	    m = UDPRewriter::add_flow(ip_p, flowid, rewritten_flowid, port);
	if (!m) {
	    checked_output_push(result, p);
	    return;
	} else if (_annos & 2)
	    m->flow()->set_reply_anno(p->anno_u8(_annos >> 2));
    }

    IPRewriterFlow *mf = static_cast<IPRewriterFlow *>(m->flow());
    mf->apply(p, m->direction(), _annos);
    mf->change_expiry_by_timeout(_heap, click_jiffies(), _timeouts);

    output(m->output()).push(p);
}


String
UDPRewriter::dump_mappings_handler(Element *e, void *)
{
    UDPRewriter *rw = (UDPRewriter *)e;
    click_jiffies_t now = click_jiffies();
    StringAccum sa;
    for (Map::iterator iter = rw->_map.begin(); iter.live(); ++iter) {
	iter->flow()->unparse(sa, iter->direction(), now);
	sa << '\n';
    }
    return sa.take_string();
}

void
UDPRewriter::add_handlers()
{
    add_read_handler("mappings", dump_mappings_handler, 0);
    add_rewriter_handlers(true);
}

CLICK_ENDDECLS
ELEMENT_REQUIRES(IPRewriterBase)
EXPORT_ELEMENT(UDPRewriter)
