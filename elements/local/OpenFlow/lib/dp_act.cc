/* Copyright (c) 2008, 2009 The Board of Trustees of The Leland Stanford
 * Junior University
 * 
 * We are making the OpenFlow specification and associated documentation
 * (Software) available for public use and benefit with the expectation
 * that others will use, modify and enhance the Software and contribute
 * those enhancements back to the community. However, since we would
 * like to make the Software available for broadest use, with as few
 * restrictions as possible permission is hereby granted, free of
 * charge, to any person obtaining a copy of this Software to deal in
 * the Software under the copyrights without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 * 
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * 
 * The name and trademarks of copyright holder(s) may NOT be used in
 * advertising or publicity pertaining to the Software or any
 * derivatives without specific, written prior permission.
 */

/* Functions for executing OpenFlow actions. */
#include "../include/config.h"
#include <arpa/inet.h>
#include "csum.hh"
#include "packets.hh"
#include "dp_act.hh"
#include "ofp-print.hh"
#include "../include/openflow/nicira-ext.hh"
CLICK_DECLS
static uint16_t
validate_output(class Datapath *dp UNUSED, const struct sw_flow_key *key, 
        const struct ofp_action_header *ah) 
{
    struct ofp_action_output *oa = (struct ofp_action_output *)ah;

    /* To prevent loops, make sure there's no action to send to the
     * OFP_TABLE virtual port.
     */
    if (oa->port == htons(OFPP_NONE) || 
            (!(key->wildcards & OFPFW_IN_PORT) 
                    && oa->port == key->flow.in_port)) {
        return OFPBAC_BAD_OUT_PORT;
    }
    return ACT_VALIDATION_OK;
}

static uint16_t
validate_queue(class Datapath *dp UNUSED, const struct sw_flow_key *key UNUSED,
               const struct ofp_action_header *ah)
{
    struct ofp_action_enqueue *ea = (struct ofp_action_enqueue *)ah;

	/* Only physical ports may have queues. */
    if (ntohs(ea->port) > OFPP_MAX && ntohs(ea->port) != OFPP_IN_PORT) {
        return OFPBAC_BAD_OUT_PORT;
    }
    return ACT_VALIDATION_OK;
}

static void
do_output(class Datapath *dp, struct ofpbuf *buffer, int in_port,
          size_t max_len, int out_port, uint32_t queue_id,
          bool ignore_no_fwd)
{
	DEBUG_TRACE_FUNCTION_CALL;
	if (out_port != OFPP_CONTROLLER) {
		dp->dp_output_port( buffer, in_port, out_port, queue_id, ignore_no_fwd);
	} else {
		dp->dp_output_control( buffer, in_port, max_len, OFPR_ACTION);
	}
}

/* Modify vlan tag control information (TCI).  Only sets the TCI bits
 * indicated by 'mask'.  If no vlan tag is present, one is added.
 */
static void
modify_vlan_tci(struct ofpbuf *buffer, struct sw_flow_key *key,
        uint16_t tci, uint16_t mask)
{
    struct vlan_eth_header *veh;

    if (key->flow.dl_vlan != htons(OFP_VLAN_NONE)) {
        /* Modify vlan id, but maintain other TCI values */
        veh = (struct vlan_eth_header*) buffer->l2;
        veh->veth_tci &= ~htons(mask);
        veh->veth_tci |= htons(tci);
    } else {
        /* Insert New vlan id. */
        struct eth_header *eh = (struct eth_header*)buffer->l2;
        struct vlan_eth_header tmp;
        memcpy(tmp.veth_dst, eh->eth_dst, ETH_ADDR_LEN);
        memcpy(tmp.veth_src, eh->eth_src, ETH_ADDR_LEN);
        tmp.veth_type = htons(ETH_TYPE_VLAN);
        tmp.veth_tci = htons(tci);
        tmp.veth_next_type = eh->eth_type;

        veh = (struct vlan_eth_header*)ofpbuf_push_uninit(buffer, VLAN_HEADER_LEN);
        memcpy(veh, &tmp, sizeof tmp);
        buffer->l2 = (char*)buffer->l2 - VLAN_HEADER_LEN;
    }

    key->flow.dl_vlan = veh->veth_tci & htons(VLAN_VID_MASK);
    key->flow.dl_vlan_pcp = (uint8_t)((ntohs(veh->veth_tci) >> VLAN_PCP_SHIFT)
                                      & VLAN_PCP_BITMASK);
}


/* Remove an existing vlan header if it exists. */
static void
vlan_pull_tag(struct ofpbuf *buffer)
{
    struct vlan_eth_header *veh = (struct vlan_eth_header*) buffer->l2;

    if (veh->veth_type == htons(ETH_TYPE_VLAN)) {
        struct eth_header tmp;

        memcpy(tmp.eth_dst, veh->veth_dst, ETH_ADDR_LEN);
        memcpy(tmp.eth_src, veh->veth_src, ETH_ADDR_LEN);
        tmp.eth_type = veh->veth_next_type;

        buffer->size -= VLAN_HEADER_LEN;
        buffer->data = (char*)buffer->data + VLAN_HEADER_LEN;
        buffer->l2 = (char*)buffer->l2 + VLAN_HEADER_LEN;
        memcpy(buffer->data, &tmp, sizeof tmp);
    }
}

static uint32_t
set_vlan_vid(class Datapath *dp UNUSED, struct ofpbuf *buffer, struct sw_flow_key *key, 
        const struct ofp_action_header *ah, size_t actions_len UNUSED )
{
    struct ofp_action_vlan_vid *va = (struct ofp_action_vlan_vid *)ah;
    uint16_t tci = ntohs(va->vlan_vid);

    modify_vlan_tci(buffer, key, tci, VLAN_VID_MASK);
    return 0;
}

static uint32_t
set_vlan_pcp(class Datapath *dp UNUSED, struct ofpbuf *buffer, struct sw_flow_key *key, 
        const struct ofp_action_header *ah, size_t actions_len UNUSED )
{
    struct ofp_action_vlan_pcp *va = (struct ofp_action_vlan_pcp *)ah;
    uint16_t tci = (uint16_t)va->vlan_pcp << 13;

    modify_vlan_tci(buffer, key, tci, VLAN_PCP_MASK);
    return 0;
}

static uint32_t
strip_vlan(class Datapath *dp UNUSED, struct ofpbuf *buffer, struct sw_flow_key *key, 
        const struct ofp_action_header *ah UNUSED, size_t actions_len UNUSED )
{
    vlan_pull_tag(buffer);
    key->flow.dl_vlan = htons(OFP_VLAN_NONE);
    return 0;
}

static uint32_t
set_dl_addr(class Datapath *dp UNUSED, struct ofpbuf *buffer, struct sw_flow_key *key UNUSED, 
        const struct ofp_action_header *ah, size_t actions_len UNUSED )
{
    struct ofp_action_dl_addr *da = (struct ofp_action_dl_addr *)ah;
    struct eth_header *eh = (struct eth_header*) buffer->l2;

    if (da->type == htons(OFPAT_SET_DL_SRC)) {
        memcpy(eh->eth_src, da->dl_addr, sizeof eh->eth_src);
    } else {
        memcpy(eh->eth_dst, da->dl_addr, sizeof eh->eth_dst);
    }
    return 0;
}

static uint32_t
set_nw_addr(class Datapath *dp UNUSED, struct ofpbuf *buffer, struct sw_flow_key *key, 
        const struct ofp_action_header *ah, size_t actions_len UNUSED )
{
    struct ofp_action_nw_addr *na = (struct ofp_action_nw_addr *)ah;
    uint16_t eth_proto = ntohs(key->flow.dl_type);

    if (eth_proto == ETH_TYPE_IP) {
        struct ip_header *nh = (struct ip_header*) buffer->l3;
        uint8_t nw_proto = key->flow.nw_proto;
        uint32_t New, *field;

        New = na->nw_addr;
        field = na->type == htons(OFPAT_SET_NW_SRC) ? &nh->ip_src : &nh->ip_dst;
        if (nw_proto == IP_TYPE_TCP) {
            struct tcp_header *th = (struct tcp_header*)buffer->l4;
            th->tcp_csum = recalc_csum32(th->tcp_csum, *field, New);
        } else if (nw_proto == IP_TYPE_UDP) {
            struct udp_header *th = (struct udp_header*)buffer->l4;
            if (th->udp_csum) {
                th->udp_csum = recalc_csum32(th->udp_csum, *field, New);
                if (!th->udp_csum) {
                    th->udp_csum = 0xffff;
                }
            }
        }
        nh->ip_csum = recalc_csum32(nh->ip_csum, *field, New);
        *field = New;
    }
    return 0;
}

static uint32_t
set_nw_tos(class Datapath *dp UNUSED, struct ofpbuf *buffer, struct sw_flow_key *key, 
           const struct ofp_action_header *ah, size_t actions_len UNUSED )
{
    struct ofp_action_nw_tos *nt = (struct ofp_action_nw_tos *)ah;
    uint16_t eth_proto = ntohs(key->flow.dl_type);

   if (eth_proto == ETH_TYPE_IP) {
       struct ip_header *nh = (struct ip_header*) buffer->l3;
       uint8_t New, *field;

       /* JeanII : Set only 6 bits, don't clobber ECN */
       New = (nt->nw_tos & 0xFC) | (nh->ip_tos & 0x03);

       /* Get address of field */
       field = &nh->ip_tos;

        /* jklee : ip tos field is not included in TCP pseudo header.
         * Need magic as update_csum() don't work with 8 bits. */
       nh->ip_csum = recalc_csum32(nh->ip_csum, htons((uint16_t)*field),
                                   htons((uint16_t)New));

       /* Change the IP ToS bits */
       *field = New;
    }
    return 0;
}

static uint32_t
set_tp_port(class Datapath *dp UNUSED, struct ofpbuf *buffer, struct sw_flow_key *key, 
        const struct ofp_action_header *ah, size_t actions_len UNUSED )
{
    struct ofp_action_tp_port *ta = (struct ofp_action_tp_port *)ah;
    uint16_t eth_proto = ntohs(key->flow.dl_type);

    if (eth_proto == ETH_TYPE_IP) {
        uint8_t nw_proto = key->flow.nw_proto;
        uint16_t New, *field;

        New = ta->tp_port;
        if (nw_proto == IP_TYPE_TCP) {
            struct tcp_header *th = (struct tcp_header*) buffer->l4;
            field = ta->type == htons(OFPAT_SET_TP_SRC) ? &th->tcp_src : &th->tcp_dst;
            th->tcp_csum = recalc_csum16(th->tcp_csum, *field, New);
            *field = New;
        } else if (nw_proto == IP_TYPE_UDP) {
            struct udp_header *th = (struct udp_header*) buffer->l4;
            field = ta->type == htons(OFPAT_SET_TP_SRC) ? &th->udp_src : &th->udp_dst;
            th->udp_csum = recalc_csum16(th->udp_csum, *field, New);
            *field = New;
        }
    }
    return 0;
}

/** 
 * jyyoo: vxs_dxt is the function that performs dxt-compression for the given flow 
 * vxs stands for VisualXSwitch
 */
static uint32_t vxs_in_network_procesing( Datapath *dp, struct ofpbuf *buffer, struct sw_flow_key *key, 
        const struct ofp_action_header *ah, size_t actions_len )
{
//	struct ofp_action_vxs_dxt *ta = (struct ofp_action_vxs_dxt *)ah;
	
	/* validate the flow with given key as if it is IP and etc. */
	uint16_t eth_proto = ntohs(key->flow.dl_type);
	uint8_t nw_proto = key->flow.nw_proto;

	if (eth_proto == ETH_TYPE_IP && nw_proto == IP_TYPE_UDP ) {
		if( dp->vxsManager.recvPacket( buffer, key, ah, actions_len, VXS_MEDIA_TYPE_RAW ) == 0 ) {
			/* if we succeeded, we need to destroy this packet 
			 * by returnning 1, it automatically frees this buffer
			 */
			return 1;
		} else {
			click_chatter("JYD VXS fails!!!\n");
		}
	}
 	return 0;
}

struct openflow_action {
	size_t min_size;
	size_t max_size;
	uint16_t (*validate)(class Datapath *dp, 
			const struct sw_flow_key *key,
			const struct ofp_action_header *ah
			);
	uint32_t (*execute)(class Datapath *dp,
			struct ofpbuf *buffer,
			struct sw_flow_key *key, 
			const struct ofp_action_header *ah, 
			size_t actions_len);
};

static struct openflow_action of_actions[OFPAT_VXS_COPY_BRANCH+1] = {
};

void init_of_actions()
{
	of_actions[OFPAT_OUTPUT].min_size =        sizeof(struct ofp_action_output);
	of_actions[OFPAT_OUTPUT].max_size =        sizeof(struct ofp_action_output);
	of_actions[OFPAT_OUTPUT].validate =        validate_output;
	of_actions[OFPAT_OUTPUT].execute =         NULL;                   /* This is optimized into execute_actions */
	of_actions[OFPAT_ENQUEUE].min_size =        sizeof(struct ofp_action_enqueue);
	of_actions[OFPAT_ENQUEUE].max_size =        sizeof(struct ofp_action_enqueue);
	of_actions[OFPAT_ENQUEUE].validate =        validate_queue;
	of_actions[OFPAT_ENQUEUE].execute =         NULL;         /* This is optimized into execute_actions */
	of_actions[OFPAT_SET_VLAN_VID].min_size =        sizeof(struct ofp_action_vlan_vid);
	of_actions[OFPAT_SET_VLAN_VID].max_size =        sizeof(struct ofp_action_vlan_vid);
	of_actions[OFPAT_SET_VLAN_VID].validate =        NULL;
	of_actions[OFPAT_SET_VLAN_VID].execute =         set_vlan_vid;
	of_actions[OFPAT_SET_VLAN_PCP].min_size =       sizeof(struct ofp_action_vlan_pcp);
	of_actions[OFPAT_SET_VLAN_PCP].max_size =       sizeof(struct ofp_action_vlan_pcp);
	of_actions[OFPAT_SET_VLAN_PCP].validate =       NULL;
	of_actions[OFPAT_SET_VLAN_PCP].execute =        set_vlan_pcp;
	of_actions[OFPAT_STRIP_VLAN].min_size =       sizeof(struct ofp_action_header);
	of_actions[OFPAT_STRIP_VLAN].max_size =       sizeof(struct ofp_action_header);
	of_actions[OFPAT_STRIP_VLAN].validate =       NULL;
	of_actions[OFPAT_STRIP_VLAN].execute =        strip_vlan;
	of_actions[OFPAT_SET_DL_SRC].min_size =       sizeof(struct ofp_action_dl_addr);
	of_actions[OFPAT_SET_DL_SRC].max_size =       sizeof(struct ofp_action_dl_addr);
	of_actions[OFPAT_SET_DL_SRC].validate =       NULL;
	of_actions[OFPAT_SET_DL_SRC].execute =        set_dl_addr;
	of_actions[OFPAT_SET_DL_DST].min_size =       sizeof(struct ofp_action_dl_addr);
	of_actions[OFPAT_SET_DL_DST].max_size =       sizeof(struct ofp_action_dl_addr);
	of_actions[OFPAT_SET_DL_DST].validate =       NULL;
	of_actions[OFPAT_SET_DL_DST].execute =        set_dl_addr;
	of_actions[OFPAT_SET_NW_SRC].min_size =       sizeof(struct ofp_action_nw_addr);
	of_actions[OFPAT_SET_NW_SRC].max_size =       sizeof(struct ofp_action_nw_addr);
	of_actions[OFPAT_SET_NW_SRC].validate =       NULL;
	of_actions[OFPAT_SET_NW_SRC].execute =        set_nw_addr;
	of_actions[OFPAT_SET_NW_DST].min_size =       sizeof(struct ofp_action_nw_addr);
	of_actions[OFPAT_SET_NW_DST].max_size =       sizeof(struct ofp_action_nw_addr);
	of_actions[OFPAT_SET_NW_DST].validate =       NULL;
	of_actions[OFPAT_SET_NW_DST].execute =        set_nw_addr;
	of_actions[OFPAT_SET_NW_TOS].min_size =       sizeof(struct ofp_action_nw_tos);
	of_actions[OFPAT_SET_NW_TOS].max_size =       sizeof(struct ofp_action_nw_tos);
	of_actions[OFPAT_SET_NW_TOS].validate =       NULL;
	of_actions[OFPAT_SET_NW_TOS].execute =        set_nw_tos;
	of_actions[OFPAT_SET_TP_SRC].min_size =       sizeof(struct ofp_action_tp_port);
	of_actions[OFPAT_SET_TP_SRC].max_size =       sizeof(struct ofp_action_tp_port);
	of_actions[OFPAT_SET_TP_SRC].validate =       NULL;
	of_actions[OFPAT_SET_TP_SRC].execute =        set_tp_port;
	of_actions[OFPAT_SET_TP_DST].min_size =       sizeof(struct ofp_action_tp_port);
	of_actions[OFPAT_SET_TP_DST].max_size =       sizeof(struct ofp_action_tp_port);
	of_actions[OFPAT_SET_TP_DST].validate =       NULL;
	of_actions[OFPAT_SET_TP_DST].execute =        set_tp_port;
	/* in-network processing */
	of_actions[OFPAT_VXS_DXTComp].min_size = 	sizeof(struct ofp_action_vxs_dxt);
	of_actions[OFPAT_VXS_DXTComp].max_size = 	sizeof(struct ofp_action_vxs_dxt);
	of_actions[OFPAT_VXS_DXTComp].validate = 	NULL;
	of_actions[OFPAT_VXS_DXTComp].execute = 	vxs_in_network_procesing;
	of_actions[OFPAT_VXS_DXTDecomp].min_size = 	sizeof(struct ofp_action_vxs_dxt_decompress);
	of_actions[OFPAT_VXS_DXTDecomp].max_size = 	sizeof(struct ofp_action_vxs_dxt_decompress);
	of_actions[OFPAT_VXS_DXTDecomp].validate = 	NULL;
	of_actions[OFPAT_VXS_DXTDecomp].execute = 	vxs_in_network_procesing;
	of_actions[OFPAT_VXS_FrameResize].min_size = 	sizeof(struct ofp_action_vxs_frame_resize);
	of_actions[OFPAT_VXS_FrameResize].max_size = 	sizeof(struct ofp_action_vxs_frame_resize);
	of_actions[OFPAT_VXS_FrameResize].validate = 	NULL;
	of_actions[OFPAT_VXS_FrameResize].execute = 	vxs_in_network_procesing;
	of_actions[OFPAT_VXS_YUV2RGB].min_size = 	sizeof(struct ofp_action_vxs_yuv2rgb);
	of_actions[OFPAT_VXS_YUV2RGB].max_size = 	sizeof(struct ofp_action_vxs_yuv2rgb);
	of_actions[OFPAT_VXS_YUV2RGB].validate = 	NULL;
	of_actions[OFPAT_VXS_YUV2RGB].execute = 	vxs_in_network_procesing;
	of_actions[OFPAT_VXS_COPY_BRANCH].min_size = 	sizeof(struct ofp_action_vxs_copy_branch);
	of_actions[OFPAT_VXS_COPY_BRANCH].max_size = 	sizeof(struct ofp_action_vxs_copy_branch);
	of_actions[OFPAT_VXS_COPY_BRANCH].validate = 	NULL;
	of_actions[OFPAT_VXS_COPY_BRANCH].execute = 	vxs_in_network_procesing;

	/* OFPAT_VENDOR is not here; since it would blow up the array size. */

}

/* Validate built-in OpenFlow actions.  Either returns ACT_VALIDATION_OK
 * or an OFPET_BAD_ACTION error code. */
static uint16_t 
validate_ofpat(class Datapath *dp, const struct sw_flow_key *key, 
        const struct ofp_action_header *ah, uint16_t type, uint16_t len)
{
    uint16_t ret = ACT_VALIDATION_OK;
    const struct openflow_action *act = &of_actions[type];

    if ((len < act->min_size) || (len > act->max_size)) {
        return OFPBAC_BAD_LEN;
    }

    if (act->validate) {
        ret = act->validate(dp, key, ah);
    }

    return ret;
}

/* Validate vendor-defined actions.  Either returns ACT_VALIDATION_OK
 * or an OFPET_BAD_ACTION error code. */
static uint16_t 
validate_vendor(class Datapath *dp UNUSED, const struct sw_flow_key *key UNUSED, 
        const struct ofp_action_header *ah, uint16_t len)
{
    struct ofp_action_vendor_header *avh;
    int ret = ACT_VALIDATION_OK;

    if (len < sizeof(struct ofp_action_vendor_header)) {
        return OFPBAC_BAD_LEN;
    }

    avh = (struct ofp_action_vendor_header *)ah;

    switch(ntohl(avh->vendor)) {
    default:
        return OFPBAC_BAD_VENDOR;
    }

    return ret;
}

/* Validates a list of actions.  If a problem is found, a code for the
 * OFPET_BAD_ACTION error type is returned.  If the action list validates, 
 * ACT_VALIDATION_OK is returned. */
uint16_t 
validate_actions(class Datapath *dp, const struct sw_flow_key *key,
        const struct ofp_action_header *actions, size_t actions_len)
{
    uint8_t *p = (uint8_t *)actions;
    int err;

    while (actions_len >= sizeof(struct ofp_action_header)) {
        struct ofp_action_header *ah = (struct ofp_action_header *)p;
        size_t len = ntohs(ah->len);
        uint16_t type;

        /* Make there's enough remaining data for the specified length
         * and that the action length is a multiple of 64 bits. */
        if ((actions_len < len) || (len % 8) != 0) {
            return OFPBAC_BAD_LEN;
        }

        type = ntohs(ah->type);
        if (type < ARRAY_SIZE(of_actions)) {
            err = validate_ofpat(dp, key, ah, type, len);
            if (err != ACT_VALIDATION_OK) {
                return err;
            }
        } else if (type == OFPAT_VENDOR) {
            err = validate_vendor(dp, key, ah, len);
            if (err != ACT_VALIDATION_OK) {
                return err;
            }
        } else {
            return OFPBAC_BAD_TYPE;
        }

        p += len;
        actions_len -= len;
    }

    /* Check if there's any trailing garbage. */
    if (actions_len != 0) {
        return OFPBAC_BAD_LEN;
    }

    return ACT_VALIDATION_OK;
}

/* Execute a built-in OpenFlow action against 'buffer'. */
static uint32_t
execute_ofpat(class Datapath *dp, struct ofpbuf *buffer, struct sw_flow_key *key, 
        const struct ofp_action_header *ah, uint16_t type, size_t actions_len)
{
    const struct openflow_action *act = &of_actions[type];

    if (act->execute) {
        return act->execute(dp, buffer, key, ah, actions_len);
    }
    return 0;
}

/* Execute a vendor-defined action against 'buffer'. */
static void
execute_vendor(struct ofpbuf *buffer UNUSED, const struct sw_flow_key *key UNUSED, 
        const struct ofp_action_header *ah)
{
    struct ofp_action_vendor_header *avh 
            = (struct ofp_action_vendor_header *)ah;

    switch(ntohl(avh->vendor)) {
    default:
        /* This should not be possible due to prior validation. */
        printf("attempt to execute action with unknown vendor: %#x\n", 
                ntohl(avh->vendor));
        break;
    }
}

/* Execute a list of actions against 'buffer'. */
void execute_actions(class Datapath *dp, struct ofpbuf *buffer,
		struct sw_flow_key *key,
		const struct ofp_action_header *actions, size_t actions_len,
		int ignore_no_fwd)
{

	DEBUG_TRACE_FUNCTION_CALL;
	/* Every output action needs a separate clone of 'buffer', but the common
	 * case is just a single output action, so that doing a clone and then
	 * freeing the original buffer is wasteful.  So the following code is
	 * slightly obscure just to avoid that. */
	int prev_port;
	uint32_t prev_queue;
	size_t max_len = UINT16_MAX;
	uint16_t in_port = ntohs(key->flow.in_port);
	uint8_t *p = (uint8_t *)actions;

	prev_port = -1;
	prev_queue = 0;

	/* The action list was already validated, so we can be a bit looser
	 * in our sanity-checking. */
	while (actions_len > 0) {
		struct ofp_action_header *ah = (struct ofp_action_header *)p;
		size_t len = htons(ah->len);

		if (prev_port != -1) {
			do_output(dp, ofpbuf_clone(buffer), in_port, max_len,
					prev_port, prev_queue, ignore_no_fwd);
			prev_port = -1;
		}

		if (ah->type == htons(OFPAT_OUTPUT)) {
			struct ofp_action_output *oa = (struct ofp_action_output *)p;
			prev_port = ntohs(oa->port);
			prev_queue = 0; /* using the default best-effort queue */
			max_len = ntohs(oa->max_len);
		} else if (ah->type == htons(OFPAT_ENQUEUE)) {
			struct ofp_action_enqueue *ea = (struct ofp_action_enqueue *)p;
			prev_port = ntohs(ea->port);
			prev_queue = ntohl(ea->queue_id);
			max_len = 0; /* we will not send to the controller anyways - useless */
		} else {
			uint16_t type = ntohs(ah->type);

			if (type < ARRAY_SIZE(of_actions)) {
				/* if execute_ofpat returns non-zero, discard this packet immediately */
				if( execute_ofpat(dp, buffer, key, ah, type, actions_len) != 0 ) {
					ofpbuf_delete(buffer);
					return;
				}
			} else if (type == OFPAT_VENDOR) {
				execute_vendor(buffer, key, ah);
			}
		}

		p += len;
		actions_len -= len;
	}
	if (prev_port != -1) {
		do_output(dp, buffer, in_port, max_len, prev_port, prev_queue, ignore_no_fwd);
	} else {
		ofpbuf_delete(buffer);
	}
}
CLICK_ENDDECLS
ELEMENT_PROVIDES(Of_DpAct)
