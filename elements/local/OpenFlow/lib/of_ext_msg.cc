/* Copyright (c) 2009 The Board of Trustees of The Leland Stanford
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

#include <click/config.h>
#include <errno.h>
#include <arpa/inet.h>
#include "../include/openflow/openflow-ext.hh"
#include "of_ext_msg.hh"
#include "../datapath.hh"

#define THIS_MODULE VLM_experimental
#include "vlog.hh"
CLICK_DECLS

#if 0
static int
new_queue(struct click_port * port, struct sw_queue * queue,
          uint32_t queue_id, uint16_t class_id,
          struct ofp_queue_prop_min_rate * mr)
{
    memset(queue, '\0', sizeof *queue);
    queue->port = port;
    queue->queue_id = queue_id;
    /* class_id is the internal mapping to class. It is the offset
     * in the array of queues for each port. Note that class_id is
     * local to port, so we don't have any conflict.
     * tc uses 16-bit class_id, so we cannot use the queue_id
     * field */
    queue->class_id = class_id;
    queue->property = ntohs(mr->prop_header.property);
    queue->min_rate = ntohs(mr->rate);

    list_push_back(&port->queue_list, &queue->node);

    return 0;
}

static int
port_add_queue(struct click_port *p, uint32_t queue_id,
               struct ofp_queue_prop_min_rate * mr) 
{
    int queue_no;
    for (queue_no = 1; queue_no < p->num_queues; queue_no++) {
        struct sw_queue *q = &p->queues[queue_no];
        if (!q->port) {
            return new_queue(p,q,queue_id,queue_no,mr);
        }
    }
    return EXFULL;
}

static int
port_delete_queue(struct click_port *p UNUSED, struct sw_queue *q) 
{
    list_remove(&q->node);
    memset(q,'\0', sizeof *q);
    return 0;
}
#endif

static void recv_of_exp_queue_delete(class Datapath *dp UNUSED,
                         const struct rconn_sender *sender UNUSED,
                         const void *oh UNUSED)
{
	/* not supported */
	fprintf(stderr, "recv_of_exp_queue_delete not supported\n");
}

/** Modifies/adds a queue. It first searches if a queue with
 * id exists for this port. If yes it modifies it, otherwise adds
 * a new configuration.
 *
 * @param dp the related datapath
 * @param sender request source
 * @param oh the openflow message for queue mod.
 */
static void
recv_of_exp_queue_modify(class Datapath *dp UNUSED,
                         const struct rconn_sender *sender UNUSED,
                         const void *oh UNUSED)
{
	/* not supported */
	fprintf(stderr, "recv_of_exp_queue_modify not supported\n");
}
/**
 * Parses a set dp_desc message and uses it to set
 *  the dp_desc string in dp
 */
static void
recv_of_set_dp_desc(class Datapath *dp,
                         const struct rconn_sender *sender UNUSED,
                         const struct ofp_extension_header * exth)
{
    struct openflow_ext_set_dp_desc * set_dp_desc = (struct openflow_ext_set_dp_desc * )  exth;
    char* dpdesc = dp->get_dp_desc();
    strncpy(dpdesc, set_dp_desc->dp_desc, DESC_STR_LEN);
    dpdesc[DESC_STR_LEN-1] = 0;        // force null for safety
}

/**
 * Receives an experimental message and pass it
 * to the appropriate handler
 */
int of_ext_recv_msg(class Datapath *dp, const struct rconn_sender *sender,
        const void *oh)
{
    const struct ofp_extension_header  *ofexth = (const struct ofp_extension_header*) oh;

    switch (ntohl(ofexth->subtype)) {
    case OFP_EXT_QUEUE_MODIFY: {
        recv_of_exp_queue_modify(dp,sender,oh);
        return 0;
    }
    case OFP_EXT_QUEUE_DELETE: {
        recv_of_exp_queue_delete(dp,sender,oh);
        return 0;
    }
    case OFP_EXT_SET_DESC:
        recv_of_set_dp_desc(dp,sender,ofexth);
        return 0;
    default:
        VLOG_ERR("Received unknown command of type %d",
                 ntohl(ofexth->subtype));
        return -EINVAL;
    }

    return -EINVAL;
}
CLICK_ENDDECLS
ELEMENT_PROVIDES(Of_OfExtMsg)
