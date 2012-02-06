/* Copyright (c) 2008 The Board of Trustees of The Leland Stanford
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

#ifndef QUEUE_H
#define QUEUE_H 1
#include <click/config.h>

CLICK_DECLS
class Of_Queue {
	int dummy;
};

/* Packet queue. */
struct ofp_queue {
    int n;                      /* Number of queued packets. */
    struct ofpbuf *head;        /* First queued packet, null if n == 0. */
    struct ofpbuf *tail;        /* Last queued packet, null if n == 0. */
};

void queue_init(struct ofp_queue *);
void queue_destroy(struct ofp_queue *);
void queue_clear(struct ofp_queue *);
void queue_advance_head(struct ofp_queue *, struct ofpbuf *next);
void queue_push_tail(struct ofp_queue *, struct ofpbuf *);
struct ofpbuf *queue_pop_head(struct ofp_queue *);


CLICK_ENDDECLS

#endif /* queue.h */
