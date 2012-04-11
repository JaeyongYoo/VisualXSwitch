#ifndef __VXS_INNETWORK_PRE_BATCHER__
#define __VXS_INNETWORK_PRE_BATCHER__

#include <click/config.h>
#include <click/element.hh>
#include "../OpenFlow/lib/switch-flow.hh"
#include "VxSInNetworkTaskQueue.hh"
#include "VxSInNetworkSegment.hh"
#include <list> /* for std::list */

CLICK_DECLS

class VxSInNetworkTaskQueue;
class Datapath;

#define VXS_TIMER_INTERVAL 10 /* 10 milli second */

/** 
 * Desc: The type of media that VxS supports for in-network processing
 * 
 * XXX: If you add a new media type here, you also have to edit the 
 *      media_type_name global variable which is defined in "VxSInNetworkPreBatcher.cc"
 */
enum vxs_media_type {
	VXS_MEDIA_TYPE_RAW,
	VXS_MEDIA_TYPE_DXT,
	VXS_MEDIA_TYPE_MPEG2,
	VXS_MEDIA_TYPE_H264,
	VXS_END_OF_TYPE
};

extern const char *media_type_name[VXS_END_OF_TYPE];


#define VXS_MAX_FLOW_TYPE_NAME 32

/**
 * Desc: An abstract class for batching a given media flow
 * 
 * Requirements for this class:
 * 
 * 1. Provides abstract interfaces that is required for batching media flows
 * 
 * 2. Provides late-initialization (for the notification of initialization, 
 *    we use @_in_use variable. (XXX: We exclude this requirement: we use linked-list though)
 * 
 * 3. Pushes a packet and pops a segment. Here, the unit of a packet is ofpbuf
 *    and the unit of a segment is also ab abstract class VxSInNetworkSegment
 * 
 */
class VxSInNetworkFlowBatcher {
protected:
	/* name of the type of media flow */
	char _media_type_name[VXS_MAX_FLOW_TYPE_NAME];

	/* the type of this media flow */
	int32_t _media_type;

	/* indicates wether this flow batcher is in use */
	bool _in_use;

	/* flow ID: do we need pointer for this? */
	struct sw_flow_key _flow_id;

	/* a task queue that pass the segments */
	VxSInNetworkTaskQueue *_task_queue_incoming;
	VxSInNetworkTaskQueue *_task_queue_outgoing;

	/* cache of action headers */
	const struct ofp_action_header *_action_headers;
	int _action_len;

	/* datapath */
	Datapath *_datapath;

public:
	VxSInNetworkFlowBatcher(Datapath *dp, const struct sw_flow_key *fid, 
		VxSInNetworkTaskQueue *tq_in, VxSInNetworkTaskQueue *tq_out );
	~VxSInNetworkFlowBatcher();

	/* 
	 * batch a packet
	 */
	virtual int pushPacket(struct ofpbuf *ob, const struct ofp_action_header *ah, int actions_len);

	/* 
 	 * send to task incoming queue 
	 */
	virtual int sendToInputTaskQueue(struct ofpbuf *ob) = 0;

	/* 
 	 * receive from task incoming queue 
	 */
	virtual int recvFromTaskQueue() = 0;


	/* 
	 * checks if this is the same flow 
	 */
	virtual int isTheSameFlow(const struct sw_flow_key *fid);

	struct sw_flow_key * getFlowId() { return &_flow_id; };

};

#define VXS_MAX_MEDIA_FLOW 512

/**
 * Desc: A batching 'element (non-click element)' that responds to the OFP_ACTION_VXS_DXT, and etc
 * 
 * Requirements for this class:
 * 
 * 1. Support batching of heterogeneous media flows (including DXT, 
 *    raw, MPEG2, H.264, etc) at a time
 * 
 * 2. Aware of the type of media flows and provides segmented data, 
 *    where the unit of the segment depends on the type.
 */
class VxSInNetworkBatchManager {
protected:
	/* 
	 * we store the handle to the media flow batchers here 
	 */
	std::list<VxSInNetworkFlowBatcher *> _batchers;
	int _total_live_batchers;

	VxSInNetworkTaskQueue *_task_queue_incoming;
	VxSInNetworkTaskQueue *_task_queue_outgoing;

	/* holding datapath */
	Datapath *_datapath;

public:
	VxSInNetworkBatchManager(Datapath *dp, VxSInNetworkTaskQueue *tq_in, VxSInNetworkTaskQueue *tq_out );
	~VxSInNetworkBatchManager();

	/* 
	 * this function receives a packet from the OFP_ACTION_VXS_DXT and tests
	 * whether we have the flow or is it a new flow 
	 * this function is within the context of Element::push
	 */
	int recvPacket(struct ofpbuf *ob, struct sw_flow_key *fid, const struct ofp_action_header *ah, 
				int actions_len, int media_type);

	/* 
	 * this function pops out a task from the outgoing task queue and
	 * sends them to the proper output port with packetizing
	 * note that the packetizing is performed by flow batcher in order 
	 * to satisfy the context 
	 * this function is within the context of Element::run_timer
	 */
	int sendPacket();


private:
	/* 
	 * functions that handle the _batcher objects 
	 */
	/* creation function */
	VxSInNetworkFlowBatcher * createBatcher(int32_t media_type, const struct sw_flow_key *fid);
	/* searching function */
	VxSInNetworkFlowBatcher * searchBatcher(const struct sw_flow_key *fid);

};

CLICK_ENDDECLS
#endif
