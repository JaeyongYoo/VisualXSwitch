#ifndef __VXS_INNETWORK_DXT_BATCHER__
#define __VXS_INNETWORK_DXT_BATCHER__

#include <click/config.h>
#include "VxSInNetworkBatchManager.hh"

CLICK_DECLS

class VxSInNetworkTaskQueue;

class VxSInNetworkDXTSegment : public VxSInNetworkSegment {
public:
	VxSInNetworkDXTSegment();
	~VxSInNetworkDXTSegment();
};

class VxSInNetworkDXTBatcher : public VxSInNetworkFlowBatcher {
public:
	VxSInNetworkDXTBatcher(const struct sw_flow_key *fid, 
		VxSInNetworkTaskQueue *tq_in, VxSInNetworkTaskQueue *tq_out );
	~VxSInNetworkDXTBatcher();
public:

        /* 
         * batch a packet
         */
        virtual int pushPacket(struct ofpbuf *ob, const struct ofp_action_header *ah, int actions_len);

	/* 
 	 * send to task queue
	 */
	virtual int sendToTaskQueue(struct ofpbuf *ob);

	/*
	 * recv from task queue (outgoing)
	 */
	virtual int recvFromTaskQueue(Datapath *);

};


CLICK_ENDDECLS
#endif
