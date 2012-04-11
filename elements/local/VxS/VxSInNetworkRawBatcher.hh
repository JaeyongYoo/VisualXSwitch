#ifndef __VXS_INNETWORK_RAW_BATCHER__
#define __VXS_INNETWORK_RAW_BATCHER__

#include <click/config.h>
#include "VxSInNetworkBatchManager.hh"
#include "VxSInNetworkTaskQueue.hh"
#include "VxSInNetworkRawSegment.hh"

CLICK_DECLS

class VxSInNetworkTaskQueue;

/**
 * Desc: this raw-video-frame batcher is aware of STIP transport protocol 
 */
class VxSInNetworkRawBatcher : public VxSInNetworkFlowBatcher {
public:
	VxSInNetworkRawBatcher(Datapath *dp, const struct sw_flow_key *fid, 
		VxSInNetworkTaskQueue *tq_in, VxSInNetworkTaskQueue *tq_out );
	~VxSInNetworkRawBatcher();

public:
        /* 
         * batch a packet
         */
        virtual int pushPacket(struct ofpbuf *ob, const struct ofp_action_header *ah, int actions_len);

	/* 
 	 * send to task queue (incoming)
	 */
	virtual int sendToInputTaskQueue(struct ofpbuf *ob);

	/*
	 * recv from task queue (outgoing)
	 */
	virtual int recvFromTaskQueue();
protected:
	VxSInNetworkRawSegment * createNewSegment();
	void stip_initiation_packet_received(struct stip_initiation_header *sihdr, 
		struct ofpbuf *ob, const struct ofp_action_header *ah, int actions_len);
	void stip_process_initiation_packet(struct stip_initiation_header *sihdr, 
		struct ofpbuf *ob, const struct ofp_action_header *ah, int actions_len);
	void stip_data_packet_received(struct stip_transport_header *shdr);

private: // for debugging purpose
	void list_all_segments();

private:
	/* a list of segments */
	std::list<VxSInNetworkRawSegment *> _segments;

	/*
	 * default byte-size of a segment 
 	 */
	uint32_t _segment_size;
	/*
	 * This STIP sender assumes to receive STIP 
	 */
        bool _initiated;

        /* receiving status */
        uint32_t _max_pixel_blocks;
        uint32_t _current_receiving_frame_index;
        uint32_t _lost_pixel_blocks;

        /* video related info */
        int32_t _src_Bpb;
        int32_t _proc_Bpb;
        int32_t _frame_height;
        int32_t _frame_width;

};


CLICK_ENDDECLS
#endif
