#ifndef __VXS_INNETWORK_RAW_BATCHER__
#define __VXS_INNETWORK_RAW_BATCHER__

#include <click/config.h>
#include "VxSInNetworkBatchManager.hh"
#include "VxSInNetworkTaskQueue.hh"

CLICK_DECLS

class VxSInNetworkTaskQueue;

class VxSInNetworkRawSegment : public VxSInNetworkSegment {
private:
	/* max segment size */
	uint32_t _segment_size;

	/* segment data */
	uint8_t *_segment;

	/* currently written size */
	uint32_t _written_size;

	/* media-property (bytes per pixel block) incoming source */
	uint32_t _src_Bpb;

	/* media-property (bytes per pixel block) outgoing source after processed */
	uint32_t _proc_Bpb;

	/* number of pixel blocks */
	uint32_t _num_pixel_blocks;

public:
	VxSInNetworkRawSegment(uint32_t seg_size);
	~VxSInNetworkRawSegment();

	/* 
	 * Desc: push data into segment 
	 * 
	 * returns the number of bytes written
	 */
	uint32_t push(uint8_t *data, int size);

	int isFull() { return _written_size == _segment_size; };
	uint8_t * getSegment() { return _segment; };
	uint32_t getSize() { return _segment_size; };
	uint32_t getWrittenSize() { return _written_size; };
	void setWrittenSize(uint32_t t) { _written_size = t; };

	void setBytePerPixelBlocks(uint32_t src, uint32_t proc) { _src_Bpb = src; _proc_Bpb = proc; };
	uint32_t getNumPixelBlocks() { return _num_pixel_blocks; };

	/* 
	 * this is the part that deals about packetizing this segment
 	 * for packetizing the segment, we use _proc_Bpb as the unit for packetizing
	 */
	uint32_t afterProcessedSize() { return _proc_Bpb * _num_pixel_blocks; };
	uint32_t getNumberOfPackets(uint32_t packet_size);
	Packet * packetize(uint32_t data_size, uint8_t *network_header, uint32_t network_header_len);
	
};

/**
 * Desc: this raw-video-frame batcher is aware of STIP transport protocol 
 */
class VxSInNetworkRawBatcher : public VxSInNetworkFlowBatcher {
public:
	VxSInNetworkRawBatcher(const struct sw_flow_key *fid, 
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
	virtual int sendToTaskQueue(struct ofpbuf *ob);

	/*
	 * recv from task queue (outgoing)
	 */
	virtual int recvFromTaskQueue(Datapath *);
protected:
	VxSInNetworkRawSegment * createNewSegment();
	void stip_initiation_packet_received(struct stip_initiation_header *sihdr);
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
