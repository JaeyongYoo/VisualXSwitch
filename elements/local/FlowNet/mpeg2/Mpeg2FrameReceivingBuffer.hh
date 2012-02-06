#ifndef __MPEG2FRAMERECEIVINGBUFFER_H__
#define __MPEG2FRAMERECEIVINGBUFFER_H__

CLICK_DECLS

#include "../common/FlowCommon.hh"
#include "../common/Flow.hh"
#include "Mpeg2Common.hh"

#define MAX_FRAME_BUFFER_SIZE 30 /* total 10 frames */


/* special purpose frame receiving buffer */
/* in this class, we can further perform jitter handling */
class FrameReceivingBuffer {
public: /* just for easy of access */
	FrameReceivingBuffer* pNext;
	FrameReceivingBuffer* pPrev;

public:
	/* in this Packet, we use Packet structure as linked list */
	Packet* pHead;
	
	/* information for video frame */
	uint32_t frametype;
	uint32_t frameindex;
	uint32_t pkts_per_frame;
	uint32_t total_pkts;

	/* additional information */
	struct timeval tv_first_packet_received;

	
public:
	FrameReceivingBuffer(int frametype, int frameindex, int pkts_per_frame);
	void reset(uint16_t*, int*);
	double received_ratio();
	int missing_packets_count();
	int enque(Packet* p);
	Packet* deque();
	void dump_buffer(FILE* fp, char* buf);
	int empty();
	int is_same_frameindex(Packet* p);
	uint32_t get_frameindex() { return frameindex; }
	uint32_t get_frametype() { return frametype; }
	struct timeval* get_first_packet_receive_time() { return &tv_first_packet_received; }

};

#endif
