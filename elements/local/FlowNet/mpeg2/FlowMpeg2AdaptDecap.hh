#ifndef __MULTIOP_MPEG2_STREAMINGSTUB_H__
#define __MULTIOP_MPEG2_STREAMINGSTUB_H__

CLICK_DECLS

#include "../common/FlowCommon.hh"
#include "../common/Flow.hh"
#include "../papmo/PaPMo.hh"
#include "Mpeg2Common.hh"
#include "Mpeg2FrameReceivingBuffer.hh"


/* 
 * description: 
 * structure for source flow proxy statistics 
 */
struct FlowMpeg2AdaptDecapStat {

	/* variables that denote statistics and information */
	/* frame statistics */
	int	total_frames_sent;
	int	total_frame_drop_count;
	int	frame_drop_unordered;
	int	frame_drop_packet_loss;

	/* packet statistics */
	int	total_packets_sent;
	int	total_packet_drop_count;

	/* throughput statistics */
	int	bytes_received;
	
	/* start time */
	struct	timeval start_time;
	struct	timeval last_update_time;

	/* window-based statistics */
	int sent_frame_index[STREAMINGSTUB_MAX_FRAMEINDEX_BUFFER];
	int drop_frame_indicator[STREAMINGSTUB_MAX_FRAMEINDEX_BUFFER];
	struct timeval sent_frame_time[STREAMINGSTUB_MAX_FRAMEINDEX_BUFFER];
	int total_frame_index;

	/* for file-tracing */
	FILE* fp_trace;

	/* methods */
	void log_frame_index( struct timeval* tv, int fi, int frametype, int drop, int missing_pkts );
	void log_pkts_arrival( int packet_index, int pkt_length );
	void flush_frame_index();
	void print_frame_index( FILE* fp, char* buf );

	void start_timestamp();

};


/* video playout options */
#define MPEG2_FPS 20
#define MPEG2_PLAYOUT_TIME_BUDGET 0.001

/* 
 * description: 
 * flow table structure
 */
class FlowMpeg2AdaptDecap : public Flow  {
public:
	FlowMpeg2AdaptDecap(); 
	~FlowMpeg2AdaptDecap(); 

	/* the following structure is a fixed size queue */
	FrameReceivingBuffer* frameBufferHead;
	FrameReceivingBuffer* frameBufferTail;
	int buffer_size;



	/* 
	 * policies of VLC streaming stub 
	 */
	/* the frame is delivered in order 
	   out of order frames are discarded */
	int stub_policy_frame_delivery_inorder;
	uint32_t sp_frame_index; /* follow up variable to enforce the above policy */
	

	/* enque and deque function */
	/* they are dealing with frame buffers, I think it would be quite a job at the moment */
	int enque( Packet* p, const Element* e, papmo* papmo );
	int deque_and_send( const Element* e, papmo*  );
	FrameReceivingBuffer* search_frame_buffer( Packet* p );

	
	int flush_one_buffer(const Element* e, papmo* );

	void deleteBuffer( FrameReceivingBuffer* );

	bool isInitialFrameBufferReady();
	FrameReceivingBuffer* create_new_buffer( int ftype, int findex, int pkts );
	void disconnect( FrameReceivingBuffer* );

	WritablePacket* decapsulate_bpadapt( Packet* p );

	void toString( char* buf, int len );


	bool video_started;
	struct timeval tv_video_play_start_time;

	/* playout continuity control */
	struct timeval tv_last_frame_sent;
	int last_frame_index;

	uint64_t get_expected_playout_time( int frameindex );

	bool isThisFrameReady( FrameReceivingBuffer* );

	void sendFrameToUpperLayer( FrameReceivingBuffer* buffer, const Element* , papmo* );

	bool checkFrameIntegrity();

	/*
	 * statistics of this queue 
	 */
	struct FlowMpeg2AdaptDecapStat stat;

	int adaptivePlayoutFactor;

	int killed_buffer;

#define MAX_KILLED_PKT_SIZE 1000
	uint16_t killed_pkt[MAX_KILLED_PKT_SIZE];
	int	total_killed_pkts;

	void print_stat();

	virtual void clear();
};

CLICK_ENDDECLS
#endif
