#ifndef __MULTIOP_MPEG2_STREAMINGPROXY_H__
#define __MULTIOP_MPEG2_STREAMINGPROXY_H__

CLICK_DECLS

#include <click/element.hh>

#include "../common/FlowCommon.hh"
#include "../common/Flow.hh"
#include "../papmo/PaPMo.hh"
#include "Mpeg2Common.hh"

#define MPEG2_PARSEMODE_NULL	(0x00000001 << 0)
#define MPEG2_PARSEMODE_NORMAL	(0x00000001 << 1)


/* 
 * description: 
 * structure for source flow proxy statistics 
 */
struct FlowMpeg2AdaptEncapStat {
	float throughput;
	int drop_count;

	/* stat of receiving frame type */
	int received_frame_types[STREAMINGPROXY_MAX_FRAMETYPE_BUFFER];
	int total_frame_types;

	void add_frame_type( int ft );
	void print_frame_type( FILE* fp );

	/* stat of receiving frame index */
        int received_frame_index[STREAMINGSTUB_MAX_FRAMEINDEX_BUFFER];
        int total_frame_index;

        void add_frame_index( int fi );
        void flush_frame_index();
        void print_frame_index( FILE* fp, char* buf );
};


/* 
 * description: 
 * flow table structure
 */
class FlowMpeg2AdaptEncap : public Flow  {

public:
	/* to be portable to accept IP or ethernet packet */
	
	static const struct bpadapt_header* get_bpadapt_header_readonly(const Packet* p);

	/* index of video frame */
	unsigned int frameindex;

	/* remember old frame type to update the follow-up ts packets */
	unsigned int prev_frame_type;

	/* remember current frame type to mark the number of packets per frame */
	/* similar counter part is implemented in streamingstub */
	unsigned int curr_enque_frametype; /* used in enque */
	unsigned int curr_deque_frametype; /* used in deque */
	
	unsigned int pkts_per_frame;

	/* to control the flushing the frames (set by enque, and used by deque) */
	unsigned int flush_frame;
	
	/*
	 * statistics of this queue 
	 */
	struct FlowMpeg2AdaptEncapStat stat;

	virtual void clear();

	/* call frequency: timer-based */
	void print_stat();


	/* enqueue and dequeue a packet */
	/* these two functions are not the trivial enque and deque_and_send function */
	/* they perform special buffering and sending to count pkts_per_frame */
	int enque( Packet* p, const Element* e );
	int deque_and_send( const Element* e, papmo* );

	/* receive a packet from host */
	int parse_packet( Packet* p_in, WritablePacket** p_out,  int *p_out_len, int parsemode );

	/* deprecated */
	int parse_packet_2( Packet* p_in, WritablePacket** p_out, int* p_out_len ) ;
	
	/* new way of separate_packet */
	int repacketization( Packet* p_in, WritablePacket** p_sep1, int start, int end, int frametype1 );


	/* backpressure adaptation header manipulation */
	WritablePacket* encapsulate_bpadapt( Packet* p, int frametype );	

	/* we have to separate a packet into two packets to classify I, P, B packets */
	/* deprecated */
	int separate_packet( Packet* p_in, WritablePacket** p_sep1, WritablePacket** p_sep2, int separation_point,
					int frametype1, int frametype2 ); 
	

	/* print ts headers */
	int print_mpeg2ts_udp_packet( Packet* p, int );

	/* ts/pes/mpeg2 parsing */
	int packet_to_ts( struct tspacket_table* ts_table, Packet* p, int vervose );
	int packet_to_ts_old( struct tspacket_table* ts_table, Packet* p );

	/* ts2pes parser */
	int mpeg2_ts_breakdown_headers(struct tspacket* ts, int video_demux_id);
	int mpeg2_demux_wrapper(struct tspacket* ts, int video_demux_id);
	int mpeg2_mpeg2parser_wrapper(struct tspacket* ts);

	/* print error message */
	void print_error_message( int errcode );

	int last_receiving_frametype;

	/* 
	 * description: 
	 * in order to use the original demux from shpark and yongjae,
	 * use the same definition
	 * only used in demux function 
	 */
        int state;
        int state_bytes;
        uint8_t head_buf[264];
	int framenum;

        pktlen_t dmx_len;
	pktlen_t* demux(uint8_t* buf, uint8_t* end, int demux_pid , int flags);
	int parse_mpeg2 (pktlen_t * par_len);
	
};



CLICK_ENDDECLS

#endif
