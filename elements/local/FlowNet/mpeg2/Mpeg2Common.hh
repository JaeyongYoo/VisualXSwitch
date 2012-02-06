#ifndef __MULTIOP_MPEG2_STREAMING_COMMON_H__
#define __MULTIOP_MPEG2_STREAMING_COMMON_H__

/* error code definition */
#define MPEG2_SUCCESS_PARSE 0
#define MPEG2_WARN_PID_NOT_CORRECT 1
#define MPEG2_WARN_NO_ES_HEADER 2
#define MPEG2_ERROR_SYNC_NOT_CORRECT 3
#define MPEG2_ERROR_ADAPTATION_FIELD_TOO_LONG 4
#define MPEG2_ERROR_ES_TOO_LONG 5
#define MPEG2_ERROR_UDP_SIZE_NOT_CORRECT 6
#define MPEG2_ERROR_OUTOFMEMORY 7

/* mpeg2 ts related parameters */
#define MPEG2_TS_PER_UDP	7
#define MPEG2_TS_SIZE		188

#define MPEG2_TS_HEADER_SIZE_WO_ADAPTATION	4
#define MPEG2_TS_HEADER_SIZE_W_ADAPTATION	5

/* mpeg2 frame type definitions */
#define MPEG2_NON_VIDEO 0
#define MPEG2_I_FRAME 1
#define MPEG2_P_FRAME 2
#define MPEG2_B_FRAME 3 

/* streaming proxy status parameters */
#define STREAMINGPROXY_MAX_FRAMETYPE_BUFFER	100
#define STREAMINGSTUB_MAX_FRAMEINDEX_BUFFER 	100
#define STREAMINGPROXY_MAX_SEPARATION	7 /* worst separation */

/* 
 * description: 
 * in order to use the original demux from shpark and yongjae,
 * use the same definition
 */
#define DEMUX_HEADER 0
#define DEMUX_DATA 1
#define DEMUX_SKIP 2
#define DEMUX_PAYLOAD_START 1

#define NEEDBYTES(x)                                            			\
        do {                                                    			\
                int missing;                                            		\
                									\
                missing = (x) - bytes;                                  		\
                if (missing > 0) {                                      		\
                        if (header == head_buf) {                               	\
                                if (missing <= end - buf) {            			\
                                        memcpy (header + bytes, buf, missing);		\
                                        buf += missing;					\
                                        bytes = (x);					\
                                }       else {						\
                                        memcpy (header + bytes, buf, end - buf);	\
                                        state_bytes = bytes + end - buf;		\
                                        return & dmx_len;				\
                                }                                               	\
                        }       else {                      				\
                                memcpy (head_buf, header, bytes);               	\
                                state = DEMUX_HEADER;                           	\
                                state_bytes = bytes;                            	\
                                return & dmx_len;                                       \
                        }                                                       	\
                }                                                       		\
        } while (0)


#define DONEBYTES(x)            	\
        do {                    	\
                if (header != head_buf) \
                buf = header + (x);     \
        } while (0)  

typedef struct pktlen_s
{
        uint8_t * pkbuf;
        uint8_t * pkend;
        int pic_type;
} pktlen_t;

/* 
 * description: 
 * structure for a ts packet
 */
struct tspacket {

	/*
	 * frame type 
	 * note that this frametype may not be exactly correct
	 * there is a possibility that a ts packet contains two video frames
	 * and this frametype may miss one 
	 */
	int frame_type;

	/* 
	 * this variable is just a debugging variable 
	 * it indicates the return value when these ts 
	 * packets are parsed through mpeg2 parser 
	 */
	int return_value;
};

/* 
 * description: 
 * structure holding the array of ts packets
 */
struct tspacket_table {
	struct tspacket ts[MPEG2_TS_PER_UDP];
	int total_ts;

	/* debugging routine */
	void dump_table(FILE* fp);

};

#define BPADAPT_MAGIC_HEADER 0xfaadf00d

/* Backpressure Adaptation Header between UDP and VLC TS packets */
struct bpadapt_header {
/*
	uint8_t frametype;
	uint8_t reserved_1;
	uint8_t reserved_2;
	uint8_t reserved_3;
*/
	
	uint32_t magicHeader;
	/* type of frame (I, B, or P) */
	uint32_t frametype;

	/* index of the frame */
	uint32_t frameindex;

	/* how many network packets to form a frame */
	uint32_t pkts_per_frame;

	/* index of the frame */
	uint32_t pkts_index;
	
};




#endif
