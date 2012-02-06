#ifndef _PAPMO_HH_
#define _PAPMO_HH_
#include <click/config.h>
#include <click/packet.hh>
#include <click/ipaddress.hh>
#include <clicknet/ip.h>
#include <clicknet/ether.h>
#include <clicknet/udp.h>


/* algorithms to be monitored */
#include "../common/Schedule.hh"
#include "../common/BWShape.hh"
#include "../common/CD.hh"


CLICK_DECLS

/*
 * lock-free circular buffer 
 */
struct lfc_buffer {
	
	int create_buffer(int bs, int is);
	int insert(void* item);
	int pop(void* item);

	bool is_full();
	bool is_empty();
	uint32_t size();

	void **_item;
	int _buffer_size;
	int _item_size;
	int _head;
	int _tail;
};


/*
 * papmo IPS component of Click
 */
struct papmo_thread_arg {
	struct lfc_buffer* p_buffer;
	int* p_thread_liveness;
};

#define SERVER_MONITOR_PORT 30004
class papmo {
public:
	papmo();
	~papmo();
	int init( int papmo_bs, IPAddress monServerIP );
	int destroy();


	int do_monitor( int tag, int pos, const Packet* p, const Flow* f, const VcSchedule* sched, const VcBWShape* shape, const VcCongestionDetection* cd );
	
	int do_monitor( struct composed_trace *ct );
	int do_monitor(Packet* p, uint32_t tag, uint32_t qlen, uint32_t qlen_next);
	int send_to_server( uint8_t* buf, uint32_t len );

	pthread_t _thread_send;
	int _thread_liveness;

	IPAddress _monServerIP;
	int _sockMonServer;
	struct sockaddr_in _sout;


	struct lfc_buffer _buffer;
};

#define MAX_COMPOSEDTRACES_PER_PACKET 1400 / sizeof(struct composed_trace)

#define PAPMO_CAPTURE_HEAD_SIZE sizeof(struct click_ether) + sizeof(struct click_ip) + 20 /*TCP header*/

#define COMPOSED_TRACE_TAG_NO_TAG	(0x00000000)
#define COMPOSED_TRACE_TAG_CORE		(0x00000001 << 0)
#define COMPOSED_TRACE_TAG_MPEG		(0x00000001 << 1)
#define COMPOSED_TRACE_TAG_FLOW		(0x00000001 << 2)
#define COMPOSED_TRACE_TAG_MPEG_FRAME	(0x00000001 << 3)

#define COMPOSED_TRACE_POS_L4_OUT	0x00004444
#define COMPOSED_TRACE_POS_L4_PRE_IN	0x00000444
#define COMPOSED_TRACE_POS_L4_IN	0x00000044
#define COMPOSED_TRACE_POS_L3_OUT	0x00003333
#define COMPOSED_TRACE_POS_L3_IN	0x00000033

struct composed_trace {
	uint32_t sec;
	uint32_t usec;
	uint32_t tag;
	uint32_t pos;

	/* flow buffer info */
	uint32_t qdrop;
	uint16_t qlen_self;
	uint16_t qlen_next;

	/* core related */
	float	core;
	float	slope;

	/* mpeg */
	uint32_t frameindex;
	uint8_t frametype;
	uint8_t pkts_per_frame;
	uint8_t pkts_index;
	uint8_t reserved;

	uint8_t header[PAPMO_CAPTURE_HEAD_SIZE];
};


CLICK_ENDDECLS
#endif
