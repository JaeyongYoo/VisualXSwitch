#ifndef RCONN_HH
#define RCONN_HH
#include <click/element.hh>
#include <stdint.h>
#include <time.h>
#include <semaphore.h>  /* Semaphore */

#include <click/timer.hh>

#include "lib/queue.hh"
#include "lib/rconn.hh"
#include "lib/list.hh"
#include "SynchronizedPacketQueue.hh"

CLICK_DECLS

/*
=c

RConn([])

=s debugging


OpenFlow remote connection elements.

=d

Acts as a gateway between openflow-protocol compatible elements (e.g. Datapath) and secure channel (ofprotocol)

Keyword arguments are:

=over 8


=back

=h active read/write

Returns or sets the ACTIVE parameter.

=a

Datapath */






#define OFPE_RCONN	0x00000000	// OpenFlow Protocol Element : RConn
#define OFPE_DATAPATH	0x00000001	// OpenFlow Protocol Element : Datapath
#define OFPE_BWSHAPER	0x00000002	// OpenFlow Protocol Element : BWShaper

#define RCONN_SEND_TIMER_CLOCK 10		// unit: milli sec
#define RCONN_REPORT_TIMER_CLOCK 1000		// unit: milli sec

class RConn;
struct rnc_param {
        bool liveness;
        RConn *rconnElement;
};

/* A connection to a secure channel. */
struct rconn_remote {
        struct list node;
        struct rconn *rconn;
#define TXQ_LIMIT 128           /* Max number of packets to queue for tx. */
        int n_txq;                  /* Number of packets queued for tx on rconn. */

        /* Support for reliable, multi-message replies to requests.
         *
         * If an incoming request needs to have a reliable reply that might
         * require multiple messages, it can use remote_start_dump() to set up
         * a callback that will be called as buffer space for replies. */
        int (*cb_dump)(class RConn *, void *aux);
        void (*cb_done)(void *aux);
        void *cb_aux;
};

struct rconn_stat {
public:
	uint32_t stat_total_control_in;
	uint32_t stat_control_in;
	uint32_t stat_total_control_out;
	uint32_t stat_control_out;


	uint32_t stat_queue_drop;
	uint32_t stat_total_queue_drop;
	uint32_t stat_queue_len;
	double	stat_ewma_avg_queue_len;
	double 	stat_ewma_dev_queue_len;

public:
	void update_queue_len( uint32_t len );	

};

class Datapath;
/*****************************************************************************
 * RConn click element packaging
 *****************************************************************************/
class RConn : public Element { public:

	RConn();
	~RConn();

	const char *class_name() const		{ return "RConn"; }
	const char *port_count() const		{ return "1/1"; }
	const char *processing() const		{ return PUSH; }
	bool can_live_reconfigure() const		{ return true; }

	int initialize(ErrorHandler*);
	int configure(Vector<String> &conf, ErrorHandler *errh);
	void add_handlers();
	void cleanup(CleanupStage stage);

	void push(int port, Packet *);
	void buffer_control_packet( int ofpe_type, void *msg, int length, struct rconn_remote *sender, uint32_t xid );
	void send_control_packets_to_click();

        Timer send_timer;
        Timer report_timer;
        void run_timer(Timer*);
	int parse_control_input(struct rconn_remote *sender, uint32_t xid, const void *msg_in, size_t length);
	void send_openflow_buffer(struct ofpbuf *buffer, struct rconn_remote *rconn_sender, uint32_t xid);

	void add_pvconn(struct pvconn *pvconn);

	/* create thread for nox communication */
	void create_thread_nox_commun();
        pthread_t       t_nox_commun;
        struct		rnc_param rnc;

	/* native rconn structures */
        /* Remote connections. */
        struct list remotes;        /* All connections (including controller). */

	/* */
	struct SynchronizedPacketQueue packetQueue;	

	struct rconn_stat* get_rconn_stat() { return &rstat; };

	struct rconn_stat rstat;


	/* for communicating to rconn */
	/* rather than using packet of click,
	 * directly communicate to rconn */
	Datapath *_datapath;

	sem_t mutex_rconn;

private:
	void report_stat();


protected:
public:
	char _pvconn_name[100];
	struct pvconn *_pvconn;
        /* Listeners. */
        struct pvconn **listeners;
        size_t n_listeners;


};

CLICK_ENDDECLS
#endif
