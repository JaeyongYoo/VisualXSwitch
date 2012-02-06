#ifndef CLICK_DATAPATH_H
#define CLICK_DATAPATH_H

#include <click/element.hh>
#include <click/string.hh>
#include <click/timer.hh>
#include <click/etheraddress.hh>
#include <pthread.h>

#include <stdint.h>

#include <sys/ioctl.h> /* include headers for get_port_desc */
#include <net/if.h>
#include <linux/sockios.h>
#include <linux/ethtool.h>

#include "include/openflow/nicira-ext.hh"
#include "lib/ofpbuf.hh"
#include "lib/timeval.hh"
#include "lib/list.hh"
#include "lib/packets.hh"
#include "../VxS/VxSInNetworkBatchManager.hh"
#include "../VxS/VxSInNetworkTaskQueue.hh"
#include "../VxS/VxSInNetworkTaskDispatcher.hh"


//#include "../standard/ratesettable_bandwidthshaper.hh"

#include "RConn.hh"

CLICK_DECLS

/*
=c

Datapath([PORT1_ADDR, PORT1_NAME, PORT2_ADDR, PORT2_NAME, I<keywords>])

=s debugging


Datapath for switching controlled by remote controller.

=d

Switches frames coming from ports (e.g., PORT1_ADDR with PORT1_NAME) to another ports (e.g., PORT2_ADDR with PORT2_NAME ).

Keyword arguments are:

=over 8

=item HOST_ADDR

The MAC address applied on tap interface which can be determined on your own.

=item DPID

Datapath ID (64bit number)

=item PORT1_ADDR

An interface MAC address that will be working as OpenFlow Port.
It supports upto 5 ports in the current implementation such as PORT2_ADDR, ..., PORT5_ADDR.

=item PORT1_NAME

The interface name that is corresponding to the PORT1_ADDR, such as eth1.
Note that PORT1_ADDR should be the interface name of PORT1_NAME.

=back

=h active read/write

Returns or sets the ACTIVE parameter.

=a

RConn */ 


#define CLICK_PORT_NUM_TAP		1
#define CLICK_PORT_START_NUM_ITF	2

/* Buffers are identified by a 31-bit opaque ID.  We divide the ID
 * into a buffer number (low bits) and a cookie (high bits).  The buffer number
 * is an index into an array of buffers.  The cookie distinguishes between
 * different packets that have occupied a single buffer.  Thus, the more
 * buffers we have, the lower-quality the cookie... */
#define PKT_BUFFER_BITS 8
#define N_PKT_BUFFERS (1 << PKT_BUFFER_BITS)
#define PKT_BUFFER_MASK (N_PKT_BUFFERS - 1)

#define PKT_COOKIE_BITS (32 - PKT_BUFFER_BITS)



/* type of ethernet */
/* jyyoo note: this one should be in clicknet/ether.h */
#define ETHERTYPE_LLDP          0x88cc


/*	
	Datapath click element description
  	Ports: 2- inputs, 2- outputs (the same number of inputs)

	Port connection usage 
	
		output port [0] ==> should connect to RConn click element input port `i` for any i
		input port [0] ==> should connect to RConn click element output port `i` for the same i
	
		output port [1] ==> should connect to FromHost click element
		input port [1] ==> should connect to ToHost click element

		output port [2~] ==> should connect to FromDevice interface `j` for any j
		output port [2~] ==> should connect to ToDevice interface `j` for the same j
	
*/
	


#define NETDEV_MAX_QUEUES 8


struct sw_flow;
struct sw_queue {
	struct list node; /* element in port.queues */
	unsigned long long int tx_packets;
	unsigned long long int tx_bytes;
	unsigned long long int tx_errors;
	uint32_t queue_id;
	uint16_t class_id; /* internal mapping from OF queue_id to tc class_id */
	struct click_port *port; /* reference to the parent port */
	/* keep it simple for now, only one property (assuming min_rate) */
	uint16_t property; /* one from OFPQT_ */
	uint16_t min_rate;
};

/********************************************************************
 * jyyoo rewrite click_port based on `sw_port` from OpenFlow 1.0
 * mostly rewrote the part that handling interface devices
 *********************************************************************/
struct click_port {

	struct list queue_list; /* list of all queues for this port */

	char devname[8];
	uint8_t macaddr[ETH_ADDR_LEN];

	uint32_t config;            /* Some subset of OFPPC_* flags. */
	uint32_t state;             /* Some subset of OFPPS_* flags. */
	class Datapath *datapath;
	struct list node; /* Element in datapath.ports. */

	// interface related variables 
	int linux_fd;
	uint32_t curr;
	uint32_t supported;
	uint32_t advertised;
	uint32_t peer;
	uint32_t speed;

	unsigned long long int rx_packets, tx_packets;
	unsigned long long int rx_bytes, tx_bytes;
	unsigned long long int tx_dropped;

	/* TODO: make a note that explains not to confuse the below two port numbers */
	int click_port_num;
	uint16_t port_no;

	/* port queues */
	uint16_t num_queues;
	struct sw_queue queues[NETDEV_MAX_QUEUES];

	/* mark for using this port or not */
	uint8_t port_on_use;
};


#define DP_MAX_PORTS 255
BUILD_ASSERT_DECL(DP_MAX_PORTS <= OFPP_MAX);

struct func_stat {
	uint32_t output_packet;
};

class Datapath;
struct onc_param {
	bool liveness;
	Datapath *of;
};

/* Datapath Packet buffering. */

#define OVERWRITE_SECS  1

struct dp_packet_buffer {
	struct ofpbuf *buffer;
	uint32_t cookie;
	time_t timeout;
};

class DatapathPacketBuffer {
private:
	struct dp_packet_buffer _buffers[N_PKT_BUFFERS];
	uint32_t _buffer_idx;
public:
	DatapathPacketBuffer();
	~DatapathPacketBuffer();
	uint32_t save_buffer(struct ofpbuf *buffer);
	struct ofpbuf *retrieve_buffer(uint32_t id);
	void discard_buffer(uint32_t id);

};


#define BASE_TIMER_INTERVAL      1000 /* basic interval is 1 second */
class Datapath : public Element { public:

	Datapath();
	~Datapath();

	const char *class_name() const		{ return "Datapath"; }
	const char *port_count() const		{ return "-/-"; }
	const char *processing() const		{ return PUSH; }
	int configure(Vector<String> &conf, ErrorHandler* errh);
	int initialize(ErrorHandler*);

	void push(int input, Packet* p);
	void add_handlers();
	void run_timer(Timer* t);

	// ngkim: get_port_desc
	void get_port_description(struct ofp_phy_port* desc, String ifname, int fd);
	void port_description(struct ofp_phy_port* desc, uint16_t port_no);

	Timer base_timer;
	Timer vxs_timer;

	pthread_t	t_openflow_nox_commun;
	struct onc_param onc;
	void create_thread_openflow_nox_commun();
	
   	void cleanup(CleanupStage);
	
	EtherAddress hostMacAddr;

	DatapathPacketBuffer packetbuffer;

	/* a VisualXSwitch component */
	VxSInNetworkBatchManager vxsManager;
	VxSInNetworkTaskDispatcher vxsDispatcher;
	VxSInNetworkTaskQueue taskQueueIncoming;
	VxSInNetworkTaskQueue taskQueueOutgoing;

public:

	struct sw_chain* get_chain() { return _chain; };
	uint64_t get_id() { return _id; };
	uint16_t get_num_queues() { return _num_queues; };
	struct click_port* get_port(int i) { return i < DP_MAX_PORTS && i >= 0 ? &_ports[i] : NULL; };
	struct click_port* get_local_port() { return _local_port; };
	struct list* get_port_list() { return &_port_list; };
	char* get_dp_desc() { return _dp_desc; };


	struct click_port *dp_lookup_port(uint16_t port_no);
	int send_openflow_buffer(struct ofpbuf *buffer, struct rconn_remote *rconn_sender, uint32_t xid );
	void dp_send_error_msg(struct rconn_remote *rconn_sender, uint32_t xid, uint16_t type, uint16_t code, 
		const void *data, size_t len);
private: public:

	/* for delayed configuration */
	bool _have_port_macaddr[DP_MAX_PORTS];
	bool _have_port_devname[DP_MAX_PORTS];
	EtherAddress _port_macaddr[DP_MAX_PORTS];
	String _port_devname[DP_MAX_PORTS];


	struct func_stat fs;
	/*********************************************************************
	 * from Datapath 
	 *********************************************************************/
	char _mfr_desc[DESC_STR_LEN];
	char _hw_desc[DESC_STR_LEN];
	char _sw_desc[DESC_STR_LEN];
	char _dp_desc[DESC_STR_LEN];
	char _serial_num[SERIAL_NUM_LEN];

	struct datapath *_dp;
	uint64_t _dpid;
	char _str_port_list[100];
	char _str_local_port[100];
	uint16_t _num_queues;

	/* Unique identifier for this datapath */
	uint64_t  _id;

	struct sw_chain *_chain;  /* Forwarding rules. */

	/* Configuration set from controller. */
	uint16_t _flags;
	uint16_t _miss_send_len;

	/* Switch ports. */
	struct click_port _ports[DP_MAX_PORTS];
	struct click_port *_local_port;
	struct list _port_list; /* All ports, including local_port. */

	/* for communicating to rconn */
	/* rather than using packet of click,
	 * directly communicate to rconn */
	RConn *_rconn;

	/* member functions */
	int dp_add_port(const char *devname, const uint8_t *macaddr, uint16_t, uint16_t click_port_num);
	int dp_add_local_port( const char *netdev, uint16_t);
	void dp_add_pvconn( struct pvconn *);
	void dp_send_flow_end(struct sw_flow *,
			enum ofp_flow_removed_reason);

	void dp_output_port(Packet *packet, int in_port, int out_port, uint32_t queue_id);
	void dp_output_port(struct ofpbuf *, int in_port, 
			int out_port, uint32_t queue_id, bool ignore_no_fwd);
	void dp_output_control(struct ofpbuf *, int in_port,
			size_t max_len, int reason);
	struct click_port * dp_lookup_port(struct datapath *, uint16_t);
	void fwd_port_input(struct ofpbuf *, struct click_port *);
	int fwd_control_input(void *msg, int length, struct rconn_remote *sender, uint32_t xid);

private:
	void fill_port_desc(struct click_port *p, struct ofp_phy_port *desc, struct ofpbuf* buffer=NULL);
	int new_port(struct click_port *port, uint16_t port_no, const char *netdev_name, 
			const uint8_t *macaddr, uint16_t num_queues, uint16_t click_port_num);
	void send_port_status(struct click_port *p, uint8_t status);
	void dp_send_features_reply(struct rconn_remote *rconn_sender, uint32_t xid);
	void update_port_flags(const struct ofp_port_mod *opm);
	int output_all(struct ofpbuf *buffer, int in_port, int flood);
	void output_packet(struct ofpbuf *buffer, uint16_t out_port, uint32_t queue_id);
	void output_packet(Packet *packet, uint16_t out_port, uint32_t queue_id);
	int run_flow_through_tables(struct ofpbuf *buffer, struct click_port *p);


	int recv_features_request(struct rconn_remote *rconn_sender, uint32_t xid, const void *msg UNUSED);
	int recv_get_config_request(struct rconn_remote *rconn_sender, uint32_t xid,	const void *msg UNUSED);
	int recv_set_config(struct rconn_remote *rconn_sender, uint32_t xid UNUSED, const void *msg);
	int recv_packet_out(struct rconn_remote *rconn_sender, uint32_t xid, const void *msg);
	int recv_port_mod(struct rconn_remote *rconn_sender, uint32_t xid UNUSED, const void *msg);
	int add_flow(struct rconn_remote *rconn_sender, uint32_t xid, const struct ofp_flow_mod *ofm);
	int mod_flow(struct rconn_remote *rconn_sender, uint32_t xid, const struct ofp_flow_mod *ofm);
	int recv_flow(struct rconn_remote *rconn_sender, uint32_t xid, const void *msg);
	int desc_stats_dump(void *state UNUSED, struct ofpbuf *buffer);
	int flow_stats_dump(void *state, struct ofpbuf *buffer);
	int aggregate_stats_dump(void *state, struct ofpbuf *buffer);
	int table_stats_dump( void *state UNUSED, struct ofpbuf *buffer);
	int port_stats_dump(void *state, struct ofpbuf *buffer);
	int queue_stats_dump(void *state, struct ofpbuf *buffer);
	int vendor_stats_dump(void *state, struct ofpbuf *buffer UNUSED);
	int recv_stats_request(struct rconn_remote *rconn_sender, uint32_t xid, const void *oh);
	int recv_queue_get_config_request(struct rconn_remote *rconn_sender, uint32_t xid, const void *oh);
	int recv_vendor(struct rconn_remote *rconn_sender, uint32_t xid, const void *oh);
	void flowtable_timer();


};

CLICK_ENDDECLS
#endif /* openflow.h */
