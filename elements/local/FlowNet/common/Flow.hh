#ifndef __MULTIOP_FLOW_H__
#define __MULTIOP_FLOW_H__

#include <click/config.h>
CLICK_DECLS

#include <float.h>
#include <clicknet/wifi.h>

#include "FlowCommon.hh"
#include "PacketQueue.hh"

struct FlowDesc;
struct Flow;
struct FlowID;
struct SchedInfo;
struct NexthopInfo;


/*
 * description: 
 * structure for flow statistics 
 */
struct FlowStat {
	/* not implemented */
	int index;
	double ingress_speed;
	double engress_speed;
	double backpressure;	

	/* total packet count statistics */
	int sent_pkt_count;
	
	/* total packet dropped */
	int drop_count;

	/* current packet count */
	int current_pkt_count;
	/* current packet dropped */
	int current_drop_count;
};


class VcFlowClassify;
/*!
 * data structures 
 */
#define MAX_FLOWID_LEN 64
struct FlowID {
	uint8_t id[MAX_FLOWID_LEN];
	uint32_t len;
	
	/* the master classifier that classifies 
	 * this FlowID */
	VcFlowClassify *classifier;
	inline int cmp(const struct FlowID* fid);
};


struct FlowDesc {
	struct Flow* flow;
};


#define FLOW_MAX_AGE 10

class Flow {
public:
	Flow() {};
	~Flow() {destroy();};

	/********************************************************
	 * Generic Flow Buffer Manipulation
	 ********************************************************/
	struct FlowID fid;
	struct FlowDesc fd;

	/* age and liveness of the flow
	 * for checking the activeness of the flow */
	uint8_t age;

	/* simple fifo queue implementation */
	uint16_t queue_len;

	/* simple stat */
	uint32_t qdrop;
	uint32_t qdrop_now;
	uint32_t total_sent;
	uint32_t sent_now;
	struct PacketQueue q;

	int init(int max_queue_size);
	int setup(const struct FlowID* fid);

	void destroy();

	inline int 	cmp(const struct FlowID*);
	inline int 	touch();
	bool		queue_empty()  { return queue_len == 0; }
	uint16_t	max_queue_length() const { return (uint16_t) q.max_size(); }
	uint16_t	queue_length() const { return queue_len; }
	uint16_t*	queue_length_ref() { return &queue_len; }
	struct FlowID*	getFlowID() { return &fid; }
	int		does_it_expire();
	int		enque(Packet* p);
	Packet* 	deque();

	double		queue_loss() { return (double)qdrop_now/(double)sent_now; };

	virtual void clear();
	virtual void toString(char* buf, int len);

	/* static function */
	static int allocate(struct Flow** f);
	static void free(struct Flow* f);

	/* statistics of this queue */
	struct FlowStat stat;

};

inline int Flow::touch()
{
	age = 0;
	return 0;
}

inline int Flow::cmp(const struct FlowID* f)
{
	return fid.cmp( f );
}

inline int FlowID::cmp(const struct FlowID* fid)
{
	return fid->len == len ? memcmp(fid->id, id, len) : -1;
}



CLICK_ENDDECLS

#endif
