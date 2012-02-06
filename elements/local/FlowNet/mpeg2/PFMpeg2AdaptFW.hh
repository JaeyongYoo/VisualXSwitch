// -*- c-basic-offset: 4 -*-
#ifndef MULTIOP_MPEG2_STREAMINGPROXYTABLE_HH
#define MULTIOP_MPEG2_STREAMINGPROXYTABLE_HH
#include <click/glue.hh>
#include <click/element.hh>

#include "../common/PF_FW.hh"
#include "../libtable/TableLinear.hh"

#include "FlowMpeg2AdaptEncap.hh"
#include "FlowMpeg2AdaptDecap.hh"

#include "Mpeg2Common.hh"



CLICK_DECLS

/* for debugging and statistics maintaining */
struct PFMpeg2AdaptFWStat {
	/* how many packets comming in */
	int pkts_in;
	/* how many packets going out */
	int pkts_out;
	/* how many packets killed */
	int pkts_killed;
};

class PFMpeg2AdaptFW : public PFFW { public:

	PFMpeg2AdaptFW();
	~PFMpeg2AdaptFW();

	int initialize(ErrorHandler*);
	void run_timer(Timer*);

	void* cast(const char*);
	int configure(Vector<String>&, ErrorHandler*);

	void push(int port, Packet* p);

	static String table_handler(Element*, void*);

	const char *class_name() const	{ return "PFMpeg2AdaptFW"; }
	const char *port_count() const	{ return "1/1"; }
	const char *processing() const	{ return PUSH; }

	virtual void dump();

	VcFlowClassify				*vcClassify;
        VcTableLinear<FlowMpeg2AdaptEncap>      *vcTableEncap;
        VcTableLinear<FlowMpeg2AdaptDecap>      *vcTableDecap;

	String mpeg2Name;

	uint32_t opParseMode;

protected:
private:

	Timer expire_timer;

	int	flow_expire_interval;
	int	flow_expiration_age;


public:
	/* operation mode */
	int	doWeEncap;

	/* stats for proxy table */
	struct PFMpeg2AdaptFWStat stat;	
};

CLICK_ENDDECLS
#endif
