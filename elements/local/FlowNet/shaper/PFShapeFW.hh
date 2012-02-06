// -*- c-basic-offset: 4 -*-
#ifndef MULTIOP_SOURCEPROXYTABLE_HH
#define MULTIOP_SOURCEPROXYTABLE_HH
#include <click/glue.hh>
#include <click/element.hh>
#include <click/timer.hh>


#include "../common/PF_FW.hh"
#include "../libtable/TableLinear.hh"

#include "FlowBWShaperable.hh"


CLICK_DECLS

/* for debugging and statistics maintaining */
struct PFShapeFWStat {
	/* how many packets comming in */
	int pkts_in;
	/* how many packets going out */
	int pkts_out;
	/* how many packets killed */
	int pkts_killed;
};

#define SEND_TIMER_CLOCK 100

class PFShapeFW : public PFFW { public:

	PFShapeFW();
	~PFShapeFW();

	int initialize(ErrorHandler*);
	void run_timer(Timer*);

	void* cast(const char*);
	int configure(Vector<String>&, ErrorHandler*);
	void cleanup(CleanupStage);

	static int write_param(const String &s, Element *e, void *, ErrorHandler *errh);
	static String read_param(Element *e, void *);
	void add_handlers();

	virtual void dump();

	void push(int i, Packet* p);
	void source_send();

	void change_rate( int );
	int get_rate();

	const char *class_name() const	{ return "PFShapeFW"; }
	const char *port_count() const	{ return "1/1"; }
	const char *processing() const	{ return PUSH; }


        /* 
         * The algorithms that are actually used
         */
        VcTableLinear<FlowBWShaperable>         *_vcTable;
        VcFlowClassify                          *_vcClassify;
	VcBWShape				*_vcShape;


private:

        /* 
         * create algorithm objects and register them to 
         * algorithm lists
         */
        void create_and_register_algorithms();

        /*
         * algorithm object lists
         */
        Vector<VcFlowClassify *>                _classification_algorithms;
	Vector<VcBWShape *>			_shaper_algorithms;

private:
	/*
	 * if this is binded to CD, it can use 
	 * congestion-detection-based rate shaping 
	 * algorithms from the congestion callback
	 * from the scheduler (especially by CD
	 * algorithms)
	 */
        bool _is_binding_to_cd;
	String _str_bind_to_CD;


public:

	inline VcBWShape* getShaperAlgorithm();
	bool _turnoff_timer;
	int _mpegShape;
	bool _have_mpegShape;
	String _shape_algorithm;
	String _classify_algorithm;

	/*
	 * wrapper functions of vcTable 
	 */
	FlowBWShaperable* lookupflow( const Packet* p );
	FlowBWShaperable* lookupflow( struct FlowID* fid );
	FlowBWShaperable* getAt(int i);
	int size();

protected:
private:

	Timer send_timer;
	Timer expire_timer;


public:
	
//	/* parameters from MIB */
//	struct paramset_sourceproxytable *ps_sourceproxy;

	/* stats for proxy table */
	struct PFShapeFWStat stat;	
};

inline VcBWShape* PFShapeFW::getShaperAlgorithm()
{
	return _vcShape;
}
CLICK_ENDDECLS
#endif
