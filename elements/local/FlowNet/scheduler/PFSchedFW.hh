// -*- c-basic-offset: 4 -*-
#ifndef MULTIOP_FLOWTABLE_HH
#define MULTIOP_FLOWTABLE_HH
#include <click/glue.hh>
#include <click/element.hh>
#include <click/timer.hh>
#include <click/notifier.hh>

#include "../common/Table.hh"
#include "../common/PF_FW.hh"

#include "../libtable/TableLinear.hh"
#include "FlowSchedulable.hh"


CLICK_DECLS

/*! 
 * For debugging and statistics maintaining of PFSchedFW. 
 * The member variables are supposed to be accessed in public.
 */
struct PFSchedFWStat {
	/*! How many packets comming in */
	uint32_t pkts_in;
	/*! How many packets going out */
	uint32_t pkts_out;
	/*! How many packets killed */
	uint32_t pkts_killed;

#ifdef ENABLE_DEBUG
	/*! function statistics: time of pull function called */
	struct timeval tv_pull;
	/*! function statistics: total of inter pull call interval  */
	uint32_t inter_pull_call_interval_total; 
	/*! function statistics: count of inter pull call  */
	uint32_t inter_pull_call_count;
	/*! function statistics: inter pull stalled  */
	uint32_t inter_pull_stalled;
#endif
	/*! function statistics for pull: pull timestamp */
	void fnst_pull_stamp(); 
	/*! function statistics for pull: pull stalled */
	void fnst_pull_stall(); 
	/*! function statistics for pull: pull timestamp reset */
	void fnst_pull_reset(); 
	/*! function statistics for pull: average pull called interval */
	double fnst_avg_pull_call(); 

#ifdef ENABLE_DEBUG
	/*! Debugging for damping timer accuracy: timestamp of damp start */
	struct timeval tv_damp_start;
	/*! Debugging for damping timer accuracy: microsecond time after damp */
	int32_t damp_after_usec;
	/*! Debugging for damping timer accuracy: microsecond time of damp error */
	int32_t damp_error_usec;
	/*! Debugging for damping timer accuracy: count of damp errors */
	uint32_t damp_error_count;
	/*! Debugging for damping timer accuracy: count of damp sleep */
	uint32_t damp_sleep_count;
#endif
	/*! function statistics for damp: damp start */
	void fnst_damp_start(int32_t usec);
	/*! function statistics for damp: damp end */
	void fnst_damp_end();
	/*! function statistics for damp: damp reset */
	void fnst_damp_reset();
	/*! function statistics for damp: average damp error */
	double fnst_avg_damp_error();
};


class VcScheduleOuter;

#define FLOW_EXPIRE_TIMER_INTERVAL 1000

/*! 
 * PFSchedFW class performs the following.
 */
class PFSchedFW : public PFFW { 

public:
	PFSchedFW();
	~PFSchedFW() {};
	
	/*
	 * click-element related functions
	 */
	int initialize(ErrorHandler*);
	void run_timer(Timer*);
	const char *class_name() const	{ return "PFSchedFW"; }
	const char *port_count() const	{ return "2/1"; }
	const char *processing() const	{ return "h/lh"; }
	void* cast(const char*);
	int configure(Vector<String>&, ErrorHandler*);

	void push(int port, Packet* p);
	Packet* pull(int);
	static String table_handler(Element*, void*);
	int check_output_ports();

	void cleanup(CleanupStage);

	virtual void dump();

	void outer_loop_expire();

	/* 
	 * The algorithms that are actually used
	 */
	VcTableLinear<FlowSchedulable>		*_vcTable;
	VcFlowClassify				*_vcClassify;
	VcSchedule				*_vcSched;
	VcScheduleOuter				*_vcSchedOuter;

private:

	/* 
	 * create algorithm objects and register them to 
	 * algorithm lists
	 */
	void create_and_register_algorithms();
	
	/*
	 * algorithm object lists
	 */
	Vector<VcFlowClassify *>		_classification_algorithms;
	Vector<VcSchedule *>			_scheduling_algorithms;
	Vector<VcScheduleOuter *>		_scheduling_outer_algorithms;


	/* In FlowNet-v0.6.0, we maintain multiple CDs and
	 * deal with multiple shapers that could depend on
	 * difference CDs */
	Vector<VcCongestionDetection *>		_cd_algorithms;

	Timer _flow_expire_timer;
	Timer _outer_loop_timer;


private: 
	/* 
	 * this is just for temporary storage for
	 * passing the arguments received from 
	 * configure function  and  store them 
	 * for a while and initiate the algorithms  
	 */

	/* TODO: Make this as structure and 
	 * allocate them for schedulers */
	double _wbs_beta;
	double _wbs_gamma;

	double _sbb_alpha;

private:
	/* 
	 * Virtual Schedule Slot Implementation
	 *
	 * In a node, a packet can be sent in virtual slots
	 * TODO: Write more
	 */ 
	#define VSS_SLEEP_THRESHOLD 1000 /* 1 milli second */
	
	int32_t _slot_unit;	/* micro-second */
	struct timeval _tv_last_schedule;
	bool _is_wakertimer_ticking;
	Timer _sleep_waker_timer;

	void init_virtual_schedule_slot(uint32_t sl);
	void timestamp_schedule();
	int32_t do_we_sleep();
	int32_t do_sleep(int32_t t);
	void wake_sleep();
public:
	int32_t get_slot_unit();
	int32_t add_slot_unit(int32_t add);


public: 
	/* 
	 * this is part of Virtual Schedule Slot Implementation
	 * and to support inner-loop schedulers use this function to sleep 
	 */
	void turnon_waker(int32_t t);
	

public:
	void listen_promisc( Packet* p );

	void put_qlen_to_FO( Packet* p, FlowSchedulable* flow );
	void setup_iptos( Packet* p, uint8_t tos );

	
	/* table wrapper */
	inline Flow* getFlowByFlowID( const FlowID* fid );

	inline VcCongestionDetection* getCDAlgorithm(const char *);
	inline VcSchedule* getScheduleAlgorithm();
	

public:
	/*! to support bind-monitoring that collects the number of drop counts
	 * this variable should be included into PFSchedFWStat for the consistency */
	int _drp_count;

	/*! to support bind-monitoring that collects the number of queue length
	 * this variable should be included into PFSchedFWStat for the consistency */
	int _qln_count;


private:
	/*! 
	 * to notify the below elements (to "ToDevice" element) 
	 */
	ActiveNotifier _empty_note;
	/*!
	 * Threshold that controls pull function call frequency.
	 * Once empty queue happens until SLEEP_TRIGGER counts,
	 * then pull function goes to sleep
	 */
	#define SLEEP_TRIGGER 9
	/*!
	 * A counter that counts sleepiness. It is compared to SLEEP_TRIGGER
	 * to detemine whether going to sleep or not. 
	 */
	int	_sleepiness;
	
	/*! 
	 * for maintaining statistics 
	 */
	struct PFSchedFWStat _stat;

};

inline VcCongestionDetection* PFSchedFW::getCDAlgorithm(const char *cdName)
{
	for( int i = 0; i<_cd_algorithms.size(); i++ ) {
		if( strcmp( _cd_algorithms[i]->name(), cdName ) == 0 ) {
			return _cd_algorithms[i];
		}
	}
	return NULL;
}
inline VcSchedule* PFSchedFW::getScheduleAlgorithm()
{
	return _vcSched;
}

inline Flow* PFSchedFW::getFlowByFlowID( const FlowID* fid )
{
	Flow* f;
	_vcTable->lookup( fid, (FlowSchedulable**)&f );
	return f;
}
CLICK_ENDDECLS
#endif
