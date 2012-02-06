#ifndef MULTIOP_FlowNetGod_HH
#define MULTIOP_FlowNetGod_HH
#include <click/element.hh>
#include <click/etheraddress.hh>
#include <click/timer.hh>
#include <clicknet/ether.h>

#include "FlowCommon.hh"
#include "Table.hh"

#include "../papmo/PaPMo.hh"

CLICK_DECLS

#define SECOND_LEVEL_LOOKUP_CACHE 1

#define DL_ERROR		-5
#define DL_WELCOME		-1
#define DL_WARNING		0
#define DL_PERIODIC_MESSAGE	1
#define DL_PER_QUEUE 		2
#define DL_PER_PACKET		3
#define DL_INSIDE_PACKET	4

#define DL_FUNCTION_ENTRY	5
#define DL_STEP_BY_STEP		6


class PFSchedFW;
class PFShapeFW;
class PFFW;


#define BASE_DUMP_INTERVAL	1000 /* basic dump interval is 1 second */
#define MAX_TABLE_ELEMENT	20

/*
 * parameter set for MultiOpMpeg2StreamingProxyTable element
 */
#define MPEG2_SHAPE_NO_SHAPING	0
#define MPEG2_SHAPE_STATIC_I	1
#define MPEG2_SHAPE_STATIC_IP	2
#define MPEG2_SHAPE_LOSS_BASED	3
#define MPEG2_SHAPE_CORE_BASED	4
struct paramset_mpeg2_shapertable {

	/* for choosing which shaping method */
	int shape_method;	

	/* for loss-rate detection based video source rate control */
	int no_loss_count_threshold;
	int loss_count_threshold;
	double loss_threshold;
};


/*
 * parameter set for MultiOpMpeg2StreamingProxyTable element
 */
struct paramset_mpeg2_streamingproxytable {

	/* maximum queue length */
	int max_qlen;
};

#define SRCM_MIMD 0
#define SRCM_THRESHOLD 1
#define SRCM_NO_CONTROL 2

/*
 * parameter set for MultiOpSourceProxyTable element
 */
struct paramset_sourceproxytable {

	/*! clock speed */
	int ticks_interval_msec;

	/*! rate update interval */
	int ru_interval;

	/*! maximum queue length */
	int max_qlen;

	/*! rate-control queue threshold */
	int thresh_qlen;

	/*! source rate control method */ 
	int source_rate_control_method;

	/* source rate control by MIMD */

	/*! packets per clock (basically sending rate) */
	double pkts_per_clock;

	/*! rate increase factor */
	double increase_factor;

	/*! rate decrease factor */
	double decrease_factor;


};


#define BVCM_QUEUE_ONLY 0
#define BVCM_QUEUE_AND_LINK 1
#define BVCM_SUPPLEMENTARY_PRESSURE 2

#define BACKPRESSURE_QUANTIZATION_STEP 30
/*
 * parameter set for MultiOpSourceProxyTable element
 */
struct paramset_flowtable {

	/* enable backpressure scheduling */
	int enable_BP;

	/* perfect source rate control assumption */
	int enable_perfect_source_rate_control;

	/* default backpressure threshold (for destination) */
        int BP_queue_threshold;

	/* backpressure value computation method (queue length only or queue length and link quality) */
	int backpressure_value_computation_method;

	/* maximum queue length */
	int	max_qlen;	

	/* queue length propagation policy */
	int	enable_postfix_calculator;
	String	strFunctionQ;
	String	strFunctionP;

	/* control periodicity (insertion of random noise) */
	int	enable_control_vibration;
	double	vibration_frequency;
	double	vibration_amplitude;

	/* random schedule policy (periodic random schedule) */
	int	enable_random_schedule;
	int	random_schedule_period;

	/* queue length monitoring (sequence check) */
	int	enable_qlen_monitor_sequence;

	/* queue length monitoring (EWMA) */
	int	enable_qlen_monitor_ewma;
	double	qlen_monitor_ewma_alpha;

	/* bandwidth damper */
	int	enable_bandwidth_damper;
	
	/* neighbor table */
	int	enable_neighbor_table;

	/* e2e_signal_interval */
	int	e2e_signal_interval;
	
	/* e2e signal stable channel assumption */
	int	e2e_signal_stable;
};

/*
 * parameter set for device
 */
struct paramset_device {
        /* this device IP address */
	IPAddress myIP;
        /* this device MAC address */
	EtherAddress myEther;
	/* wireless interface name */
	String myWirelessIfname;
};

#define MAX_PFFW 100

class FlowNetGod : public Element {

public:

	FlowNetGod();
	~FlowNetGod() {};

	const char *class_name() const { return "FlowNetGod"; }
	const char *processing() const { return AGNOSTIC; }
	int configure(Vector<String> &, ErrorHandler *);
	int initialize(ErrorHandler*);


	// handlers for this element
	void add_handlers();

	/* pointers of main objects for cross-reference */
	PFSchedFW			*pfsched;
	PFShapeFW			*pfshape;


	void register_pffw( PFFW* pffw );

	PFFW*		vcTable[MAX_PFFW];
	int	current_pffw;


	/* option for papmo monitoring */
	int enable_papmo;
	struct papmo papmo;

	/* dump timer */
	Timer dump_timer;
/*
	MultiOpTable*	tableOfTable[MAX_TABLE_ELEMENT];
	int		interval[MAX_TABLE_ELEMENT];
	char		nameOfTable[MAX_TABLE_ELEMENT][64];
	int		total_tableOfTable;
*/

public: /* for global packet filtering */
	int accept_this_packet( click_ip* ip );
	
public: /* various parameters and infos */
	
	/* parameters for this device */
	struct paramset_device ps_device;
	
	/* parameters for MultiOpFlowTable */
	struct paramset_flowtable ps_flowtable;

	/* parameters for MultiOpSourceProxy */
	struct paramset_sourceproxytable ps_sourceproxy;

	/* parameters for MultiOpMpeg2StreamingProxy */
	struct paramset_mpeg2_streamingproxytable ps_mpeg2_streamingproxy;

	/* parameters for MultiOpMpeg2Shaper */
	struct paramset_mpeg2_shapertable ps_mpeg2_shaper;

	
public: /* for debugging support */

	/* welcome message */
	void welcome_message();

	/* dumping function */
	void run_timer(Timer*);

//	void register_table_tobe_dump( MultiOpTable* table, int fi, char* name );

	/* universal debug message interface */
	void debug_message(int debug_level, const char* fmt, ...);
	int debug_code(int debug_level);

	/* universal debug level */
	int	debug_level;

	/* do system("clear") while dumping messages */
	int	dump_print_clear;

	/* function statistics */
	int sched_pull_called_num;
	int sched_pull_called_num_nopacket;

};

CLICK_ENDDECLS
#endif
