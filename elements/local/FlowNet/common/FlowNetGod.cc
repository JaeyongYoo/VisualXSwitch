// -*- c-basic-offset: 4 -*-
/*
 * 
 * Jae-Yong Yoo
 *
 * Copyright (c) 2010 Gwangju Institute of Science and Technology, Korea
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, subject to the conditions listed in the Click LICENSE
 * file. These conditions include: you must preserve this copyright
 * notice, and you cannot mention the copyright holders in advertising
 * related to the Software without their permission.  The Software is
 * provided WITHOUT ANY WARRANTY, EXPRESS OR IMPLIED. This notice is a
 * summary of the Click LICENSE file; the license in that file is
 * legally binding.
 */
#include <click/config.h>
#include <click/confparse.hh>
#include <click/error.hh>

#include <pthread.h>
#include <arpa/inet.h>
#include <unistd.h>

#include "Table.hh"
#include "FlowNetGod.hh"
#include "../scheduler/PFSchedFW.hh"

CLICK_DECLS

/*
 * jyyoo debugging : indenting stack depth 
 */
D_DEFINE_EXTERN;


/*
 * default parameter setting
 */
struct paramset_sourceproxytable ps_spt_default = { 
	5	/* ticks interval milli sec: 5 means 200 times per second */,
	10	/* rate update interval (unit: clock) */,
	250	/* maximum queue length */,
	100	/* queue length threshold (source side) */ ,
	SRCM_THRESHOLD /* source rate control method */,
	0.1	/* packets per clock */,
	0.1	/* increase factor (additive) */,
	2.0	/* decrease factor (multiplicative) */

};


struct paramset_mpeg2_streamingproxytable ps_m2sp_default = { 
	250	/* maximum queue length */
};


struct paramset_mpeg2_shapertable ps_m2s_default = {

        MPEG2_SHAPE_NO_SHAPING, /* shape method; */
        500, /* int no_loss_count_thresh; */
        50, /* int loss_count_thresh; */ 
        0.1 /* double loss_thresh; */
};



struct paramset_flowtable ps_ft_default = {
	1	/* enable Backpressure scheduling */,
	1	/* perfect source control assumption */,
	100	/* backpressure threshold (destination side) */,
	BVCM_QUEUE_ONLY /* backpressure value computation method */,
	200	/* maximum queue length */,
	0	/* post fix calculator policy */,
	"X"	/* post fix calculator for Q function (monitoring function) */,
		/* "X" is basic monitoring: X stands for my queue length */
	"X-Y"	/* post fix calculator for P function (priority function) */,
		/* "X-Y" is pure backpressure: Y stands for nexthop queue length */
	0	/* control vibration policy */,
	0.0	/* vibration frequency */,
	0.0	/* vibration amplitude */,
	0	/* random schedule policy */,
	1000	/* random schedule period */,
	0	/* queue length monitoring policy (sequencing enable option) */,
	0	/* nexthop queue length monitor statistical filtering (EWMA) option */,
	0.5	/* nexthop queue length monitor statistical filtering EWMA value */,
	1	/* bandwidth damper enable */,
	1	/* enable neighbor table */,
	1000	/* e2e_signal_interval */,
	0	/* e2e signal stable channel assumption */
};




FlowNetGod::FlowNetGod(): 
		pfsched(NULL), 
		dump_timer(this), 
		debug_level(0), 
		sched_pull_called_num(0) 
{
	current_pffw = 0;
}

/*
 * FlowNetGod member function bodies 
 */

int FlowNetGod::configure(Vector<String> &conf, ErrorHandler *errh)
{

	IPAddress my_mask;	
	bool have_postfix_calculator;
	bool have_control_vibration;
	bool have_random_schedule;
	bool have_qlen_monitor_sequence;
	bool have_qlen_monitor_ewma;
	bool have_threshold;
	bool have_perfect_src_control;
	bool have_debug_level;
	bool have_dump_print_clear;
	bool have_bw_damper;
	bool have_neighbor_table;
	bool have_e2e_signal_interval;
	bool have_e2e_signal_stable;
	bool have_loss_count_thresh;
	bool have_no_loss_count_thresh;
	bool have_loss_thresh;
	bool have_shape_method;

        IPAddress monServerIP;

	/* set to default values */
	ps_flowtable = ps_ft_default;
	ps_sourceproxy = ps_spt_default;
	ps_mpeg2_streamingproxy = ps_m2sp_default;
	ps_mpeg2_shaper = ps_m2s_default;

	/* change user-supplied options */
	if (cp_va_kparse(conf, this, errh,
				/* device parameter setting */
				"ETH", cpkP, cpEthernetAddress, &(ps_device.myEther),
				"IP", cpkP, cpIPAddressOrPrefix, &(ps_device.myIP), &my_mask,
				"WIRELESS_IFNAME", cpkP, cpString, &(ps_device.myWirelessIfname),
					
				/* papmo enable option */
				"ENABLE_PAPMO", cpkP, cpInteger, &enable_papmo,
				"MONSERVERIP", cpkP, cpIPAddress, &(monServerIP),
				
				/* flow table parameter setting */
                                "ENABLE_BACKPRESSURE", cpkP, cpInteger, &(ps_flowtable.enable_BP),
				"BACKPRESSURE_VALUE_COMPUTATION_METHOD", cpkP, cpInteger, &(ps_flowtable.backpressure_value_computation_method),
                                "FUNCTION_Q", cpkP, cpString, &(ps_flowtable.strFunctionQ),
                                "FUNCTION_P", cpkP, cpString, &(ps_flowtable.strFunctionP),
                                "CONTROL_FREQUENCY", cpkP, cpDouble, &(ps_flowtable.vibration_frequency),
                                "CONTROL_AMPLITUDE", cpkP, cpDouble, &(ps_flowtable.vibration_amplitude),
                                "RANDOM_SCHEDULE_PERIOD", cpkP, cpInteger, &(ps_flowtable.random_schedule_period),
                                "QLEN_MONITOR_EWMA_ALPHA", cpkP, cpDouble, &(ps_flowtable.qlen_monitor_ewma_alpha),
				"FLOW_MQLEN", cpkP, cpInteger, &(ps_flowtable.max_qlen),

				/* source proxy table parameter setting */
				"TICKS_INTERVAL_MSEC", cpkP, cpInteger, &(ps_sourceproxy.ticks_interval_msec),
				"RATE_UPDATE_INTERVAL", cpkP, cpInteger, &(ps_sourceproxy.ru_interval),
				"PKTS_PER_CLOCK", cpkP, cpDouble, &(ps_sourceproxy.pkts_per_clock),
				"SOURCE_RATE_CONTROL_METHOD", cpkP, cpInteger, &(ps_sourceproxy.source_rate_control_method),
				"INCREASE_FACTOR", cpkP, cpDouble, &(ps_sourceproxy.increase_factor),
				"DECREASE_FACTOR", cpkP, cpDouble, &(ps_sourceproxy.decrease_factor),
				"SRCPROXY_MQLEN", cpkP, cpInteger, &(ps_sourceproxy.max_qlen),
				"SRCPROXY_THRESH_QLEN", cpkP, cpInteger, &(ps_sourceproxy.thresh_qlen),

				/* flow table policy parameter setting */
				"ENABLE_CALC", cpkC, &have_postfix_calculator, cpUnsigned, &(ps_flowtable.enable_postfix_calculator),
				"ENABLE_STATIC_VIBRATION", cpkC, &have_control_vibration, cpUnsigned, &(ps_flowtable.enable_control_vibration),
				"ENABLE_RANDOM_SCHEDULE", cpkC, &have_random_schedule, cpUnsigned, &(ps_flowtable.enable_random_schedule),
				"ENABLE_QLEN_MONITOR_CHECKSEQ", cpkC, &have_qlen_monitor_sequence, cpUnsigned, &(ps_flowtable.enable_qlen_monitor_sequence),
				"ENABLE_QLEN_MONITOR_EWMA", cpkC, &have_qlen_monitor_ewma, cpUnsigned, &(ps_flowtable.enable_qlen_monitor_ewma),
				"PERFECT_SOURCE_RATE_CONTROL", cpkC, &have_perfect_src_control, cpInteger, &(ps_flowtable.enable_perfect_source_rate_control),
				"BP_QUEUE_THRESHOLD", cpkC, &have_threshold, cpUnsigned, &(ps_flowtable.BP_queue_threshold),
				"ENABLE_BANDWIDTH_DAMPER", cpkC, &have_bw_damper, cpUnsigned, &(ps_flowtable.enable_bandwidth_damper),
				"ENABLE_NEIGHBOR_TABLE", cpkC, &have_neighbor_table, cpUnsigned, &(ps_flowtable.enable_neighbor_table),

				/* loss-rate detection based video srouce rate control */
				"SHAPE_METHOD", cpkC, &have_shape_method, cpUnsigned, &(ps_mpeg2_shaper.shape_method),
				"VS_LRD_NO_LOSS_COUNT_THRESH", cpkC, &have_no_loss_count_thresh, cpUnsigned, &(ps_mpeg2_shaper.no_loss_count_threshold),
				"VS_LRD_LOSS_COUNT_THRESH", cpkC, &have_loss_count_thresh, cpUnsigned, &(ps_mpeg2_shaper.loss_count_threshold),
				"VS_LRD_LOSS_THRESH", cpkC, &have_loss_thresh, cpDouble, &(ps_mpeg2_shaper.loss_threshold),

				/* queue monitoring using ip id field */
		
				/* e2e signalling for flow-connectivity management */
				"E2E_SIGNAL_INTERVAL", cpkC, &have_e2e_signal_interval, cpUnsigned, &(ps_flowtable.e2e_signal_interval),
				/* stable e2e signal channel assumption */
				"E2E_SIGNAL_STABLE", cpkC, &have_e2e_signal_stable, cpUnsigned, &(ps_flowtable.e2e_signal_stable),

				/* debugging related setting */
				"DEBUG_LEVEL", cpkC, &have_debug_level, cpUnsigned, &debug_level,
				"DUMP_PRINT_CLEAR", cpkC, &have_dump_print_clear, cpUnsigned, &dump_print_clear,

				cpEnd) < 0)
		return -1;


	if( enable_papmo )
	{
		papmo.init(1000, monServerIP );
	}



	if( debug_code( DL_WELCOME) )
	{
		welcome_message();
	}
	return 0;
}

int
FlowNetGod::initialize(ErrorHandler*)
{
	int res=0;
	dump_timer.initialize(this);
	if( BASE_DUMP_INTERVAL ) dump_timer.schedule_after_msec(BASE_DUMP_INTERVAL);

	return res;
}

int FlowNetGod::accept_this_packet( click_ip* ip )
{
	if(     ip->ip_p != IP_PROTO_UDP &&
		ip->ip_p != IP_PROTO_TCP )
		return 1;
	return 0;

}


void FlowNetGod::register_pffw( PFFW* pffw )
{
	if( current_pffw < MAX_PFFW )
	{
		vcTable[current_pffw] = pffw;
		current_pffw ++;
	}
}
void FlowNetGod::welcome_message()
{
	click_chatter("##############################################################\n");	
	click_chatter("# Welcome to FlowNet\n");
	click_chatter("##############################################################\n");	
	click_chatter("FlowNetGod Configuration Status\n");

	/*
	 * print device information
	 */
	click_chatter("\tDevice Info:\n");
	click_chatter("\t\tIP address: %d.%d.%d.%d\n",
			*(ps_device.myIP.data() ), 
			*(ps_device.myIP.data()+1), 
			*(ps_device.myIP.data()+2), 
			*(ps_device.myIP.data()+3) );
	

	click_chatter("\t\tMAC address: %x:%x:%x:%x:%x:%x\n",
			((unsigned char)*(ps_device.myEther.data()+0) ), 
			((unsigned char)*(ps_device.myEther.data()+1) ), 
			((unsigned char)*(ps_device.myEther.data()+2) ), 
			((unsigned char)*(ps_device.myEther.data()+3) ), 
			((unsigned char)*(ps_device.myEther.data()+4) ), 
			((unsigned char)*(ps_device.myEther.data()+5) ) );

	click_chatter("\n");
	/*
	 * print per-flow table information
	 */
	click_chatter("\tFlowTableInfo:\n");
	click_chatter("\t\tMax Flow Number: Unknown\n");
	click_chatter("\t\tMax Queue Length: %d\n", ps_flowtable.max_qlen);
	click_chatter("\t\tBackpressure Threshold: %d\n", ps_flowtable.BP_queue_threshold);
	click_chatter("\t\tOptional Parameters\n");
	click_chatter("\t\tEnable Backpressure: %s\n", ps_flowtable.enable_BP ? "ON" : "OFF" );


	click_chatter("\t\tBackpressure value computation method: ");
	if( ps_flowtable.backpressure_value_computation_method == BVCM_QUEUE_ONLY )
		click_chatter("QUEUE_ONLY\n");
	if( ps_flowtable.backpressure_value_computation_method == BVCM_QUEUE_AND_LINK )
		click_chatter("QUEUE AND LINK\n");
	if( ps_flowtable.backpressure_value_computation_method == BVCM_SUPPLEMENTARY_PRESSURE )
		click_chatter("SUPPLEMENTARY_PRESSURE\n");

	if( ps_flowtable.enable_BP )
	{
		click_chatter("\t\tStatic Vibration: %s\n", ps_flowtable.enable_control_vibration ? "ON" : "OFF" );
		if( ps_flowtable.enable_control_vibration )
			click_chatter("\t\t\tControl Amplitude: %f\n", ps_flowtable.vibration_amplitude);	
		if( ps_flowtable.enable_control_vibration )
			click_chatter("\t\t\tControl Frequency: %f\n", ps_flowtable.vibration_frequency);	
		click_chatter("\t\tRandom Schedule: %s\n", ps_flowtable.enable_random_schedule ? "ON" : "OFF" );
		if( ps_flowtable.enable_random_schedule )
			click_chatter("\t\t\tRandom Schedule Period: %d packets\n", ps_flowtable.random_schedule_period);	
		click_chatter("\t\tQueue Monitoring Sequence: %s\n", ps_flowtable.enable_qlen_monitor_sequence ? "ON" : "OFF" );
		click_chatter("\t\tQueue Monitoring Smoothing: %s\n", ps_flowtable.enable_qlen_monitor_ewma ? "ON" : "OFF" );
		if( ps_flowtable.enable_qlen_monitor_ewma )
			click_chatter("\t\t\tSmoothing EWMA alpha:%f\n", ps_flowtable.qlen_monitor_ewma_alpha);
		click_chatter("\t\tPostfix Calculator: %s\n", ps_flowtable.enable_postfix_calculator ? "ON" : "OFF" );
		if( ps_flowtable.enable_postfix_calculator ) {
			click_chatter("\t\t\tQ = %s\n", ps_flowtable.strFunctionQ.c_str() );
			click_chatter("\t\t\tP = %s\n", ps_flowtable.strFunctionP.c_str() );
		}
		click_chatter("\t\tPerfect Source Rate Control Assumption: %s\n", ps_flowtable.enable_perfect_source_rate_control ? "ON" : "OFF" );
		click_chatter("\t\tEnable Bandwidth Damper: %s\n", ps_flowtable.enable_bandwidth_damper ? "ON" : "OFF" );
		click_chatter("\t\tEnable Neighbor Table: %s\n", ps_flowtable.enable_neighbor_table ? "ON" : "OFF" );
		
	}
	/*
	 * print source proxy information 
	 */
	click_chatter("\n");
	click_chatter("\tSourceProxyInfo:\n");
	click_chatter("\t\tMax Source Proxy Number: Unknown\n");
	click_chatter("\t\tTicks Interval: %d milli seconds\n", ps_sourceproxy.ticks_interval_msec);
	click_chatter("\t\tRate Update Interval: %d clock\n", ps_sourceproxy.ru_interval);
	click_chatter("\t\tPackets per clock: %f packets\n", ps_sourceproxy.pkts_per_clock);
	click_chatter("\t\tIncrease factor: %f \n", ps_sourceproxy.increase_factor);
	click_chatter("\t\tDecrease factor: %f \n", ps_sourceproxy.decrease_factor);
	click_chatter("\t\tMaximum queue length: %d packets\n", ps_sourceproxy.max_qlen);
	click_chatter("\t\tQueue threshold: %d packets\n", ps_sourceproxy.thresh_qlen);
	click_chatter("\n");

	click_chatter("\tMpeg2StreamingProxyInfo:\n");
	click_chatter("\t\tMax Source Proxy Number: Unknown\n");
	click_chatter("\t\tMaximum queue length: %d packets\n", ps_mpeg2_streamingproxy.max_qlen);
	click_chatter("\n\n");

}

/* timer based report function */
void FlowNetGod::run_timer(Timer* t)
{
	static int c=0;
	c++;
	if( t == &dump_timer )
	{
		/* just for pretty debug output */
		if( dump_print_clear )
		{
			int re;
			re=system("clear");
		}
		printf("FlowNet Table Monitor [");
		switch( c % 4 ) 
		{
			case 0: printf("-");break;
			case 1: printf("\\");break;
			case 2: printf("|");break;
			case 3: printf("/");break;
			default: break;
		}
		printf("] : clear ON\n");

		for( int i = 0; i<current_pffw; i++ )
		{
			vcTable[i]->dump();
		}
	
		t->schedule_after_msec( BASE_DUMP_INTERVAL );
	}
}

void FlowNetGod::debug_message(int level, const char* fmt, ...)
{
	if( level > debug_level ) return;

	int n, size = 1024;
	char buf[1024];
	char sz_myip[20];
	va_list ap;

	va_start(ap, fmt);
	n = vsnprintf(buf, size, fmt, ap);
	va_end(ap);
	strcpy( sz_myip, inet_ntoa(ps_device.myIP.in_addr()));
	printf("Node:%s DebugLevel(%d): %s", sz_myip, debug_level, buf);
}
int FlowNetGod::debug_code(int level)
{
	if( level > debug_level ) return 0;
	return 1;
}

/**********************************
 * ControlSocket Handlers
 *********************************/
enum {H_W_PFSCHEDFW, H_W_RRSCHED, H_W_SOURCEPROXY, H_W_E2E_SIGNAL_GEN, H_W_NEIGH_TABLE, H_W_DEST_STUB};

int write_param_FlowNetGod(const String &in_s, Element *e, void *vparam, ErrorHandler *errh)
{
	FlowNetGod* mom = (FlowNetGod*)e;
	Element *element;

	switch( (int)vparam )
	{
		case H_W_PFSCHEDFW:
			{
				Vector<String> args;
				cp_spacevec(in_s, args);

				/*int res = cp_va_parse( in_s, mom, errh, cpElement, "MultiOpFlowTable", &element, 0);*/
				int res = cp_va_kparse( in_s, mom, errh, "PFSchedFW", cpkP, cpElement,  &element, cpEnd);
				if( res < 0 ) { return res; }
				if( strcmp(element->class_name(),  "PFSchedFW") != 0 ) {
					printf("Class name is not \"PFSchedFW\"" );
					return -1;
				}
				mom->pfsched = (PFSchedFW*)element;
				break;
			}

		default:
			break;
	}
	return 0;
}

void FlowNetGod::add_handlers()
{
	add_write_handler("PFSchedFW", write_param_FlowNetGod, (void*)H_W_PFSCHEDFW);
}

CLICK_ENDDECLS
EXPORT_ELEMENT(FlowNetGod)
