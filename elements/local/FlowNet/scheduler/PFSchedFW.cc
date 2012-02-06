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
#include <click/ipaddress.hh>
#include <click/confparse.hh>
#include <click/error.hh>
#include <click/glue.hh>
#include <click/straccum.hh>
#include <click/router.hh>
#include <sys/time.h>
#include <clicknet/wifi.h>
#include <click/packet_anno.hh>

#include "PFSchedFW.hh"

/* for classification algorithm objects */
#include "../libclassify/ClassifyNoClassify.hh"
#include "../libclassify/Classify5Tuple.hh"
/* for table algorithm objects */
#include "../libtable/TableLinear.hh"
/* for scheduling algorithm objects */
#include "sched_algorithms/SchedNoSchedule.hh"
#include "sched_algorithms/SchedBackpressure.hh"
#include "sched_algorithms/SchedHorizon.hh"
#include "sched_algorithms/SchedWBS.hh"
#include "sched_algorithms/SchedSBB.hh"
#include "sched_algorithms/SchedSBBLogWeight.hh"
#include "sched_algorithms/SchedLogWeight.hh"
#include "sched_outer_algorithms/SchedOuterVirtualpressure.hh"
#include "sched_outer_algorithms/SchedOuterNoSchedule.hh"
/* for congestion detection (CD) algorithm objects */
#include "CD_algorithms/CD_CORE.hh"
#include "CD_algorithms/CD_QueueLoss.hh"
#include "CD_algorithms/CD_Threshold.hh"
#include "CD_algorithms/CD_Direct.hh"

/* for timevaldiff */
#include "../utils/FlowNetUtils.hh"

CLICK_DECLS

D_DEFINE_BODY;

PFSchedFW::PFSchedFW() : _flow_expire_timer(this), _outer_loop_timer(this), _sleep_waker_timer(this)
{ 
	_god = NULL;
        _vcClassify = NULL;
        _vcSched = NULL;
        _vcSchedOuter = NULL;
}

void * PFSchedFW::cast(const char *name)
{
	if (strcmp(name, "PFSchedFW") == 0)
		return (void *)this;
	else if(strcmp(name, Notifier::EMPTY_NOTIFIER) == 0 )
		return static_cast<Notifier *>(&_empty_note);
	else
		return Element::cast(name);
}

int PFSchedFW::configure(Vector<String> &conf, ErrorHandler *errh)
{
	D_START_FUNCTION;

	String classifyAlgorithm;
	String schedAlgorithm;
	String schedAlgorithmOuter;
	String cdAlgorithm;
	bool have_wbs_beta, have_wbs_gamma, have_sbb_alpha;

	int before = errh->nerrors();
	if( cp_va_kparse(conf, this, errh, 
				"GOD", cpkP, cpElement, &_god,
				"CLASSIFY_ALGORITHM", cpkP, cpString, &classifyAlgorithm,
                                "SCHEDULE_ALGORITHM", cpkP, cpString, &schedAlgorithm,
                                "SCHEDULE_OUTER_ALGORITHM", cpkP, cpString, &schedAlgorithmOuter,
				"VSS_SLOTUNIT", cpkP, cpInteger, &_slot_unit,
				"WBS_BETA", cpkC,  &have_wbs_beta, cpDouble, &_wbs_beta,
				"WBS_GAMMA", cpkC, &have_wbs_gamma, cpDouble, &_wbs_gamma,
				"SBB_ALPHA", cpkC, &have_sbb_alpha, cpDouble, &_sbb_alpha,
				//"EXPIRE_INTERVAL", cpkP, cpInteger, &flow_expire_interval,
				cpEnd) < 0 ) {
		D_END_FUNCTION;
		return -1;
	}

	/* 
	 * setup for scheduler WBS 
	 */
	if( have_wbs_beta == false ) {
		_wbs_beta = 1.6;
	} else {
	}
 
	if( have_wbs_gamma == false ) {
		_wbs_gamma = 1.2;
	} else {
	}

	if( have_sbb_alpha == false ) {
		_sbb_alpha = 1.0;
	} else {
	}
	click_chatter("SBB_ALPHA set: %x\n", _sbb_alpha);

	/* 
	 * register this framework to FlowNet God 
	 */
	_god->pfsched = this;
	_god->register_pffw( this );	

	/*
	 * create and register scheduling, 
	 * congestion detection, classification algorithms
	 */
	create_and_register_algorithms();

	/* TODO: Make this as CHOOSE_ALGORITHM macro */
	/* choose algorithms */
	for( int i = 0; i<_classification_algorithms.size(); i++ ) {
		if( strncmp( classifyAlgorithm.c_str(), 
				_classification_algorithms[i]->name(), 
				FLOWNET_ALGORITHM_NAME_SIZE ) == 0 ) {
				_vcClassify = _classification_algorithms[i];
			break;
		}
	}

	for( int i = 0; i<_scheduling_algorithms.size(); i++ ) {
		if( strncmp( schedAlgorithm.c_str(), 
				_scheduling_algorithms[i]->name(), 
				FLOWNET_ALGORITHM_NAME_SIZE ) == 0 ) {
				_vcSched = _scheduling_algorithms[i];
			break;
		}
	}

	for( int i = 0; i<_scheduling_outer_algorithms.size(); i++ ) {
		if( strncmp( schedAlgorithmOuter.c_str(), 
				_scheduling_outer_algorithms[i]->name(), 
				FLOWNET_ALGORITHM_NAME_SIZE ) == 0 ) {
				_vcSchedOuter = _scheduling_outer_algorithms[i];
			break;
		}
	}

	/* 
	 * TODO: if an algorithm is NULL, then print out the available algorithm list
	 * for the corresponding algorithm 
	 */
	if( _vcSched == NULL || 
		_vcClassify == NULL || _vcTable == NULL || _vcSchedOuter == NULL) {
		click_chatter("Error! You have to specify all algorithms\n");
		click_chatter("[%s]_vcClassify == %x\n",classifyAlgorithm.c_str(),  _vcSched);
		click_chatter("[%s]_vcSchedOuter == %x\n",schedAlgorithmOuter.c_str(),  _vcSchedOuter);
		click_chatter("_vcTable == %x\n", _vcTable);
		return -1;
	}

	/*
	 * bind outer scheduling and inner scheduling to work interactively together 
	 */
	if( _vcSchedOuter->bind( _vcSched, this ) ) {
		click_chatter("Error! Outer Scheduling bind failed to inner scheduling\n");
		return -1;
	}

	D_END_FUNCTION;
	return (errh->nerrors() != before ? -1 : 0);
}

void PFSchedFW::create_and_register_algorithms()
{
	/*
	 * create and register algorithms 
	 */
	Vc5TupleClassify *vc5TupleClassify;
	VcNoClassify *vcNoClassify;
	VcScheduleBackpressure *vcSchedBackpressure;
	VcScheduleNoSchedule *vcSchedNoSchedule;
	VcScheduleHorizon *vcSchedHorizon;
	VcScheduleWBS *vcSchedWBS;
	VcScheduleSBB *vcSchedSBB;
	VcScheduleSBBLogWeight *vcSchedSBBLogWeight;
	VcScheduleLogWeight *vcSchedLW;
	VcScheduleOuterVirtualpressure *vcSchedOuterVP;
	VcScheduleOuterNoSchedule *vcSchedOuterNo;
	VcCDCORE *vcCDCORE;
	VcCDQueueLoss *vcCDQueueLoss;
	VcCDThreshold *vcCDThreshold;
	VcCDDirect *vcCDDirect;

	/* register classification algorithms */
	vc5TupleClassify = new Vc5TupleClassify();
	vcNoClassify = new VcNoClassify();

	_classification_algorithms.push_back( vc5TupleClassify );
	_classification_algorithms.push_back( vcNoClassify );

	/* register backpressure scheduling algorithms (Inner) */
	vcSchedBackpressure = new VcScheduleBackpressure();
	vcSchedNoSchedule = new VcScheduleNoSchedule();
	vcSchedHorizon = new VcScheduleHorizon();
	vcSchedWBS = new VcScheduleWBS(_wbs_beta, _wbs_gamma);
	vcSchedSBB = new VcScheduleSBB(this, _sbb_alpha);
	vcSchedSBBLogWeight = new VcScheduleSBBLogWeight(this, _sbb_alpha);
	vcSchedLW = new VcScheduleLogWeight();

	_scheduling_algorithms.push_back( vcSchedBackpressure );
	_scheduling_algorithms.push_back( vcSchedNoSchedule );
	_scheduling_algorithms.push_back( vcSchedHorizon );
	_scheduling_algorithms.push_back( vcSchedWBS );
	_scheduling_algorithms.push_back( vcSchedSBB );
	_scheduling_algorithms.push_back( vcSchedSBBLogWeight );
	_scheduling_algorithms.push_back( vcSchedLW );

	/* register backpressure scheduling algorithms (Outer) */
	vcSchedOuterVP = new VcScheduleOuterVirtualpressure();
	vcSchedOuterNo = new VcScheduleOuterNoSchedule();

	_scheduling_outer_algorithms.push_back( vcSchedOuterVP );
	_scheduling_outer_algorithms.push_back( vcSchedOuterNo );

	/* register congestion detection algorithms */
	vcCDCORE = new VcCDCORE();
	vcCDQueueLoss = new VcCDQueueLoss();
	vcCDThreshold = new VcCDThreshold( 100 ); /* set the threshold as 100 */
	vcCDDirect = new VcCDDirect( 100 ); /* set the threshold as 100 */

	/* FIXME: Make this happen when you create objects by using static variable */
	vcCDCORE->set_cd_algorithm_index( 0 ); /* gradually increase one-by-one */
	vcCDQueueLoss->set_cd_algorithm_index( 1 ); /* gradually increase one-by-one */
	vcCDThreshold->set_cd_algorithm_index( 2 ); /* gradually increase one-by-one */
	vcCDDirect->set_cd_algorithm_index( 3 ); /* gradually increase one-by-one */

	_cd_algorithms.push_back( vcCDCORE );
	_cd_algorithms.push_back( vcCDQueueLoss );
	_cd_algorithms.push_back( vcCDThreshold );
	_cd_algorithms.push_back( vcCDDirect );

	/* for table algorithm, we only use linear search table */
	_vcTable =  new VcTableLinear<FlowSchedulable>("PFSchedFW", 250 /*queue size*/, 100 /*flow table size */ );
}

int PFSchedFW::initialize(ErrorHandler* )
{
	D_START_FUNCTION;

	/* setup notifier chain */
	_sleepiness=0;
	_empty_note.initialize(Notifier::EMPTY_NOTIFIER, router());

	/* start up timers */
        _flow_expire_timer.initialize(this);
        if( BASE_TIMER_CLOCK ) _flow_expire_timer.schedule_after_msec( BASE_TIMER_CLOCK );

	_outer_loop_timer.initialize(this);
	int t;
	_vcSchedOuter->periodic_monitor( &t );
	_outer_loop_timer.schedule_after_msec( t );


	init_virtual_schedule_slot(10000); /* 10 milli-second */

	D_END_FUNCTION;
	return 0;
}

void PFSchedFW::cleanup(CleanupStage)
{
	if( _vcTable ) {
		delete _vcTable;
	}

	/* TODO: make this as MACRO */
	for( int i = 0; i<_classification_algorithms.size(); i++ ) {
		delete _classification_algorithms[i];
	}

	for( int i = 0; i<_scheduling_algorithms.size(); i++ ) {
		delete _scheduling_algorithms[i];
	}

	for( int i = 0; i<_scheduling_outer_algorithms.size(); i++ ) {
		delete _scheduling_outer_algorithms[i];
	}

	for( int i = 0; i<_cd_algorithms.size(); i++ ) {
		delete _cd_algorithms[i];
	}
}

/************************************************************************
 * Virtual Schedule Slot Implementation 
 ***********************************************************************/
void PFSchedFW::turnon_waker(int32_t t)
{
	if( t == 0 ) {
		_sleepiness = 0;
		_empty_note.wake();
	} else if( _is_wakertimer_ticking == false ) {
        	_sleep_waker_timer.schedule_after_msec( t );
	}
}
void PFSchedFW::init_virtual_schedule_slot(uint32_t)
{
	// _slot_unit = su; /* we set this at configuration step */

	_sleep_waker_timer.initialize(this);
	memset( &_tv_last_schedule, 0, sizeof(struct timeval));
	_is_wakertimer_ticking = false;
}

void PFSchedFW::timestamp_schedule()
{
	gettimeofday( &_tv_last_schedule, NULL );
}

int PFSchedFW::do_we_sleep()
{
	struct timeval tv;
	int64_t diff;
	int32_t sleep_time;
	
	/* if _slot_unit is zero, it means no virtual schedule slotting */
	if( _slot_unit == 0 ) { 
		return 0; 
	}

	gettimeofday( &tv, NULL );
	diff = timevaldiff( &_tv_last_schedule, &tv );
	if( diff > (int64_t)_slot_unit ) {
		return 0;
	}
	
	sleep_time = _slot_unit - (int32_t)diff;

	_empty_note.sleep();
	_is_wakertimer_ticking = true;

	if( sleep_time > VSS_SLEEP_THRESHOLD ) {
        	_sleep_waker_timer.schedule_after_msec( sleep_time / 1000 );
	} else {
		_sleep_waker_timer.schedule_after_msec( VSS_SLEEP_THRESHOLD / 1000 );
	}

	return 1;
}
void PFSchedFW::wake_sleep()
{
	_is_wakertimer_ticking = false;
	_empty_note.wake();
}

void PFSchedFW::dump()
{
	char algorithms[100];
	sprintf( algorithms, "%s", _vcSched->name() );
	_vcTable->dump( _vcClassify, algorithms, 0 );
}

/*
 * call frequency : timer-based
 * click embeded timer
 */
void PFSchedFW::run_timer(Timer* t) 
{
	D_START_FUNCTION;
        if( t == &_flow_expire_timer ) {
		_vcTable->time_tick();

                t->schedule_after_msec( BASE_TIMER_CLOCK );
        } else if ( t == &_outer_loop_timer ) { 	
		int time;
		_vcSchedOuter->periodic_monitor( &time );	
		t->schedule_after_msec( time );
	} else if ( t == &_sleep_waker_timer ) {
		wake_sleep();
	}


	D_END_FUNCTION;
}

/*
 * call frequency: per-packet
 * desc: push
 */
void PFSchedFW::push(int input, Packet *p)
{
	D_START_FUNCTION;		
	

	/* receive data packet */
	if( input == 0 ) {

		struct FlowID fid;
		FlowSchedulable* flow;

		_vcClassify->classify( p, &fid );

		if( _vcTable->lookup( &fid, (FlowSchedulable**)&flow) )	{

			click_ether* ethdr = (click_ether*)p->data();

			_vcTable->add( &fid, (FlowSchedulable**)&flow );

			flow->setNexthopInfo(ethdr->ether_dhost, p->dst_ip_anno() );
		}

		_god->papmo.do_monitor( 
				COMPOSED_TRACE_TAG_FLOW | COMPOSED_TRACE_TAG_CORE ,
				COMPOSED_TRACE_POS_L3_IN, 
				p,
				flow,
				_vcSched,
				NULL,
				NULL );

		/* notifier the upstream elements */
		_sleepiness = 0;

		_empty_note.wake();

		_vcSched->pre_push( flow, p );

		/* In FlowNet-v0.6.0, we maintain multiple CDs and
		 * deal with multiple shapers that could depend on
		 * difference CDs */
		for( int i = 0; i<_cd_algorithms.size(); i++ ) {
			VcCongestionDetection *cd = _cd_algorithms[i];
			cd->packet_enter( flow, p );
		}

		flow->enque( p );

		_vcSched->post_push( flow, p );
	}
	/* 
	 * this packet is promiscuosly received packets.
	 * we can observe neighbor node's queue length status from this packets 
	 */
	else if( input == 1 ) {
		listen_promisc( p );
	}
	/*
	 * receive e2e signalling packet 
	 */ 
	else if( input == 2 ) {
	}

	D_END_FUNCTION;
}

/*
 * call frequency: per-packet
 * desc: pull
 */
Packet * PFSchedFW::pull(int)
{
	D_START_FUNCTION;

	FlowSchedulable *f;
	uint8_t tos;
	Packet *p=NULL;

	/*
	 * Virtual Schedule Slot Implementation 
	 * if we are sleeping, (do_we_sleep gives 1), just do nothing
	 * then, the slee_notifier from to_device will put us to sleep 
	 */
	if( do_we_sleep() == 0 ) {

		_vcSched->schedule( (VcTable<Flow> *) _vcTable, (Flow **) &f );

		if( f != NULL ) {
			/* we have a flow @f that has a packet to transmit */
			f->touch();

			_vcSched->l2_mapping( f, &tos );

			p = f->deque();
			if( p == NULL ) {
				click_chatter("Error! NULL packet is scheduled (flow has %d pkts)\n", f->queue_len );
			}
			else {
				/* we indeed get a packet @p to transmit */

				/* for next-hop queue monitoring, we put the queue length of
				 * the flow itself into the packet */
				put_qlen_to_FO( p, f );

				/* we set the ip tos value for the link scheduling 
				 * at IEEE 802.11 MAC layer (madwifi) */
				setup_iptos( p, tos );

				/* In FlowNet-v0.6.0, we maintain multiple CDs and
				 * deal with multiple shapers that could depend on
				 * difference CDs */
				for( int i = 0; i<_cd_algorithms.size(); i++ ) {
					VcCongestionDetection *cd = _cd_algorithms[i];
					cd->packet_leave( f, p );
				}

				timestamp_schedule();

				/* PaPMo IPS (In-kernel Protocol Sniffer) */
				_god->papmo.do_monitor( 
						COMPOSED_TRACE_TAG_FLOW | COMPOSED_TRACE_TAG_CORE,
						COMPOSED_TRACE_POS_L3_OUT, 
						p,
						f,
						_vcSched,
						NULL,
						NULL );

				_sleepiness = 0;
			}
		} else {

			if( _sleepiness >= SLEEP_TRIGGER ) {
				_empty_note.sleep();
#if HAVE_MULTITHREAD
				/* work around race condition between push() and pull()
				 * We might have just undone push()'s Notifier::wake() call.
				 * Easiest lock-free solution: check wether we should wake again!
				 */
				if( _sleepiness == 0 )
					_empty_note.wake();
#endif
			} else {
				_sleepiness ++;
			}

			p = NULL;
		}
	}

	D_END_FUNCTION;
	return p;
}

void PFSchedFW::listen_promisc( Packet* p )
{
	D_START_FUNCTION;

	struct FlowID fid;
	FlowSchedulable* flow;

	_vcClassify->classify( p, &fid );

	if( _vcTable->lookup( &fid, (FlowSchedulable**)&flow ) )
	{
		D_END_FUNCTION;
		return;
	}

	flow->update_nexthop_queuelen( p );

	_vcSched->listen_promisc( flow );

	p->kill();

	D_END_FUNCTION;
}


/*
 * call frequency: per-packet
 * desc: utility function
 *	put queue length into FO fileds of IP header
 */
void PFSchedFW::put_qlen_to_FO( Packet* p, FlowSchedulable* flow )
{

	D_START_FUNCTION;

	/* ip header mangling (put queue length) */
	WritablePacket *wp = p->uniqueify();
	struct click_ip* cip = (struct click_ip*)(wp->data()+sizeof(struct click_ether));
	uint16_t qlen_mask=0xff00;
	unsigned hlen;
	hlen = cip->ip_hl << 2;
	wp = wp->uniqueify();

	int16_t qlen;

	_vcSched->queue_monitor_policy( flow, &qlen );

	cip->ip_off &= ~qlen_mask;
	cip->ip_off |= qlen_mask & (qlen << 8);
	int queue_size = (cip->ip_off & 0xff00) >> 8;
	if( queue_size != qlen )
		printf("WE GOT THE BUG!!(%d,%d)\n", queue_size, qlen);

	cip->ip_sum = 0;
	cip->ip_sum = click_in_cksum((unsigned char *)cip, hlen);

	D_END_FUNCTION;
}

void PFSchedFW::setup_iptos(Packet* p, uint8_t priority)
{
	D_START_FUNCTION;

	/* ip header mangling (put queue length) */
	WritablePacket *wp = p->uniqueify();
	struct click_ip* cip = (struct click_ip*)(wp->data()+sizeof(struct click_ether));
	unsigned hlen;
	hlen = cip->ip_hl << 2;
	wp = wp->uniqueify();

	cip->ip_tos = priority;

	cip->ip_sum = 0;
	cip->ip_sum = click_in_cksum((unsigned char *)cip, hlen);

	D_END_FUNCTION;
}

int32_t PFSchedFW::get_slot_unit()
{
	return _slot_unit;
}
int32_t PFSchedFW::add_slot_unit(int add)
{
	_slot_unit += add;
	if( _slot_unit < 0 ) _slot_unit = 0;
	if( _slot_unit > 10000 ) _slot_unit = 10000;
	return _slot_unit;
}

CLICK_ENDDECLS
EXPORT_ELEMENT(PFSchedFW)

