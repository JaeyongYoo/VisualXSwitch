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

#include "../scheduler/PFSchedFW.hh"
#include "../libclassify/Classify5Tuple.hh"
#include "../libclassify/ClassifyNoClassify.hh"
#include "../libtable/TableLinear.hh"

#include "shaper_algorithms/BWShapeByCDCORE_CFB.hh"
#include "shaper_algorithms/BWShapeByCDCORE.hh"
#include "shaper_algorithms/BWShapeNULL.hh"
#include "shaper_algorithms/BWShapeSimple.hh"
#include "shaper_algorithms/BWShapeProportional.hh"
#include "shaper_algorithms/BWShapeStaticMpeg2.hh"

#include "PFShapeFW.hh"

CLICK_DECLS


/*
 * jyyoo debugging : indenting stack depth 
 */
D_DEFINE_EXTERN;


void * PFShapeFW::cast(const char *name)
{
	if (strcmp(name, "PFShapeFW") == 0)
		return (void *)this;
	else
		return Element::cast(name);
}

PFShapeFW::PFShapeFW() : send_timer(this), expire_timer(this)
{
	_god = NULL;
	_vcClassify = NULL;
	_vcShape = NULL;
	_is_binding_to_cd = false;
}

PFShapeFW::~PFShapeFW()
{
}

int PFShapeFW::configure(Vector<String> &conf, ErrorHandler *errh)
{
	D_START_FUNCTION;
	bool tt;
	
	_mpegShape = 3;
	_turnoff_timer = false;

        if (cp_va_kparse(conf, this, errh,
                                /* device parameter setting */
				"GOD", cpkP, cpElement, &_god,
                                "SHAPE_ALGORITHM", cpkP, cpString, &_shape_algorithm,
                                "CLASSIFY_ALGORITHM", cpkP, cpString, &_classify_algorithm,
				"BIND_TO_CD", cpkC, &_is_binding_to_cd, cpString, &_str_bind_to_CD,
				"MPEG2_STATIC", cpkC, &_have_mpegShape, cpInteger, &_mpegShape,
				"TURNOFF_TIMER", cpkC, &tt, cpBool, &_turnoff_timer,
                                cpEnd) < 0)
	{
		D_END_FUNCTION;
                return -1;
	}
	
	/*
	 *  we have no _god 
	 */
	if( _god ) {
		_god->pfshape = this;
		_god->register_pffw( this );
	}

	D_END_FUNCTION;
	return 0;
}

int PFShapeFW::initialize(ErrorHandler*)
{
	D_START_FUNCTION;

	/* register algorithms */
	create_and_register_algorithms();

        /* TODO: Make this as CHOOSE_ALGORITHM macro */
        /* choose algorithms */
        for( int i = 0; i<_classification_algorithms.size(); i++ ) {
                if( strncmp( _classify_algorithm.c_str(),
                                _classification_algorithms[i]->name(),
                                FLOWNET_ALGORITHM_NAME_SIZE ) == 0 ) {
                                _vcClassify = _classification_algorithms[i];
                        break;
                }
        }

	for( int i = 0; i<_shaper_algorithms.size(); i++ ) {
		if( strncmp( _shape_algorithm.c_str(),
				_shaper_algorithms[i]->name(),
				FLOWNET_ALGORITHM_NAME_SIZE ) == 0 ) {
				_vcShape = _shaper_algorithms[i];
			break;
		}
	}

	/* 
	 * TODO: if an algorithm is NULL, then print out the available algorithm list
	 * for the corresponding algorithm 
	 */
	if( _vcShape == NULL || _vcClassify == NULL || _vcTable == NULL ) {
		
		click_chatter("Error! You have to specify all algorithms\n");
		click_chatter("selected algorithms\n");
		click_chatter("[%s]\t_vcShape=%x\n", _shape_algorithm.c_str(), _vcShape);
		click_chatter("[%s]\t_vcClassify == %x\n",_classify_algorithm.c_str(),  _vcClassify);
		click_chatter("\t_vcTable == %x\n", _vcTable);

		return -1;
	}

	/* 
	 * shapers depend on CD (Congestion Detection) algorithm 
	 * thus, get this from PFscheduler (Per-Flow scheduler)
	 */
	if( _is_binding_to_cd ) {
		assert( _god );
		assert( _god->pfsched );
		VcCongestionDetection* cd = _god->pfsched->getCDAlgorithm( _str_bind_to_CD.c_str() );
		_vcShape->registerCDCallback( cd );
	}

	/* initialize timers */
	send_timer.initialize(this);
	expire_timer.initialize(this);

	if( BASE_TIMER_CLOCK ) expire_timer.schedule_after_msec(BASE_TIMER_CLOCK);

	if( !_turnoff_timer && SEND_TIMER_CLOCK ) send_timer.schedule_after_msec(SEND_TIMER_CLOCK);

	
	D_END_FUNCTION;
	return 0;
}

/* 
 * Create and register new algorithms 
 * NOTE: if you implement new algorithm,
 * register here 
 */
void PFShapeFW::create_and_register_algorithms()
{
	/* 
	 * create and register classification algorithms 
	 */
        Vc5TupleClassify *vc5TupleClassify;
        VcNoClassify *vcNoClassify;

        vc5TupleClassify = new Vc5TupleClassify();
        vcNoClassify = new VcNoClassify();

        _classification_algorithms.push_back( vc5TupleClassify );
        _classification_algorithms.push_back( vcNoClassify );

	/* 
	 * create and register shaper algorithms 
	 */
	VcBWShapeSimple				*vcShapeSimple;
	VcBWShapeProportional			*vcShapeProportional;
	VcBWShapeNULL				*vcShapeNULL;
	VcBWShapeStaticMpeg2			*vcShapeStaticMpeg2;
	VcBWShapeByCDCORE				*vcShapeByCDCORE;
	VcBWShapeByCDCORE_CFB			*vcShapeByCDCORE_CFB;

	vcShapeSimple = new VcBWShapeSimple( this );
	vcShapeProportional = new VcBWShapeProportional( this );
	vcShapeNULL = new VcBWShapeNULL();
	vcShapeStaticMpeg2 = new VcBWShapeStaticMpeg2( _mpegShape );
	vcShapeByCDCORE = new VcBWShapeByCDCORE( this );
	vcShapeByCDCORE_CFB = new VcBWShapeByCDCORE_CFB( this );

	_shaper_algorithms.push_back( vcShapeSimple );
	_shaper_algorithms.push_back( vcShapeProportional );
	_shaper_algorithms.push_back( vcShapeNULL );
	_shaper_algorithms.push_back( vcShapeStaticMpeg2 );
	_shaper_algorithms.push_back( vcShapeByCDCORE );
	_shaper_algorithms.push_back( vcShapeByCDCORE_CFB );

	/* 
	 * we only have just one table algorithm, so just use it 
	 */
        _vcTable =  new VcTableLinear<FlowBWShaperable>(
								"PFShaper",
								250, /*queue size*/
                                                                100 /*flow table size */ );
}

void PFShapeFW::cleanup(CleanupStage)
{
	if( _vcTable )	{
		delete _vcTable;
	}

        /* TODO: make this as MACRO */
        for( int i = 0; i<_classification_algorithms.size(); i++ ) {
                delete _classification_algorithms[i];
        }
	
	for( int i = 0; i<_shaper_algorithms.size(); i++ ) {
		delete _shaper_algorithms[i];
	}

}

void PFShapeFW::dump()
{
	click_chatter("[%x] Dump! _vcTable: %x\n", this, _vcTable);
	_vcTable->dump( _vcClassify, _vcShape->name(), 0 );
}

/*
 * call frequency : timer-based
 * click embeded timer
 */
void PFShapeFW::run_timer(Timer* t) 
{

	if( t == &expire_timer) { /* count age */

		_vcTable->time_tick();

		if( BASE_TIMER_CLOCK )
			t->schedule_after_msec(BASE_TIMER_CLOCK);

	} else if( !_turnoff_timer && t == &send_timer) {
		source_send();
		/* do nothing */
		t->schedule_after_msec(SEND_TIMER_CLOCK);
	}
	else {
		printf("Error! unregistered timer\n");
	}
}


/*
 * call frequency: per-packet
 * desc: push
 */
void PFShapeFW::push(int i, Packet *p)
{
	D_START_FUNCTION;
	
	if( i == 0 )
	{
		bool just_added = false;
		FlowBWShaperable *flow;
		struct FlowID fid;

		_vcClassify->classify( p, &fid );

		#if 0 /* useful debug for [Warning! Shaper gets NULL flow] */
		{
			char buf[1024];
			click_chatter("Debugging for Warning! shaper gets NULL flow\n");
			_vcClassify->to_string( (const struct FlowID *) &fid, buf, 1024 );
			click_chatter("PFShapeFW[%x]::push packet: %s\n", this, buf  );
			
			dump();
		}
		#endif

		/* FIXME: use wrapper function instead */
		if( _vcTable->lookup( &fid, (FlowBWShaperable**)&flow) != 0 )
		{
			_vcTable->add( &fid, (FlowBWShaperable**)&flow );

			/* for information dereferencing */
			flow->setShaper( _vcShape );

			just_added = true;
		}

		if( just_added )
		{
			/* 
			 * if this packet is the first packet of this particular flow,
			 * send immediately to prepare ARQ stuffs
			 */

			/* count table stats */
			stat.pkts_out ++;

			output(0).push(p);


		} else {

			if( _vcShape ) {
				_vcShape->do_we_send( (Flow*)flow, p, output(0) );
			}
			else {
				output(0).push(p);
			}
		}
	}

	D_END_FUNCTION;
}
/*
 * call frequency: timer-based (but very fast)
 * desc: source_send
 */
void PFShapeFW::source_send()
{

	D_START_FUNCTION;
	int s = size();
	FlowBWShaperable* flow;
	for( int i=0; i<s; i++ )
	{
		flow = getAt(i);
		if( flow == NULL )
		{
			printf("Error! scheduler gets NULL flow\n");
			exit(-1);
		}
		
		_vcShape->do_we_send( flow, NULL, output(0) );
	}
	D_END_FUNCTION;
}

void PFShapeFW::change_rate( int r ) 
{
	/* this is an exceptional call for vcShapeStaticMpeg2 */
	if( strncmp( _vcShape->name(), "StaticMpeg2", 20 ) == 0 ) {
		((VcBWShapeStaticMpeg2*)_vcShape)->change_rate( r );
	}
}

int PFShapeFW::get_rate()
{
	/* this is an exceptional call for vcShapeStaticMpeg2 */
	if( strncmp( _vcShape->name(), "StaticMpeg2", 20 ) == 0 ) {
		return ((VcBWShapeStaticMpeg2*)_vcShape)->get_rate();
	}
	return -1;
}

/*
 * wrapper functions of _vcTable 
 */
FlowBWShaperable* PFShapeFW::lookupflow( const Packet* p )
{
	struct FlowID fid;
	if( _vcClassify->classify( p, &fid ) ) {
		return NULL;
	}
	return lookupflow( &fid );
}
FlowBWShaperable* PFShapeFW::lookupflow( struct FlowID* fid )
{
	FlowBWShaperable* flow;
	if( _vcTable->lookup( fid, &flow ) ) {
		return NULL;
	}
	return flow; 
}

FlowBWShaperable* PFShapeFW::getAt(int i)
{
	FlowBWShaperable* flow;
	_vcTable->getAt(i, (FlowBWShaperable**) &flow);
	return flow;
}

int PFShapeFW::size()
{
	return _vcTable->size();
}


/***************************
 * ControlSocket handlers
 **************************/
int PFShapeFW::write_param(const String &s, Element *e, void *, ErrorHandler *errh)
{
	PFShapeFW *sw = (PFShapeFW *)e;
	int ch_rate;
	if (!cp_integer(s, &ch_rate))
		return errh->error("PFShapeFW output must be integer");
	
	sw->change_rate( ch_rate );
	return 0;
}
String PFShapeFW::read_param(Element *e, void *)
{
  PFShapeFW *sw = (PFShapeFW *)e;
  return String(sw->get_rate());
}


void PFShapeFW::add_handlers()
{
	add_write_handler("set_mpeg2_shape", write_param, (void*)0, Handler::NONEXCLUSIVE);
	add_read_handler("get_mpeg2_shape", read_param, (void*)0);
}


#include <click/vector.hh>
CLICK_ENDDECLS
EXPORT_ELEMENT(PFShapeFW)

