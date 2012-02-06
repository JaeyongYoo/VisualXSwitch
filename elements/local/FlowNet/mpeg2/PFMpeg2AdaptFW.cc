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


#include "../libclassify/Classify5Tuple.hh"
#include "PFMpeg2AdaptFW.hh"


CLICK_DECLS

/*
 * jyyoo debugging : indenting stack depth 
 */
D_DEFINE_EXTERN;



void * PFMpeg2AdaptFW::cast(const char *name)
{
	if (strcmp(name, "PFMpeg2AdaptFW") == 0)
		return (void *)this;
	else
		return Element::cast(name);
}
PFMpeg2AdaptFW::PFMpeg2AdaptFW() : expire_timer(this)
{ 
	memset( &stat, 0, sizeof(stat) );
	doWeEncap = 1;
	_god = NULL;
        vcTableEncap=NULL;
        vcTableDecap=NULL;

}

PFMpeg2AdaptFW::~PFMpeg2AdaptFW() 
{
}

int PFMpeg2AdaptFW::configure(Vector<String> &conf, ErrorHandler *errh)
{
	D_START_FUNCTION;

	String strMode;
	String strModeParse;
	int before = errh->nerrors();
	bool name=false;
	bool opmode_parse=false;
	bool hasGod = false;
	if( cp_va_kparse(conf, this, errh, 
				"OPMODE", cpkP, cpString, &strMode, 
				"GOD", cpkC, &hasGod, cpElement, &_god,
				"OPPARSEMODE", cpkC, &opmode_parse, cpString, &strModeParse, 
				"NAME", cpkC, &name, cpString, &mpeg2Name,
				cpEnd) < 0 ) {
		D_END_FUNCTION;
		return -1;
	}

	if( _god )
	{
		_god->register_pffw( this );
	}

	if( !name ) {
		mpeg2Name = "Mpeg2 Generic";
	}
	
	/* op mode  */
	if( strcmp( "Encap", strMode.c_str() ) == 0 ) {
		doWeEncap = 1;
	} else if( strcmp( "Decap", strMode.c_str() ) == 0 ) {
		doWeEncap = 0;
	} else {
		fprintf(stderr, "Error! OPMODE should be either Encap or Decap\n");
		return -1;
	}

	if( doWeEncap == 1 ) { /* parse op mode only works for encap */ 
		/* parse op mode */
		if( strcmp( "Normal", strModeParse.c_str() ) == 0 ) {

			opParseMode = MPEG2_PARSEMODE_NORMAL;
		}
		else 
		{
			opParseMode = MPEG2_PARSEMODE_NULL;
			fprintf(stderr, "Warning: OPMODE_PARSE is No parsing\n");
		}
	}

	D_END_FUNCTION;
	return (errh->nerrors() != before ? -1 : 0);
}


int PFMpeg2AdaptFW::initialize(ErrorHandler*)
{

	D_START_FUNCTION;
	vcClassify = new Vc5TupleClassify();

        vcTableEncap=NULL;
        vcTableDecap=NULL;

	if( doWeEncap )
	{
	        vcTableEncap =  new VcTableLinear<FlowMpeg2AdaptEncap>(
									mpeg2Name.c_str(),
									250, /*queue size*/
        	                                                        20 /*flow table size */ );
	}
	else {
	        vcTableDecap =  new VcTableLinear<FlowMpeg2AdaptDecap>(
									mpeg2Name.c_str(),
									250, /*queue size*/
        	                                                        20 /*flow table size */ );
	}

	expire_timer.initialize(this);

	if( BASE_TIMER_CLOCK ) expire_timer.schedule_after_msec(BASE_TIMER_CLOCK);

	srand(time(NULL));

	D_END_FUNCTION;
	return 0;
}


void PFMpeg2AdaptFW::dump()
{
	if( doWeEncap )
	{
		vcTableEncap->dump( vcClassify, NULL, 0 );
	} else {
		vcTableDecap->dump( vcClassify, NULL, 0 );
	}
}

/*
 * call frequency : timer-based
 * click embeded timer
 */

void PFMpeg2AdaptFW::run_timer(Timer* t) 
{
	if( t == &expire_timer) { /* count age */
		
		if( vcTableEncap ) vcTableEncap->time_tick();
		if( vcTableDecap ) vcTableDecap->time_tick();

		t->schedule_after_msec(BASE_TIMER_CLOCK);
	}
	else {
		printf("Error!: unregistered timer\n");
	}
}


/*
 * call frequency: per-packet
 * desc: push
 */
void PFMpeg2AdaptFW::push(int input, Packet *p)
{
	D_START_FUNCTION;

	if( input == 0 )
	{

		/* pass over-sized packets */
		if( p->length() > 1440 /* rough threshold */ ) 
		{
			output(0).push( p );
			return;
		}


		struct FlowID fid;

		vcClassify->classify( p, &fid );



		/* encap part */
		if( vcTableEncap )
		{
			FlowMpeg2AdaptEncap *flow;
			if( vcTableEncap->lookup( &fid, (FlowMpeg2AdaptEncap**)&flow) )
			{
				vcTableEncap->add( &fid, (FlowMpeg2AdaptEncap**)&flow );
			}


			/* mpeg2 parsing */
			WritablePacket* arr[STREAMINGPROXY_MAX_SEPARATION];
			int arr_size;
			int re;
			if( (re = flow->parse_packet( p, arr, &arr_size, opParseMode )) != 0 )
			{
				fprintf(stderr, "Frame Dropped\n");
				flow->print_error_message(re);

				p->kill();
			} else {

				/* count table stats */
				for( int i = 0; i<arr_size; i++ )
				{
					/* NOTE: this part is quite tricky 
					 * we have to put some information into each packet 
					 * the information is about how many network packets are constructing a video frame
					 * and what is the current index for the video frame 
					 * thus, we buffer packets for a entire video frame and mark the two information into the header 
					 * and when the marking is done, we flush the entire packets constructing the frame at once 
					 * XXX: this might cause packet bursting problem */
					flow->enque( arr[i], this );

					/* FIXME: we monitor papmo in flow 
					 * this viloates the rule of per-flow framework.
					 * should fix it later */
					/*
					   if( _god ) _god->papmo.do_monitor(
					   COMPOSED_TRACE_TAG_MPEG ,
					   COMPOSED_TRACE_POS_L4_OUT,
					   (Packet*)arr[i],
					   flow,
					   NULL,
					   NULL,
					   NULL );*/

					flow->deque_and_send( this, _god != NULL ? &_god->papmo : NULL );
				}
			}
		}
		else if( vcTableDecap ) {

			FlowMpeg2AdaptDecap *flow;
			if( vcTableDecap->lookup( &fid, (FlowMpeg2AdaptDecap**)&flow) )
			{
				vcTableDecap->add( &fid, (FlowMpeg2AdaptDecap**)&flow );
			}

			if( _god ) { 
				_god->papmo.do_monitor(
						COMPOSED_TRACE_TAG_MPEG ,
						COMPOSED_TRACE_POS_L4_PRE_IN,
						p,
						flow,
						NULL,
						NULL,
						NULL );
			}


			flow->enque( p, this, _god != NULL ? &_god->papmo : NULL );
			/* FIXME: we monitor papmo inside flow 
			 * this viloates the rule of per-flow framework.
			 * should fix it later */
			/*
			   if( _god ) _god->papmo.do_monitor(
			   COMPOSED_TRACE_TAG_MPEG ,
			   COMPOSED_TRACE_POS_L4_IN,
			   p,
			   flow,
			   NULL,
			   NULL,
			   NULL );
			 */


			flow->deque_and_send( this, _god != NULL ? &_god->papmo : NULL );


		}
	}
	D_END_FUNCTION;
}

CLICK_ENDDECLS
EXPORT_ELEMENT(PFMpeg2AdaptFW)

