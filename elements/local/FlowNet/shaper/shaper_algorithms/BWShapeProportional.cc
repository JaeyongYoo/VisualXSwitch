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

#include <clicknet/ip.h>
#include <clicknet/udp.h>
#include <clicknet/ether.h>

#include "BWShapeProportional.hh"
#include "../PFShapeFW.hh"
#include "../FlowBWShaperable.hh"
#include "../../utils/FlowNetUtils.hh"

CLICK_DECLS

/*
 * jyyoo debugging : indenting stack depth 
 */
D_DEFINE_EXTERN;


VcBWShapeProportional::VcBWShapeProportional(PFShapeFW *shape)
{
	set_name("Proportional");
	_pfshape = shape;
	_averaged_rate = 0;
	_alpha = 95;
}

VcBWShapeProportional::~VcBWShapeProportional()
{

}

int VcBWShapeProportional::do_we_send(Flow *flow_in, Packet *p, const Element::Port &e)
{
	if( p == NULL ) return 0;

	FlowBWShaperable *flow = (FlowBWShaperable*)flow_in;
	struct BWShapeInfo *shapeInfo = flow->getBWShapeInfo();
	struct CBShapeProportional *cbShape = (CBShapeProportional*)shapeInfo->cb;

	assert( cbShape != NULL );

	/* compute target rate */
	/* we use the algorithm from the one that we use utility function as ln(x) */
	if( cbShape->queue_length != 0 ) {
		_target_rate = 1000 * 1000 /  cbShape->queue_length;
	} else {
		_target_rate = 1000 * 1000;
	}

	/* here, we profile the speed and throtle the speed */
	int32_t diff;
	struct timeval tv;
	uint32_t current_rate; /* bytes per sec */
	gettimeofday( &tv, NULL );
	diff = timevaldiff(  &_tv_last_packet_out, &tv );

	if( _averaged_rate < _target_rate ) {
		e.push( p );
		current_rate = 1000*1000 * p->length() / diff;
		
		

	} else {
		p->kill();
		current_rate = 0;
	}

	memcpy( &_tv_last_packet_out, &tv, sizeof(struct timeval) );

	/* do EWMA */
	_averaged_rate = ( _averaged_rate * _alpha + current_rate * (100 - _alpha) ) / 100;

//	click_chatter("averaged_rate = %d (cur: %d, diff=%d), target_rate = %d queue length = %d \n", 
//		_averaged_rate, current_rate, diff,  _target_rate, cbShape->queue_length );

	return 0;
}

void VcBWShapeProportional::queue_length_changed(struct FlowID* fid, const Packet* p, uint32_t ql)
{
        FlowBWShaperable *flow = _pfshape->lookupflow( fid );

        if( flow == NULL ) {

                /* second try using packet p */
                flow = _pfshape->lookupflow( p );


                if( flow == NULL ) {
                        static int show_warning=1;
                        if( show_warning % 1000 == 0 ) {
                                click_chatter("Warning! VcBWShapeProportional gets null flow [%d]. Three possibilities\n", show_warning);
                                click_chatter("1. You must be using NoSchedule with NoClassify.\n");
                                click_chatter("2. Or this shaper is only for video flow, and the current flow would be non-video.\n");
                                click_chatter("3. Or this shaper is not the source node of this flow.\n");
                                click_chatter("If not, you it probably is an error, need to be investigated!\n");


                                #if 0 /* investigation for the above warning. Also try to uncomment in PFShapeFW.cc */
                                click_chatter("fid=%x, packet=%x\n", fid, p );
                                if( fid != NULL ) {
                                        click_chatter("classifier name=%s\n", fid->classifier->name() );
                                        char buf[1024];
                                        fid->classifier->to_string( (const struct FlowID *) fid, buf, 1024 );
                                        click_chatter("classify string: %s\n", buf  );
                                }
                                _pfshape->dump();

                                click_chatter("INSPECTION END-MARK\n");
                                #endif
                        }
                        show_warning ++;

                        return;
                }
        }
        struct BWShapeInfo* shapeInfo = flow->getBWShapeInfo();
        struct CBShapeProportional* cbShape = (CBShapeProportional*)shapeInfo->cb;

        assert(cbShape != NULL);
	cbShape->queue_length = ql;
}



CLICK_ENDDECLS
ELEMENT_PROVIDES(VcBWShaperProportional)

