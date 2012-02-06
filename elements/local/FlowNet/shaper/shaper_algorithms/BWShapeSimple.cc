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

#include "BWShapeSimple.hh"

#include "../PFShapeFW.hh"
#include "../FlowBWShaperable.hh"

CLICK_DECLS

/*
 * jyyoo debugging : indenting stack depth 
 */
D_DEFINE_EXTERN;


VcBWShapeSimple::VcBWShapeSimple(PFShapeFW *shape)
{
	set_name("Simple");
	_pfshape = shape;
}

VcBWShapeSimple::~VcBWShapeSimple()
{
}

int VcBWShapeSimple::do_we_send(Flow *flow_in, Packet *p, const Element::Port &e)
{
	if( p == NULL ) return 0;

	FlowBWShaperable *flow = (FlowBWShaperable*)flow_in;
	struct BWShapeInfo *shapeInfo = flow->getBWShapeInfo();
	struct CBShapeSimple *cbShape = (CBShapeSimple*)shapeInfo->cb;

	assert( cbShape != NULL );

	if( cbShape->congested ) {
		/* just discard the packet */
		p->kill();
	} else {
		e.push( p );
	}

	return 0;
}

void VcBWShapeSimple::congestion_action(struct FlowID *fid, const Packet *p)
{
	FlowBWShaperable *flow = _pfshape->lookupflow( fid );

	if( flow == NULL ) {

		/* second try using packet p */
		flow = _pfshape->lookupflow( p );


		if( flow == NULL ) {
			static int show_warning=1;
			if( show_warning % 1000 == 0 ) {
				click_chatter("Warning! VcBWShapeSimple gets null flow [%d]. Three possibilities\n", show_warning);
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
	struct CBShapeSimple* cbShape = (CBShapeSimple*)shapeInfo->cb;

	assert(cbShape != NULL);

	cbShape->congested = true;

}

void VcBWShapeSimple::nocongestion_action(struct FlowID *fid, const Packet *p)
{
	FlowBWShaperable *flow = _pfshape->lookupflow( fid );

	if( flow == NULL ) {

		/* second try using packet p */
		flow = _pfshape->lookupflow( p );

		if( flow == NULL ) {
			static int show_warning=1;
			if( show_warning % 1000 == 0 ) {
				click_chatter("Warning! VcBWShapeSimple gets null flow [%d].\n", show_warning);
				click_chatter("You must be using NoSchedule with NoClassify.\n");
				click_chatter("Or this shaper is only for video flow, and the current flow would be non-video\n");
				click_chatter("If not, you it probably is an error, need to be investigated!\n");

				/* investigation */
				#if 1
				click_chatter("fid=%x, packet=%x\n", fid, p );
				if( fid != NULL ) {
					click_chatter("classifier name=%s\n", fid->classifier->name() );
					char buf[1024];
					fid->classifier->to_string( (const struct FlowID *) fid, buf, 1024 );
					click_chatter("classify string: %s\n", buf  );
				}
				_pfshape->dump();
				#endif
			}
			show_warning ++;

			return;
		}
	}
	struct BWShapeInfo* shapeInfo = flow->getBWShapeInfo();
	struct CBShapeSimple* cbShape = (CBShapeSimple*)shapeInfo->cb;

	assert(cbShape != NULL);

	cbShape->congested = false;;
}
CLICK_ENDDECLS
ELEMENT_PROVIDES(VcBWShaperSimple)

