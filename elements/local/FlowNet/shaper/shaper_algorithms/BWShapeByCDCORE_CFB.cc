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

#include "BWShapeByCDCORE_CFB.hh"
#include "../../utils/FlowNetUtils.hh"

#include "../../mpeg2/Mpeg2Common.hh"

#include "../PFShapeFW.hh"
#include "../FlowBWShaperable.hh"
CLICK_DECLS

/*
 * jyyoo debugging : indenting stack depth 
 */
D_DEFINE_EXTERN;

VcBWShapeByCDCORE_CFB::VcBWShapeByCDCORE_CFB(PFShapeFW* shape)
{
	_pfshape = shape;
	_accept_frame = 1; /* only accept I frame */
	set_name("ByCDCORE_CFB");
}

VcBWShapeByCDCORE_CFB::~VcBWShapeByCDCORE_CFB()
{
}

void VcBWShapeByCDCORE_CFB::toString(Flow* flow_in, char* buf, int)
{
	FlowBWShaperable* flow = (FlowBWShaperable*)flow_in;
	struct BWShapeInfo* shapeInfo = flow->getBWShapeInfo();
	struct CBhapeByCDCORE_CFB* cbShape = (CBhapeByCDCORE_CFB*)shapeInfo->cb;


	sprintf( buf, "[rate:%d]", cbShape->accept_frame );
}


int VcBWShapeByCDCORE_CFB::do_we_send(Flow* flow_in, Packet* p, const Element::Port &e)
{

	if( p == NULL ) return 0;
	click_ether* ethdr = (click_ether*)(p->data());
	click_ip* iphdr = (click_ip*)(ethdr+1);
	click_udp* udphdr = (click_udp*)(iphdr+1);
	struct bpadapt_header* bpadapt = (struct bpadapt_header*)(udphdr+1);

	if( bpadapt->magicHeader != BPADAPT_MAGIC_HEADER )
	{
		printf_ip( iphdr );
		printf_udp( udphdr );
		printf("%x\n", (uint32_t)bpadapt->magicHeader );
		click_chatter("Warning!! Mpeg2 Shaper receives non mpeg-2 packet\n");
	
		e.push( p );
	} else {

		FlowBWShaperable* flow = (FlowBWShaperable*)flow_in;

		struct BWShapeInfo* shapeInfo = flow->getBWShapeInfo();
		struct CBhapeByCDCORE_CFB* cbShape = (CBhapeByCDCORE_CFB*)shapeInfo->cb;

		assert( cbShape != NULL );

		/* this is the first frame */
		if( cbShape->accept_frame == 0 ) {
			cbShape->accept_frame = 1;
			gettimeofday( &cbShape->tv_last_congestion, NULL );
			cbShape->cd = (VcCongestionDetection*)this;
		}
		
		if( (uint32_t)bpadapt->frametype <= (uint32_t)cbShape->accept_frame )
			e.push(p);
		else
			p->kill();
	}

	return 0;
}

void VcBWShapeByCDCORE_CFB::congestion_action(struct FlowID *, const Packet *)
{

	/* 
	 *consult with PFShaper (who is master of this algorithm)
	 * that whom I have to decrease the rate 
	 */
	FlowBWShaperable *flow;
	struct BWShapeInfo *shapeInfo;
	struct CBhapeByCDCORE_CFB *cbShape;
	struct CBhapeByCDCORE_CFB *cbShape_highest_rate = NULL;

	int s = _pfshape->size();
	for( int i = 0; i<s; i++ ) {
		
		flow = _pfshape->getAt( i );
		shapeInfo = flow->getBWShapeInfo();
		cbShape = (CBhapeByCDCORE_CFB*)shapeInfo->cb;

		assert( cbShape != NULL );

		if( cbShape_highest_rate == NULL ) {
			cbShape_highest_rate = cbShape;
		} else {
			if( cbShape_highest_rate->accept_frame < cbShape->accept_frame ) {
				cbShape_highest_rate = cbShape;
			} else if( cbShape_highest_rate->accept_frame == cbShape->accept_frame && rand() % 2 ) {
				cbShape_highest_rate = cbShape;
			}
		}
	}

	/* TODO: we can extend this not to allow the consecutive 
	 * congestions by using @&cbShape_highest_rate->tv_last_congestion */
	if( cbShape_highest_rate != NULL ) {
		if( cbShape_highest_rate->accept_frame > 1 ) {
			cbShape_highest_rate->accept_frame --;
		}
		gettimeofday( &cbShape_highest_rate->tv_last_congestion, NULL );
	}
}

void VcBWShapeByCDCORE_CFB::nocongestion_action(struct FlowID* fid, const Packet *p)
{
	D_START_FUNCTION;
	
	FlowBWShaperable* flow = _pfshape->lookupflow( fid );
	struct BWShapeInfo* shapeInfo;
	struct CBhapeByCDCORE_CFB* cbShape;
	struct timeval tv;
	long diff;

	if( flow == NULL ) {

		/* second try using packet p */
		flow = _pfshape->lookupflow( p );

		if( flow == NULL ) {
			/* to prevent warning storm */
			static int show_warning = 1;
			if( show_warning % 1000 == 0 ) {
				click_chatter("Warning! VcBWShapeByCDCORE_CFB gets null flow [%d].\n", show_warning);
				click_chatter("You must be using NoSchedule with NoClassify.\n");
				click_chatter("Or this shaper is only for video flow, and the current flow would be non-video\n");
				click_chatter("If not, you it probably is an error, need to be investigated!\n");

				/* investigation */
				#if 0
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
			goto function_out;
		}
	}

	shapeInfo = flow->getBWShapeInfo();
	cbShape = (CBhapeByCDCORE_CFB*)shapeInfo->cb;

	assert( cbShape != NULL );

	gettimeofday( &tv, NULL );

	diff = timevaldiff( &cbShape->tv_last_congestion, &tv );

	if( diff > 10*1000*1000 /* 10 seconds */ )
	{
		/* if we adapt the rate, we update the last signal */
		gettimeofday( &cbShape->tv_last_congestion, NULL );
		if( cbShape->accept_frame < 3 ) cbShape->accept_frame ++;
	}

function_out:

	D_END_FUNCTION;
}

CLICK_ENDDECLS
ELEMENT_PROVIDES(VcBWShapeByCDCORE_CFB)

