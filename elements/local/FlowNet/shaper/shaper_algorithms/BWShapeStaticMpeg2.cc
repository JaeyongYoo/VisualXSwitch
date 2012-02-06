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

#include "BWShapeStaticMpeg2.hh"
#include "../../utils/FlowNetUtils.hh"

#include "../../mpeg2/Mpeg2Common.hh"

CLICK_DECLS

/*
 * jyyoo debugging : indenting stack depth 
 */
D_DEFINE_EXTERN;


VcBWShapeStaticMpeg2::VcBWShapeStaticMpeg2(int af)
{
	_accept_frame = af;
	set_name("StaticMpeg2");
}

VcBWShapeStaticMpeg2::~VcBWShapeStaticMpeg2()
{
}


int VcBWShapeStaticMpeg2::do_we_send(Flow*, Packet* p, const Element::Port &e)
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
		fprintf(stderr, "Warning!! Mpeg2 Shaper receives non mpeg-2 packet\n");
	
		e.push( p );
	} else {
		if( (uint32_t)bpadapt->frametype <= (uint32_t)_accept_frame )
		{
			e.push(p);
		}
		else
			p->kill();
	}

	return 0;
}

void VcBWShapeStaticMpeg2::change_rate( int r )
{
	_accept_frame = r;
}

int VcBWShapeStaticMpeg2::get_rate()
{
	return _accept_frame;
}
CLICK_ENDDECLS
ELEMENT_PROVIDES(VcBWShaperStaticMpeg2)

