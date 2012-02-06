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


#include "PF_FW.hh"


CLICK_DECLS
/*
 * jyyoo debugging : indenting stack depth 
 */
D_DEFINE_EXTERN;


PFFW::PFFW()
{
	_god = NULL;
}

PFFW::~PFFW()
{
}

int PFFW::configure(Vector<String> &conf, ErrorHandler *errh)
{
        if (cp_va_kparse(conf, this, errh,
                                "GOD", cpkP, cpElement, &_god,
                                cpEnd) < 0)
        {
                return -1;
        }

	return 0;
}

void PFFW::dump()
{
	fprintf(stderr, "Not implemented\n");
}


/***************************
 * ControlSocket handlers
 **************************/
int PFFW::write_paramFlowNetGod(const String &in_s, Element *e, void *, ErrorHandler *errh)
{
        PFFW* pffw = (PFFW*)e;
        Element* element;


        Vector<String> args;
        cp_spacevec(in_s, args);

        int res = cp_va_kparse(in_s, pffw, errh, "FlowNetGod", cpkP, cpElement, &element, cpEnd);
        if( res < 0 ) { return res; }
        if( strcmp( element->class_name(), "FlowNetGod" ) ) {
                printf("Class name is not \"FlowNetGod\"\n");
                D_END_FUNCTION;
                return -1;
        }

        pffw->_god = (FlowNetGod*)element;

        return 0;
}

void PFFW::add_handlers()
{
        add_write_handler("_god", write_paramFlowNetGod, (void*)0);
}


CLICK_ENDDECLS
EXPORT_ELEMENT(PFFW)

