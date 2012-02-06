// -*- c-basic-offset: 4 -*-
#ifndef _FLOWNET_PFSINKFW_HH_
#define _FLOWNET_PFSINKFW_HH_
#include <click/glue.hh>
#include <click/element.hh>
#include <click/timer.hh>

#include "../common/PF_FW.hh"


CLICK_DECLS


class PFSinkFW : public PFFW { public:

	PFSinkFW();
	~PFSinkFW();

	void *cast(const char *name);
	int configure(Vector<String>&, ErrorHandler*);

	void push(int i, Packet* p);

	const char *class_name() const	{ return "PFSinkFW"; }
	const char *port_count() const	{ return "1/1"; }
	const char *processing() const	{ return PUSH; }


};

CLICK_ENDDECLS
#endif
