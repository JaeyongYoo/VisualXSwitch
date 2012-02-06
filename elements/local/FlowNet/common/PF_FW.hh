#ifndef _PFFW_HH
#define _PFFW_HH
#include <click/glue.hh>
#include <click/element.hh>

#include "FlowCommon.hh"
#include "FlowNetGod.hh"


/* per-flow framework */
class PFFW : public Element { public:

	PFFW();
	~PFFW();

	int configure(Vector<String> &conf, ErrorHandler *);
        const char *class_name() const  { return "PFFW"; }
        const char *port_count() const  { return "1/1"; }
        const char *processing() const  { return PUSH; }


        void add_handlers();
	
	virtual void dump();

public:
        FlowNetGod* _god;

	
public:

	static int write_paramFlowNetGod(const String &in_s, Element *e, void *vparam, ErrorHandler *errh);

};



#endif
