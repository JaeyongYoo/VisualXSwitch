#ifndef MULTIOP_UNFORGE_HH
#define MULTIOP_UNFORGE_HH
#include <click/element.hh>
#include <click/string.hh>
CLICK_DECLS


class IPFragReset : public Element { public:

	IPFragReset();
	~IPFragReset();

	const char *class_name() const		{ return "IPFragReset"; }
	const char *port_count() const		{ return PORTS_1_1; }
	const char *processing() const		{ return AGNOSTIC; }

	int configure(Vector<String> &, ErrorHandler *);
	bool can_live_reconfigure() const		{ return true; }

	Packet *simple_action(Packet *);

	private:
};

CLICK_ENDDECLS
#endif
