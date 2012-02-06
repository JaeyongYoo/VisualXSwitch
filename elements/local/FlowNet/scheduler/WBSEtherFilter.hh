#ifndef MULTIOP_CHECKWIFIHEADER_HH
#define MULTIOP_CHECKWIFIHEADER_HH
#include <click/element.hh>
#include <click/string.hh>
#include <click/etheraddress.hh>
CLICK_DECLS


class WBSEtherFilter : public Element { public:

	WBSEtherFilter();
	~WBSEtherFilter();

	const char *class_name() const		{ return "WBSEtherFilter"; }
	const char *port_count() const		{ return "1/2"; }
	const char *processing() const		{ return PUSH; }

	int configure(Vector<String> &, ErrorHandler *);
	bool can_live_reconfigure() const		{ return true; }

	void push(int, Packet* p);
private:
	EtherAddress myEther;		
};

CLICK_ENDDECLS
#endif
