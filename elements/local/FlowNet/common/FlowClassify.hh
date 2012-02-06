// -*- c-basic-offset: 4 -*-
#ifndef _FLOW_CLASSIFY_HH
#define _FLOW_CLASSIFY_HH
#include <click/glue.hh>
#include <click/element.hh>
#include <click/timer.hh>

#include "FlowCommon.hh"
#include "Flow.hh"
#include "Algorithm.hh"

CLICK_DECLS

/*!
 * virtual functions 
 */

class VcFlowClassify : public Algorithm {
public:

	VcFlowClassify() {};
	~VcFlowClassify() {};

	virtual int classify(const Packet* p, struct FlowID* fid) = 0;
	virtual int to_string(const struct FlowID* fid, char* buf, int len) = 0;

};

CLICK_ENDDECLS
#endif
