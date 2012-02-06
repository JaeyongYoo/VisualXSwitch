// -*- c-basic-offset: 4 -*-
#ifndef _TABLE_HH
#define _TABLE_HH
#include <click/glue.hh>
#include <click/element.hh>
#include <click/timer.hh>
#include <click/notifier.hh>

#include "FlowCommon.hh"
#include "Flow.hh"
#include "FlowClassify.hh"

CLICK_DECLS

/*!
 * virtual functions 
 */

template <class T>
class VcTable {
public:
	VcTable() { };
	VcTable(const char* n) { };
	~VcTable() {};
public:
	virtual int add(const FlowID* fid, T** f)=0;
	virtual int lookup(const FlowID* fid, T** f)=0;
	virtual int removeByFlowID(const FlowID* fid)=0;
	virtual int removeByFlow(T *f)=0;
	virtual int time_tick()=0;
	virtual int dump( VcFlowClassify *, const char *buf, int len)=0;
	virtual int size()=0;
	virtual int getAt(int i, T **f)=0;

protected:
};

CLICK_ENDDECLS
#endif
