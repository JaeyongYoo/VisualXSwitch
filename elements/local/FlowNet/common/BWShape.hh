// -*- c-basic-offset: 4 -*-
#ifndef _BWSHAPER_HH
#define _BWSHAPER_HH
#include <click/glue.hh>
#include <click/element.hh>
#include <click/timer.hh>
#include <click/notifier.hh>

#include "Algorithm.hh"
#include "CD.hh"
#include "FlowCommon.hh"
#include "Flow.hh"

CLICK_DECLS

/*!
 * virtual functions 
 */

class VcBWShape : public Algorithm {
public:
	VcBWShape();
	~VcBWShape();
public:

	virtual int do_we_send(Flow* flow, Packet* p, const Element::Port &e) = 0;

	virtual void toString( Flow* flow, char* buf, int len );

	void registerCDCallback(VcCongestionDetection* CD);

        static void _congestion_detected(struct CongestionNotification *cn);
        static void _nocongestion_detected(struct CongestionNotification *cn);
        static void _queue_length_changed(struct QueueLengthChangeNotification *qn);

        virtual void congestion_action(struct FlowID* fid, const Packet* p);
        virtual void nocongestion_action(struct FlowID* fid, const Packet* p);
        virtual void queue_length_changed(struct FlowID* fid, const Packet* p, uint32_t ql);


protected:
	VcCongestionDetection* _cd;

};

CLICK_ENDDECLS
#endif
