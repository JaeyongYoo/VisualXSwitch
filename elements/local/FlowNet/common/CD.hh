// -*- c-basic-offset: 4 -*-
#ifndef _CongestionDetection_HH
#define _CongestionDetection_HH
#include <click/glue.hh>
#include <click/element.hh>
#include <click/timer.hh>
#include <click/notifier.hh>

#include "Algorithm.hh"
#include "FlowCommon.hh"
#include "Flow.hh"

CLICK_DECLS

struct QueueLengthChangeNotification {
	void* object;
	struct FlowID* fid;
	const Packet* packet;
	uint32_t queue_length;
};

struct CongestionNotification {
	void* object;
	struct FlowID* fid;
	const Packet* packet;
};

/*!
 * virtual functions 
 */

class VcCongestionDetection : public Algorithm {
public:
        VcCongestionDetection() : congest_detected(NULL), nocongest_detected(NULL), 
		queue_length_change_notify(NULL), _cd_algorithm_index(-1)
	{};
        ~VcCongestionDetection() {};
public:

	virtual int packet_enter(Flow *flow, const Packet *p) = 0;
	virtual int packet_leave(Flow *flow, const Packet *p) = 0;

	inline int register_congestion_callback( void (*cd)(struct CongestionNotification*), void *);
	inline int register_nocongestion_callback( void (*cd)(struct CongestionNotification*), void *);

	inline int register_queue_length_change_notify_callback( void (*cd)(struct QueueLengthChangeNotification *), void *pd );

	inline void set_cd_algorithm_index( int32_t i );
protected:
	void* private_data;

	/* TODO: need to extend this callbacks as arrays
	 * to accept multiple registrations from
	 * the same CD by many shapers */
	void (*congest_detected)(struct CongestionNotification *);
	void (*nocongest_detected)(struct CongestionNotification *);

	/*
	 * for directly using queue length for shaping 
	 */
	void (*queue_length_change_notify)(struct QueueLengthChangeNotification *);

	int32_t _cd_algorithm_index;
};

inline int VcCongestionDetection::register_congestion_callback( void (*fn)(struct CongestionNotification*), void* pd )
{
	private_data = pd;
	congest_detected = fn;
	return 0;
}
inline int VcCongestionDetection::register_nocongestion_callback( void (*fn)(struct CongestionNotification*), void* pd )
{
	private_data = pd;
	nocongest_detected = fn;
	return 0;
}
inline int VcCongestionDetection::register_queue_length_change_notify_callback( void (*fn)(struct QueueLengthChangeNotification *), void* pd )
{
	private_data = pd;
	queue_length_change_notify = fn;
	return 0;
}

inline void VcCongestionDetection::set_cd_algorithm_index( int32_t i ) 
{
	_cd_algorithm_index = i;
}

CLICK_ENDDECLS
#endif
