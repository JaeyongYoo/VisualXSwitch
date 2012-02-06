// -*- c-basic-offset: 4 -*-
#ifndef _SCHEDULE_OUTER_HH
#define _SCHEDULE_OUTER_HH
#include <click/glue.hh>
#include <click/element.hh>
#include <click/timer.hh>
#include <click/notifier.hh>

#include "Algorithm.hh"
#include "FlowCommon.hh"
#include "Flow.hh"
#include "Table.hh"

CLICK_DECLS

#define	SCHEDULE_OUTER_LOOP_TIMESCALE	1000 // milli-seconds

class VcSchedule;
class PFSchedFW;

/*!
 * virtual functions of scheduling algorithms
 */

class VcScheduleOuter : public Algorithm {
public:
        VcScheduleOuter() {};
        ~VcScheduleOuter() {};
public:

	/* generic outer-loop scheduling functions */
	virtual void periodic_monitor( int *next_period ) = 0;
	virtual void act() = 0;
	virtual int bind( VcSchedule *, PFSchedFW * ) = 0;

protected:
	VcSchedule* _vcSched;
};

CLICK_ENDDECLS
#endif
