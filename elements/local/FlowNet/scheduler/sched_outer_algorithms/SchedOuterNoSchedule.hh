#ifndef __SCHED_OUTER_NO_H__
#define __SCHED_OUTER_NO_H__

#include <click/config.h>

#include "../../common/FlowCommon.hh"
#include "../../common/ScheduleOuter.hh"

CLICK_DECLS


class PFSchedFW;
/* 
 * WBS stands for Weighted Backpressure Scheduling 
 */
class VcScheduleOuterNoSchedule : public VcScheduleOuter {
public:
	VcScheduleOuterNoSchedule();
	~VcScheduleOuterNoSchedule();
public:

        void periodic_monitor( int *next_period );
        void act();
        int bind( VcSchedule *, PFSchedFW * );

private:
};


CLICK_ENDDECLS

#endif
