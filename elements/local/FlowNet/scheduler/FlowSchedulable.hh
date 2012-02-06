#ifndef __MULTIOP_FLOWSCHEDULABLE_H__
#define __MULTIOP_FLOWSCHEDULABLE_H__

#include <click/config.h>
CLICK_DECLS

#include <float.h>
#include <clicknet/wifi.h>

#include "../common/Flow.hh"
#include "../common/FlowCommon.hh"
#include "../common/PacketQueue.hh"

struct FlowID;
struct SchedInfo;
struct NexthopInfo;


/* 
 * description: 
 * structure for flow statistics 
 */
struct FlowSchedulableStat {

	/* performance status [count]; */
	int sched_VI;
	int sched_VO;
	int sched_BE;
	int sched_BK;
	int sched_P4;
	int sched_P5;
	int sched_P6;
	int sched_P7;
	int sched_uncovered;
};

/*!
 * data structures 
 */

struct E2ESignalInfo {
	float aggregated_path_quality;
	float throughput;
};


struct NexthopInfo {
	int queuelen;
	uint16_t queuelen_monitor_seq;
	float queuelen_average;

	uint8_t macaddr[WIFI_ADDR_LEN];
	IPAddress ipaddr;
};

/*
 * common scheduling information 
 * that is used universally by all
 * the schedulers 
 */
struct SchedInfo {
	/* for nexthop queue monitoring */
	struct NexthopInfo ni;
	/* end-to-end signalling aggregated PER of a path */
	struct E2ESignalInfo ei;
	/* for backpressure scheduling */
	double backpressure_value;
	/* last schedule time */
	struct timeval tv_last_schedule;

	/* 
	 * 1024 bytes control buffer 
	 * Could be any data depending on scheduling algorithms 
	 * originally, it was 1024 byte control buffer 
	 * but in order to anticipate the outer scheduling algorithm
	 * we divide cb in half and let inner scheduling algorithm
	 * takes the lower half and the outer scheduling algorithm
	 * takes the upper half.
	 */

//	uint8_t cb[1024];
	uint8_t cb_lhalf[512];
	uint8_t cb_uhalf[512];

};

/* 
 * congestion detection algorithm information
 * that is used universally by all
 * the congestion detection algorithms
 */ 
struct CDInfo {
	
	/* common structure */
	int32_t algorithm_index;

	/* Could be any data depending on scheduling algorithms */
	void *cb;
};


#define SCHEDULE_STATUS_QLEN_MONITOR_SEQ 0x00000001
#define MAX_CD_ALGORITHMS 	64

class FlowSchedulable : public Flow {
public:
	FlowSchedulable();
	~FlowSchedulable();
	
	/*!
	 * overiding functions
	 */
	virtual void clear();
	virtual void toString(char* buf, int);

	/**********************************************************
	 * Schedulable Flow
	 **********************************************************/

	struct SchedInfo si;
	struct CDInfo ci[MAX_CD_ALGORITHMS];

	/* schedule status */
	uint32_t schedule_status;


	/* statistics of this queue */
	struct FlowSchedulableStat stat;


	void setNexthopInfo(uint8_t* macaddr, IPAddress ip);

	int update_nexthop_queuelen();

	inline struct SchedInfo* getSchedInfo();
	int update_nexthop_queuelen(Packet* p);

	inline struct CDInfo* getCDInfo(int32_t);

	inline uint32_t getScheduleStatus();
	inline void setScheduleStatus(uint32_t);
};

inline uint32_t FlowSchedulable::getScheduleStatus()
{
	return schedule_status;
}

inline void FlowSchedulable::setScheduleStatus(uint32_t ss)
{
	schedule_status = ss;
}

inline SchedInfo* FlowSchedulable::getSchedInfo()
{	
	return &si;
}

inline CDInfo* FlowSchedulable::getCDInfo(int32_t i)
{
	return &ci[i];
}

CLICK_ENDDECLS

#endif
