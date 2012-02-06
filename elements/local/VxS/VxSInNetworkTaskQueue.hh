#ifndef __VXS_INNETWORK_TASK_QUEUE__
#define __VXS_INNETWORK_TASK_QUEUE__

#include <click/config.h>
#include <click/element.hh>
#include <semaphore.h>  
#include <list>
#include "VxSInNetworkBatchManager.hh" 

#define VXS_MAX_NETWORK_HEADER 14 + 8 + 20 + 20 + 20

CLICK_DECLS

class VxSInNetworkSegment;
class VxSInNetworkFlowBatcher;

/**
 * This is more likely a structure, 
 * but we just use class for this structure-purpose
 */
class VxSInNetworkTask {
private:
	/*
	 *  inputs for this task
	 */
	/* for holding a data to be processed */
	VxSInNetworkSegment *segment; 

	/* for holding a flow to know which process will be done */
	VxSInNetworkFlowBatcher *flow;

	/* the network headers for corresponding to this task */
	uint8_t network_headers[VXS_MAX_NETWORK_HEADER];
	int network_header_len;
	uint16_t in_port;
	uint16_t out_port;

	/* task-done field */
	bool	_task_done;
	int32_t _return_value;

public:
	void set( VxSInNetworkSegment *, VxSInNetworkFlowBatcher *, struct ofpbuf *ob, int len );

	struct ofp_action_header * getNextActionHeader();
	
	VxSInNetworkSegment * getSegment() { return segment; };
	
	void taskDone() { _task_done = true; };
	int isTaskDone() { return _task_done; };
	void setReturnValue( int32_t r ) { _return_value = r; };
	int getReturnValue() { return _return_value; };
	VxSInNetworkFlowBatcher * getFlow() { return flow; };
	
	uint8_t * getNetworkHeader() { return network_headers; };
	int getNetworkHeaderLen() { return network_header_len; };

	void set_prev_in_port(uint16_t i, uint16_t o) { in_port = i; out_port = o; };

	uint16_t getInPort() { return in_port; };
	uint16_t getOutPort() { return out_port; };

	/* debug function */
	void print_to_chatter();
};

/**
 * A single multi-thread safe queue for passing 
 * tasks between network and computing devices 
 */
class VxSInNetworkTaskQueue {
public:
	VxSInNetworkTaskQueue();
	~VxSInNetworkTaskQueue();

	int pushTask( VxSInNetworkTask *task );
	VxSInNetworkTask * popTask();
	int size() { return _tasks.size(); }

private:
	std::list<VxSInNetworkTask *> _tasks;
	sem_t _sem_tasks;

};

CLICK_ENDDECLS
#endif
