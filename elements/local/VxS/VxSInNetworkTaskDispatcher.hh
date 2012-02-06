#ifndef __VXS_INNETWORK_TASK_DISPATCHER__
#define __VXS_INNETWORK_TASK_DISPATCHER__

#include <click/config.h>
#include <click/element.hh>
#include <pthread.h>
#include <list>

#define VXS_MAX_THREADS 32
#define VXS_DISPATCH_INTERVAL 50000 /* micro seconds */

CLICK_DECLS
class VxSInNetworkTaskQueue;
class VxSInNetworkTask;
class VxSInNetworkCompute;

class VxSInNetworkTaskDispatcher {

public:
	VxSInNetworkTaskDispatcher(VxSInNetworkTaskQueue *incoming, VxSInNetworkTaskQueue *outgoing);
	~VxSInNetworkTaskDispatcher();

	/* start a set of threads for doing this job 
	 * the number of threads is @thread_num
	 */
	int startDispatching( int thread_num );

	/* a thread-called functions; 
	 * double underbar indicates that it is thread-context
	 */
	int __dispatch();

	int alive() { return _on_the_go; };

	/* initialize this task dispatcher */
	void init_computes();

	/* run the specific action on the task */
	int run_action_on_task( VxSInNetworkTask *task, struct ofp_action_header *ah );

	
	VxSInNetworkCompute * lookupCompute(const char *name);
private:
	/* a list of computes that support this dispatcher */
	std::list<VxSInNetworkCompute *> _list_computes;

	/* task queue for getting tasks */
	VxSInNetworkTaskQueue *_task_queue_incoming;
	VxSInNetworkTaskQueue *_task_queue_outgoing;

	/* number of all live threads */
	int _num_of_live_threads;
	
	/* indicates whether threads should run or not */
	int _on_the_go;

	/* thread handlers */
	pthread_t _thread_handles[VXS_MAX_THREADS];

};


CLICK_ENDDECLS
#endif
