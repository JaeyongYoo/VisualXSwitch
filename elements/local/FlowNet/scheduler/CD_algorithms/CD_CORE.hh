#ifndef __CD_CORE_H__
#define __CD_CORE_H__

CLICK_DECLS
#include <float.h>

#include "../../common/CD.hh"

/*******************************************************************
 * For CORE 
 ******************************************************************/

struct LinearRegression
{
	long n;             // number of data points input so far
	double sumX, sumY;  // sums of x and y
	double sumXsquared, // sum of x squares
	       sumYsquared; // sum y squares
	double sumXY;       // sum of x*y

	double a, b;        // coefficients of f(x) = a + b*x
	double coefD,       // coefficient of determination
	       coefC,       // coefficient of correlation
	       stdError;    // standard error of estimate

};

/* to maintain window of queue length */
struct queue_length {
	double qlen;
	double time;
	/* do we use linked list? or array? */
	/* array performs better in this case */
};


/* queue of queue length */
#define MAX_QL_WINDOW_SIZE 25
struct queue_QL {
	struct queue_length arr_ql[MAX_QL_WINDOW_SIZE];
	int size;
	int head;
	int tail;
};

#define MH_INIT 0x1f8edb38

/*
 * TODO: this CORE structure should be smaller than CBInfo structure.
 * I need to add assertion that checks this.
 * Very important. Do this! 
 */
struct CORE {

	uint32_t magicHeader_initiated;	
	Flow*	flow;

	/* core computation */
	int _max_qlen;
	int _current_qlen;
	double _slope;
	double _core;

	/* important variable */
	int tma_window_size;
	int lr_size;

	struct timeval tv_start;

	/* peak sampling */
	struct queue_QL ql_peak_sample;
	int direction;
	int old_direction;
	int old_qlen;

	/* Triangular Moving Average */
	struct queue_QL ql_tma;

	/* Linear Regression */
	LinearRegression lr;


	/* for analysis */
	int queuelen_last_update_per_packet;
	int queuelen_last_update_timer;

	struct timeval tv_last_update_per_packet;
	struct timeval tv_last_update_timer;

	int monitoring_interval;
	int monitoring_interval_count;
};

/* COngestion Residual timE */
class VcCDCORE : public VcCongestionDetection {
public:
	VcCDCORE();
	~VcCDCORE();

        virtual int packet_enter(Flow* flow, const Packet* p);
        virtual int packet_leave(Flow* flow, const Packet* p);
	
	double core_value(const Flow* flow) const;
	double slope_value(const Flow* flow) const;
	
private:


	/* functions that handle CORE */
	int CORE_queue_len_change(struct CORE* core, int qlen);
	void CORE_set(struct CORE* core, Flow* flow, int max_qlen);
	int CORE_process_packet(struct CORE* core,  int qlen, double time );
	int CORE_congestion_indication(struct CORE* core, double feedback_delay);
	double CORE_exceeded_rate(struct CORE* core);



	/* functions that handle queue_QL */
	void queue_QL_clear(struct queue_QL*);
	void queue_QL_set(struct queue_QL*,int s);
	int queue_QL_push(struct queue_QL*, double qlen, double time );
	int queue_QL_pop(struct queue_QL*, double *qlen, double *time );
	int queue_QL_get_item(struct queue_QL*, int index, double *qlen, double *time );
	int queue_QL_get_size(struct queue_QL*);
	int queue_QL_empty(struct queue_QL*);
	int queue_QL_full(struct queue_QL*);

	/* functions that handle LinearRegression */
	void LR_addXY(struct LinearRegression*, const double& x, const double& y);
	void LR_clear(struct LinearRegression* l) { l->sumX = l->sumY = l->sumXsquared = l->sumYsquared = l->sumXY = l->n = 0; };

	// Must have at least 3 points to calculate
	// standard error of estimate.  Do we have enough data?
	int LR_haveData(struct LinearRegression* l) const { return (l->n > 2 ? 1 : 0); };
	long LR_items(struct LinearRegression* l) const { return l->n; };

	double LR_getA(struct LinearRegression* l) const { return l->a; };
	double LR_getB(struct LinearRegression* l) const { return l->b; };

	void LR_Calculate(struct LinearRegression*);   // calculate coefficients

};


CLICK_ENDDECLS

#endif
