// -*- c-basic-offset: 4 -*-
/*
 * 
 * Jae-Yong Yoo
 *
 * Copyright (c) 2010 Gwangju Institute of Science and Technology, Korea
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, subject to the conditions listed in the Click LICENSE
 * file. These conditions include: you must preserve this copyright
 * notice, and you cannot mention the copyright holders in advertising
 * related to the Software without their permission.  The Software is
 * provided WITHOUT ANY WARRANTY, EXPRESS OR IMPLIED. This notice is a
 * summary of the Click LICENSE file; the license in that file is
 * legally binding.
 */
#include <click/config.h>
#include <clicknet/ether.h>

#include <stdarg.h>

#include "CD_CORE.hh"
#include "../FlowSchedulable.hh"
#include "../../utils/FlowNetUtils.hh"

CLICK_DECLS
/*
 * jyyoo debugging : indenting stack depth 
 */
D_DEFINE_EXTERN;

VcCDCORE::VcCDCORE()
{
	set_name("CDCORE");
}

VcCDCORE::~VcCDCORE()
{
}

int VcCDCORE::packet_enter(Flow* flow_in, const Packet*)
{
	D_START_FUNCTION;		

	FlowSchedulable* flow = (FlowSchedulable*) flow_in;
	struct CDInfo *ci = flow->getCDInfo(_cd_algorithm_index);
	struct CORE *core = NULL;

	/* this is when the flow is just arrived and, 
	 * we need to generate CORE-related variable
	 * structures */
	if( ci->cb == NULL ) {
		ci->cb = malloc( sizeof(struct CORE) );
		core = (struct CORE*)(ci->cb);
		CORE_set( core, flow,  flow->max_queue_length() );
	} 
	core = (struct CORE*)(ci->cb);

	assert( core->magicHeader_initiated == MH_INIT );

	if( CORE_queue_len_change( core, flow->queue_length() ) != 0 ) {
		click_chatter("Error: in VcCDCORE\n");
	}

	D_END_FUNCTION;		
	return 0;
}

int VcCDCORE::packet_leave(Flow* flow_in, const Packet*)
{
	D_START_FUNCTION;		

	FlowSchedulable* flow = (FlowSchedulable*) flow_in;
	struct CDInfo *ci = flow->getCDInfo(_cd_algorithm_index);
	struct CORE *core = NULL;

	/* this is when the flow is just arrived and, 
	 * we need to generate CORE-related variable
	 * structures */
	if( ci->cb == NULL ) {
		ci->cb = malloc( sizeof(struct CORE) );
		core = (struct CORE*)(ci->cb);
		CORE_set( core, flow,  flow->max_queue_length() );
	} 
	core = (struct CORE*)(ci->cb);

	assert( core->magicHeader_initiated == MH_INIT );

	if( CORE_queue_len_change( core, flow->queue_length() ) != 0 ) {
		click_chatter("Error: in VcCDCORE\n");
	}

	D_END_FUNCTION;	
	return 0;
}

double VcCDCORE::core_value(const Flow* flow_in) const
{
	FlowSchedulable* flow = (FlowSchedulable*) flow_in;
	struct CDInfo* ci = flow->getCDInfo(_cd_algorithm_index);

	if( ci->cb != NULL ) {
		struct CORE* core = (struct CORE*)(ci->cb);

		if( core->magicHeader_initiated != MH_INIT )
			return 0.0;
		return core->_core;
	}

	return 0.0;
}

double VcCDCORE::slope_value(const Flow* flow_in) const
{
	FlowSchedulable* flow = (FlowSchedulable*) flow_in;
	struct CDInfo* ci = flow->getCDInfo(_cd_algorithm_index);

	if( ci->cb != NULL ) {
		struct CORE* core = (struct CORE*)(ci->cb);

		if( core->magicHeader_initiated != MH_INIT )
			return 0.0;

		return core->_slope;
	}
	return 0.0;
}

/*******************************************************************
 * implementation of queue_QL
 *******************************************************************/
void VcCDCORE::queue_QL_clear(struct queue_QL* q)
{
	q->tail = q->head = 0;
}
void VcCDCORE::queue_QL_set(struct queue_QL* q,int s)
{
	queue_QL_clear(q);
	if( s > MAX_QL_WINDOW_SIZE )
	{
		printf("warning: queue_QL size limit\n");
		q->size = MAX_QL_WINDOW_SIZE;
	} else
		q->size = s;
}
int VcCDCORE::queue_QL_push(struct queue_QL* q,double qlen, double time)
{
	/* if it is full, automatic pop() */
	if( queue_QL_full( q ) ) 
		queue_QL_pop(q, NULL, NULL);
	q->arr_ql[q->head].qlen = qlen;
	q->arr_ql[q->head].time = time;
	q->head ++;
	if( q->head >= q->size ) q->head -= q->size;
	return 0;
}
int VcCDCORE::queue_QL_pop(struct queue_QL* q,double *qlen, double *time)
{
	if( queue_QL_empty(q) ) return 0;
	if( qlen && time )
	{
		*qlen = q->arr_ql[q->tail].qlen;
		*time = q->arr_ql[q->tail].time;
	}
	q->tail ++;
	if( q->tail >= q->size ) q->tail -= q->size;
	return 0;
}
int VcCDCORE::queue_QL_get_item(struct queue_QL* q,int index, double *qlen, double *time )
{
	if( index >= queue_QL_get_size(q) ) return -1;
	
	int i = q->head - index - 1;
	if( i < 0 ) i += q->size;
	*qlen = q->arr_ql[i].qlen;
	*time = q->arr_ql[i].time;
	return 0;
}

int VcCDCORE::queue_QL_get_size(struct queue_QL* q)
{
	int s = q->head - q->tail;
	if( s < 0 ) s += q->size;
	return s;
}
int VcCDCORE::queue_QL_empty(struct queue_QL* q)
{
	return q->head == q->tail;
}
int VcCDCORE::queue_QL_full(struct queue_QL* q) 
{
	int h = q->head;
	h ++;
	if( h >= q->size ) h -= q->size;
	return h == q->tail;
}

/*******************************************************************
 * implementation of CORE
 *******************************************************************/
void VcCDCORE::CORE_set(struct CORE* core, Flow* flow, int mq)
{

	core->magicHeader_initiated = MH_INIT;
	core->flow = flow;

	core->tma_window_size = 21;
	core->lr_size = 21;

	/* decide the window size to be 11 */
	/* we have to set it 8, since the queue itself takes 1 for maintaining condition variable */
	queue_QL_set( &core->ql_peak_sample, core->tma_window_size + 1 );

	/* decide the window size to be 9 */
	queue_QL_set( &core->ql_tma, core->lr_size + 1 );

	core->direction = 0;
	core->old_direction = 0;
	core->old_qlen = 0;

	core->_max_qlen = mq;
	core->_current_qlen = 0;
	core->_core = DBL_MAX;
}

int VcCDCORE::CORE_process_packet(struct CORE* core, int qlen, double time )
{
	/* perform peak-sampling */

	core->direction = qlen - core->old_qlen;
	core->_current_qlen = qlen;
	if( core->direction != core->old_direction )
	{
		queue_QL_push( &core->ql_peak_sample, (double)qlen, time );

		int s = queue_QL_get_size(&core->ql_peak_sample);
	
		core->old_direction = core->direction;

		/* if it has tma_window_size items, compute triangular moving average */
		if( s == core->tma_window_size )
		{
			double q, t;
			double smooth_q;
			int total = 0;
			int cnt=0;
			int coef = 0;
			int max_peak = (core->tma_window_size + 1) / 2;
			int increase_phase = 1;
			q = t = 0.0;

			for( int i = 0; i<s; i++ )
			{

				if( increase_phase == 1 )
				{
					coef ++;
				} else {
					coef --;
				}

				if( coef == max_peak ) increase_phase = 0;

				queue_QL_get_item(&core->ql_peak_sample, i, &q, &t );

				total += coef*q;
				cnt += coef;
			}
			smooth_q = (double)total/(double)cnt;

			queue_QL_push(&core->ql_tma, smooth_q, time );

			int s_tma = queue_QL_get_size(&core->ql_tma);
		
			/* if it has 5 items, do linear regression */
			if( s_tma == core->lr_size )
			{
				LR_clear(&core->lr);
				for( int i = 0; i<s_tma; i++ )
				{
					queue_QL_get_item( &core->ql_tma, i, &q, &t );

					/* time is x-axis and queue length is y-axis */
					LR_addXY( &core->lr, t, q ); 
				}
				int residual = core->_max_qlen - qlen;
				core->_slope = LR_getB( &core->lr );
				core->_core = (double)residual/core->_slope;

				if( core->_core < 0.0 || core->_core > 1000.0 ) core->_core = 1000.0;
			}
		}
	}
	core->old_qlen = qlen;

	/* indicate congestion */
	return CORE_congestion_indication( core, 0.1 );
}

int VcCDCORE::CORE_queue_len_change(struct CORE* core, int qlen)
{
	if( core->monitoring_interval_count == 0 )
	{
		gettimeofday( &core->tv_start, NULL );
	}
	core->monitoring_interval_count ++;

	struct timeval tv;
	uint32_t offset_off_tv;
	double time_diff;
	int queuelen_diff;
	gettimeofday( &tv, NULL );

	time_diff = timevaldiff( &tv, &core->tv_last_update_per_packet );
	queuelen_diff = core->queuelen_last_update_per_packet - qlen;

	core->queuelen_last_update_per_packet = qlen;
	core->tv_last_update_per_packet = tv;

	offset_off_tv = timevaldiff( &core->tv_start, &tv );	

	return CORE_process_packet( core, qlen, offset_off_tv/1000000.0 );

}
int VcCDCORE::CORE_congestion_indication(struct CORE* core, double feedback_delay)
{
	D_START_FUNCTION;

	/* need to compute time-buffer due to the delay of window-based moving average */
	double adapted_moving_average_interval = 0.1;
	
	/* should be monitored from the delay of cross-correlation function 
	   between the queue length and the smoothed queue length */
	double time_buffer = 0.5;
	double comparison_time;

	if( adapted_moving_average_interval > feedback_delay ) {
		comparison_time = adapted_moving_average_interval + time_buffer;
	}
	else {
		comparison_time = feedback_delay + time_buffer;
	}

	/* by considering the current queue level, make a scaling */
	double scale_factor = 1.0;
	
	/* at the moment (2010-09-03), I don't know the systematic relationship 
         * between queue length and scale factor, so just apply linear mapping 
         * from -infinity to 1 */
	/* DO NOT USE IT YET WITHOUT ANALYSIS */
	scale_factor += 3*(double)core->_current_qlen / (double)core->_max_qlen;
	scale_factor = 1.0;

	struct CongestionNotification cn;
	cn.object = private_data;
	cn.fid = &(core->flow->fid);
	/* it is ok to set the packet NULL
	 * since we have valid fid */
	cn.packet = NULL; 

	if( comparison_time*scale_factor > core->_core ) {
		if( congest_detected ) {
			congest_detected( &cn );
		} else {
			return -1;
		}
	} else {
		if( nocongest_detected ) {
			nocongest_detected( &cn );
		} else {
			return -1;
		}
	}

	D_END_FUNCTION;		
	return 0;
}

/* return unit is Mbps */
double VcCDCORE::CORE_exceeded_rate(struct CORE* core)
{
	/* we should observe this pkt_size average, but 
	 * for early implemention, we use 1300 which is the video packet size */

	double pkt_size = 1300.0; 
	return core->_slope * pkt_size * 8.0 / 1000000.0; 
}

/*******************************************************************
 * implementation of LinearRegression
 *******************************************************************/
void VcCDCORE::LR_addXY(struct LinearRegression* lr, const double& x, const double& y)
{
	lr->n++;
	lr->sumX += x;
	lr->sumY += y;
	lr->sumXsquared += x * x;
	lr->sumYsquared += y * y;
	lr->sumXY += x * y;
	LR_Calculate(lr);
}

void VcCDCORE::LR_Calculate(struct LinearRegression* lr)
{
	if( LR_haveData(lr) ) {
		if( fabs( double(lr->n) * lr->sumXsquared - lr->sumX * lr->sumX) > DBL_EPSILON ) {
			lr->b = ( double(lr->n) * lr->sumXY - lr->sumY * lr->sumX) /
				( double(lr->n) * lr->sumXsquared - lr->sumX * lr->sumX);
			lr->a = (lr->sumY - lr->b * lr->sumX) / double(lr->n);

			double sx = lr->b * ( lr->sumXY - lr->sumX * lr->sumY / double(lr->n) );
			double sy2 = lr->sumYsquared - lr->sumY * lr->sumY / double(lr->n);
			double sy = sy2 - sx;

			lr->coefD = sx / sy2;
			lr->coefC = sqrt(lr->coefD);
			lr->stdError = sqrt(sy / double(lr->n - 2));
		}
		else {
			lr->a = lr->b = lr->coefD = lr->coefC = lr->stdError = 0.0;
		}
	}
}

	CLICK_ENDDECLS
ELEMENT_PROVIDES(VcCDCORE)
