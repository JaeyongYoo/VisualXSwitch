#ifndef __FLOW_COMMON_HH__
#define __FLOW_COMMON_HH__

#include <click/config.h>


#define TL_MAX_FLOW 64

/* 
 * if you see segmentation fault, 
 * it would be a good first step 
 * by setting this DO_DEBUG 1
 * and watch which function is the one
 * that generates the seg fault.
 */
#define DO_DEBUG 0


#if DO_DEBUG 

#define D_DEFINE_EXTERN \
		extern int	g_stack_depth;\
		extern uint16_t *g_ptr_interest;

#define D_DEFINE_BODY	\
		int g_stack_depth=0;\
		uint16_t *g_ptr_interest;

#define REGISTER_UINT16_T(p) g_ptr_interest=(uint16_t*)(p);


#define D_START_FUNCTION {\
	for(int i=0;i<g_stack_depth; i++) printf("  ");	\
	g_stack_depth++;\
	click_chatter("debug [start] ===> %s\n", __PRETTY_FUNCTION__ ); \
	if( g_ptr_interest ) printf("value:%d\n", *g_ptr_interest );\
	}

#define D_END_FUNCTION {\
	g_stack_depth--;\
	for(int i=0;i<g_stack_depth; i++) printf("  ");	\
	click_chatter("debug [end] ===> %s\n", __PRETTY_FUNCTION__ ); \
	if( g_ptr_interest ) printf("value:%d\n", *g_ptr_interest );\
	}

#else

#define D_DEFINE_EXTERN
#define D_DEFINE_BODY
#define D_START_FUNCTION 
#define D_END_FUNCTION 

#endif

#define BASE_TIMER_CLOCK 1000


#endif
