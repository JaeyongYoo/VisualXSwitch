#ifndef _TABLE_LINEAR_HH
#define _TABLE_LINEAR_HH

#include <click/config.h>
#include "../common/Table.hh"

CLICK_DECLS
/* test implementation */


template<class T>
class VcTableLinear : public VcTable<T> {
public:
	VcTableLinear(int, int);
	VcTableLinear(const char *, int, int);
	~VcTableLinear();

	virtual int add(const FlowID *fid, T **f);
	virtual int lookup(const FlowID *fid, T **f);
	virtual int removeByFlowID(const FlowID *fid);
	virtual int removeByFlow(T *f);
	virtual int dump( VcFlowClassify *, const char *buf, int len);
	virtual int time_tick();
	virtual int size();
	virtual int getAt(int i, T **f);

public:
	/* debugging functions */
	int check_validity();

public:

	T** table_flow;
	bool* valid;

	int table_max_size;
	int table_current_size;
	int table_total_entry;
	
};

template <class T>
VcTableLinear<T>::VcTableLinear(const char *, int max_queue_size, int tmsize )
{
	T* t;
	table_total_entry = 0;
	table_current_size = 0;

	table_max_size = tmsize;

	table_flow = new T*[table_max_size];
	valid = new bool[table_max_size];

	for( int i = 0; i<table_max_size; i++ )
	{
		Flow* flow;
		t = new T;

		flow = (Flow*)t;
		flow->init(max_queue_size);

		table_flow[i] = t;
		valid[i] = false;
	}
}

template <class T>
VcTableLinear<T>::VcTableLinear(int max_queue_size, int tmsize )
{
	table_total_entry = 0;
	table_current_size = 0;

	table_max_size = tmsize;

	table_flow = new T*[table_max_size];
	valid = new bool[table_max_size];

	for( int i = 0; i<table_max_size; i++ )
	{
		Flow* flow;
		T* t = new T;

		flow = (Flow*)t;
		flow->init(max_queue_size);

		table_flow[i] = t;
		valid[i] = false;
	}
}

template <class T>
VcTableLinear<T>::~VcTableLinear()
{
	for( int i = 0; i<table_max_size; i++ )
	{
		delete table_flow[i];
	}

	delete table_flow;
	delete valid;	

}

template <class T>
int VcTableLinear<T>::add(const FlowID* fid, T** f)
{
	*f = NULL;
	for (int i = 0; i < table_total_entry+1; i++)
	{
		Flow* flow = table_flow[i];
		if (!valid[i]) 
		{

			if( i == table_max_size )
			{
				printf("Error! flow table overflow ( you need to assign bigger flow table )\n");

				return -1;
			}
			else {

				flow->setup( fid );
		
				valid[i] = true;

				if( i == table_total_entry ) table_total_entry ++;
				*f = (T*)flow;
				table_current_size ++;

				return 0;
			}
		} 
	}
	return 0;
}


template <class T>
int VcTableLinear<T>::lookup(const FlowID* fid, T** f)
{
	*f = NULL;
	for( int i = 0; i < table_total_entry; i++ )
	{
		Flow* flow = (Flow*)table_flow[i];
		if( valid[i] && flow->cmp(fid) == 0 )
		{
			*f = (T*)flow;
			flow->touch();
			return 0;
		}
	}
	return -1;
}

template <class T>
int VcTableLinear<T>::removeByFlowID(const FlowID* fid)
{

	for( int i = 0; i < table_total_entry; i++ )
	{
		Flow* flow = (Flow*)table_flow[i];

		if(valid[i] && flow->cmp(fid) == 0 )
		{
			valid[i] = false;
			if( i == table_total_entry - 1 )
				table_total_entry --;

			table_current_size --;
			return 0;
		}
	}

	return 0;
}

template <class T>
int VcTableLinear<T>::size()
{
	return table_current_size;
}

template <class T>
int VcTableLinear<T>::getAt(int index, T** f)
{
	int j=0;
	*f = NULL;
	for( int i = 0; i < table_total_entry; i++ )
	{
		if( valid[i] ) {
			if( j == index ) {
				*f = table_flow[i];
				return 0;
			}
			j++;
		}
	}
	return -1;
}

template <class T>
int VcTableLinear<T>::time_tick()
{
	Flow* flow;

	for( int i = 0; i < table_total_entry; i++ )
	{
		if( valid[i] ) {
			flow = table_flow[i];
			
			if( flow->does_it_expire() == 0 )
				removeByFlow( (T*)flow );
		}
	}

	return 0;
}

template <class T>
int VcTableLinear<T>::removeByFlow(T* f)
{
	const Flow* flow = f;
	const FlowID* fid = &(flow->fid);
	int re;

	re = removeByFlowID(fid);

	return re;
}

template <class T>
int VcTableLinear<T>::dump(VcFlowClassify *clfy, const char* additional_name, int )
{
	int size = table_current_size;
	char str[4096];
	Flow* flow;
	click_chatter("FlowTableLinear size - %s [%d]\n", additional_name, size);

	for( int i = 0; i < table_total_entry; i++ )
	{
		if( valid[i] ) {
			flow = table_flow[i];
			
			clfy->to_string( &flow->fid, str, 4096 );
			click_chatter("\t[%s] [a:%d s:%d %s] ", 
					str, flow->age, flow->queue_length(), valid[i] ? "live" : "dead" );
			
			flow->toString( str, 4096 );
			click_chatter("%s\n", str );
		}
	}

	return 0;
}


template <class T>
int VcTableLinear<T>::check_validity()
{
	click_chatter("Enter VcTableLinear Check validty\n");
	for( int i = 0; i<table_max_size; i++ ) {
		Flow* flow;
		flow = table_flow[i];
		click_chatter("Flow Inspection [queue's pointer address = %x]\n", 
			&(flow->q) );
	} 
	return 0;
}

CLICK_ENDDECLS
#endif
