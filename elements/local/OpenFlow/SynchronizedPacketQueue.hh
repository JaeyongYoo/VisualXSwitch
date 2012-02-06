// -*- c-basic-offset: 4 -*-
#ifndef RCONN_SYNCHRONIZED_QUEUE_HH
#define RCONN_SYNCHRONIZED_QUEUE_HH
CLICK_DECLS
#include <click/element.hh>
#include <semaphore.h>  /* Semaphore */

struct rconn_remote;

struct buf_rconn {
	void *msg;
	int length;
	struct rconn_remote *sender;
	int32_t xid;
};


/* jyyoo TODO: should make this synchornized by lock-free */

struct SynchronizedPacketQueue {
public:
	inline int init(int size);
	inline int destroy();
	inline int max_size() const { return _size; };

	inline struct buf_rconn pop();
	inline int	push( struct buf_rconn * p);
	inline int length() { return (_h-_t)>=0?(_h - _t) : (_h-_t+_size); };

private:
        sem_t mutex;

public:
	struct buf_rconn *_storage;
	int	_size;
	int	_h;
	int	_t;
};

inline int SynchronizedPacketQueue::init(int size)
{

        sem_init(&mutex, 0, 1);      /* initialize mutex to 1 - binary semaphore */
	_storage = (struct buf_rconn *) CLICK_LALLOC(sizeof(struct buf_rconn) * (size + 1));
	_t = 0;
	_h = 0;

        if (_storage == 0)
        {
                _size = 0;
                return -1;
        }
        _size = size;
	return 0;
}
inline int SynchronizedPacketQueue::destroy()
{
	sem_destroy(&mutex); /* destroy semaphore */
        CLICK_LFREE(_storage, sizeof(struct buf_rconn) * (_size + 1));
        return 0;
}
/* TODO this returning of buf_rconn is very inefficient design
 * but, just go for it for the proof of concept. 
 * revisit here and please fix this */
inline struct buf_rconn SynchronizedPacketQueue::pop()
{

	sem_wait(&mutex);       /* down semaphore */
        struct buf_rconn p;
	p.msg = NULL;
        if( _h == _t ) 
	{
		sem_post(&mutex);	/* up semaphore */
		return p;
	}
	memcpy( &p, &(_storage[_t]), sizeof( struct buf_rconn ) );
        _t ++;
        if( _t >= _size ) _t = 0;

	sem_post(&mutex);
        return p;
}

inline int SynchronizedPacketQueue::push(struct buf_rconn *p)
{
	sem_wait(&mutex);       /* down semaphore */
        int next_h = _h + 1;
        if( next_h >= _size ) next_h -= _size;
        if( next_h == _t ) {
		sem_post(&mutex);
		return -1; /* queue is full */
	}
        _storage[_h] = *p;
        _h = next_h;
	sem_post(&mutex);
        return 0;
}

CLICK_ENDDECLS
#endif
