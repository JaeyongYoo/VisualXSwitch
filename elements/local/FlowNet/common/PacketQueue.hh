// -*- c-basic-offset: 4 -*-
#ifndef CLICK_MULTIOPQUEUE_HH
#define CLICK_MULTIOPQUEUE_HH
CLICK_DECLS
#include <click/config.h>
#include <click/element.hh>

struct PacketQueue {
public:
	inline int init(int size);
	inline int destroy();
	inline int max_size() const { return _size; };

	inline Packet* pop();
	inline Packet* observe();
	inline int	push(Packet* p);
	inline int length() { return (_h-_t)>=0?(_h - _t) : (_h-_t+_size); };

	inline int check_validity();

private:
public:
	Packet* volatile *_storage;
	int	_size;
	int	_h;
	int	_t;
};

inline int PacketQueue::init(int size)
{
	_storage = (Packet **) CLICK_LALLOC(sizeof(Packet *) * (size + 1));
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

inline int PacketQueue::destroy()
{
        CLICK_LFREE(_storage, sizeof(Packet *) * (_size + 1));
        return 0;
}

inline Packet* PacketQueue::pop()
{
        Packet* p;
        if( _h == _t ) return NULL;
        p = _storage[_t];
        _t ++;
        if( _t >= _size ) _t = 0;
        return p;
}

inline Packet* PacketQueue::observe()
{
        Packet* p;
        if( _h == _t ) return NULL;
        p = _storage[_t];
        return p;
}

inline int PacketQueue::check_validity()
{
	int size = _h - _t;

	if( size < 0 ) size += _size;
	for( int i = 0; i < size; i++ )
	{
		int j = _t + i;		
		if( j >= _size ) j -= _size;
		if( ((uint32_t) _storage[j]) < 0x0000ffff )
		{
			printf("Error!! validaty failed (%d, %d) %d, %x\n", _h, _t, j, (uint32_t)_storage[j]);
			return 0;
		}
	}
	printf("Validty OK\n");
	return 0;
}
inline int PacketQueue::push(Packet* p)
{
        int next_h = _h + 1;
        if( next_h >= _size ) next_h -= _size;
        if( next_h == _t ) return -1; /* queue is full */
        _storage[_h] = p;
        _h = next_h;
        return 0;
}

CLICK_ENDDECLS
#endif
