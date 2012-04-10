#ifndef __VXS_INNETWORK_SEGMENT__
#define __VXS_INNETWORK_SEGMENT__

#include <click/config.h>
#include <click/element.hh>
#include "../OpenFlow/lib/switch-flow.hh"

CLICK_DECLS

#define VXS_MAX_ACTION_HEADER 512

/**
 * Desc: An abstract class for storing a segment which differs depending on
 *       the media type of a flow.
 */
class VxSInNetworkSegment {
public:
	/* no particular variables or functions */
	VxSInNetworkSegment();
	~VxSInNetworkSegment();

	int32_t getActionLen() { return _action_len; };
	uint8_t *getActionHeader() { return _action_header; };
	uint8_t *getActionHeaderProgramCounter() { return _action_header_program_counter; };
	int32_t getActionOffset() { return _action_header_program_counter - _action_header; };

	int setActionHeader( const uint8_t *d, uint32_t size );
	uint8_t *getNextActionHeader();
	virtual void print_to_chatter();
	virtual VxSInNetworkSegment * clone() = 0;

protected:
	/* 
	 * Desc: meta data for actions 
	 * we use static array rather than using a memory allocation
	 * for the sake of the simplicity
	 * 
	 * XXX: this static allocation of action_header needs a discussion
	 * since this allocation would be the waste of time and space because
	 * the original action_header in OpenFlow dp_act is unlikely changing.
	 * but, if it is under changing and there always the possibility of 
	 * integrity failure since TaskDispatcher works with threads.
	 * then, it requires locking to protect this thing and to avoid this ugly
	 * situation, we sacrifice some computing time and space.
	 */
	uint8_t _action_header[VXS_MAX_ACTION_HEADER];
	uint8_t *_action_header_program_counter;
	int32_t _action_len;
	
};

CLICK_ENDDECLS
#endif
