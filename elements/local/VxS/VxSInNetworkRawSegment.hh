#ifndef __VXS_INNETWORK_RAW_SEGMENT__
#define __VXS_INNETWORK_RAW_SEGMENT__

#include <click/config.h>
#include "VxSInNetworkSegment.hh"

CLICK_DECLS

void checksumIP_v2( Packet* p, int offset );
void checksumUDP_v2( Packet* p, int offset );

class VxSInNetworkRawSegment : public VxSInNetworkSegment {
private:
	/* max segment size */
	uint32_t _segment_size;

	/* segment data */
	uint8_t *_segment;

	/* currently written size */
	uint32_t _written_size;

	/* media-property (bytes per pixel block) of this contents */
	uint32_t _Bpb;

	/* height and width */
	uint32_t _height;
	uint32_t _width;

public:
	VxSInNetworkRawSegment(uint32_t seg_size);
	~VxSInNetworkRawSegment();

	/* 
	 * Desc: push data into segment 
	 * 
	 * returns the number of bytes written
	 */
	uint32_t push(uint8_t *data, int size);
	uint32_t prepareSegment( uint32_t size );

	int isFull() { return _written_size == _segment_size; };
	uint8_t * getSegment() { return _segment; };
	uint32_t getSize() { return _segment_size; };
	uint32_t getWrittenSize() { return _written_size; };
	void setWrittenSize(uint32_t t) { _written_size = t; };

	void setBytePerPixelBlocks(uint32_t src) { _Bpb = src; };
	void setWidthHeight(uint32_t w, uint32_t h) { _width = w; _height = h;};

	uint32_t getBytePerPixelBlocks() { return _Bpb; };
	uint32_t getCurrentNumPixelBlocks() { return _written_size / _Bpb; };
	uint32_t getMaxNumPixelBlocks() { return _height * _width / (4*4); /* TODO: make this as a variable (4x4) pixel block*/ };
	uint32_t getWidth() { return _width; };
	uint32_t getHeight() { return _height; };

	/* 
	 * this is the part that deals about packetizing this segment
 	 * for packetizing the segment, we use _proc_Bpb as the unit for packetizing
	 */
	uint32_t getNumberOfPackets(uint32_t packet_size);
	Packet * packetize(uint32_t data_size, uint8_t *network_header, uint32_t network_header_len);

	VxSInNetworkSegment * clone();
	void copy(VxSInNetworkRawSegment *raw);
};

CLICK_ENDDECLS
#endif
