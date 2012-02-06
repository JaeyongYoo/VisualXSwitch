#ifndef __MULTIOP_FLOWSHAPERABLE_H__
#define __MULTIOP_FLOWSHAPERABLE_H__

#include <click/config.h>
CLICK_DECLS

#include <float.h>
#include <clicknet/wifi.h>

#include "../common/Flow.hh"
#include "../common/FlowCommon.hh"
#include "../common/PacketQueue.hh"
#include "../common/BWShape.hh"


/* 
 * description: 
 * structure for flow statistics 
 */
struct FlowBWShaperableStat {

};



/*!
 * data structures 
 */
struct BWShapeInfo {
	uint16_t* ref_queuelen;

	/* 32 bytes control buffer */
	/* could be any data */
	uint8_t cb[32];
};



#define SHAPE_STATUS_QLEN_MONITOR_SEQ 0x00000001

class FlowBWShaperable : public Flow {
public:
	FlowBWShaperable() {_vcShape=NULL;};
	~FlowBWShaperable() {};
	
	/*!
	 * overiding functions
	 */
	virtual void clear();


	/**********************************************************
	 * Shaperable Flow
	 **********************************************************/

	struct BWShapeInfo si;

	/* schedule status */
	uint32_t bwshape_status;

	inline uint32_t getBWShapeStatus();
	inline void setBWShapeStatus(uint32_t);

	BWShapeInfo* getBWShapeInfo();

	inline void setLowerLayerFlow( Flow* f );
	inline Flow* getLowerLayerFlow();
	inline void hookLowerLayerQueueLen();

	void toString( char* buf, int len );
	void setShaper( VcBWShape* vcb ) { _vcShape = vcb; }
private:

	VcBWShape	*_vcShape;

private:
	
	/* a hack to find and connect to the lower layer (in this case scheduler) */
	Flow* lower_layer_flow;
	

};

inline uint32_t FlowBWShaperable::getBWShapeStatus()
{
	return bwshape_status;
}
inline void FlowBWShaperable::setBWShapeStatus(uint32_t ss)
{
	bwshape_status = ss;
}
inline BWShapeInfo* FlowBWShaperable::getBWShapeInfo()
{	
	return &si;
}
inline void FlowBWShaperable::setLowerLayerFlow( Flow* f )
{
	lower_layer_flow = f;
}
inline Flow* FlowBWShaperable::getLowerLayerFlow()
{
	return lower_layer_flow;
}

inline void FlowBWShaperable::hookLowerLayerQueueLen()
{
	if( lower_layer_flow )
	{
		si.ref_queuelen = lower_layer_flow->queue_length_ref();
	}
}

CLICK_ENDDECLS

#endif
