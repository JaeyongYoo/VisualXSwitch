#ifndef _CLASSIFY_NO_CLASSIFY_HH
#define _CLASSIFY_NO_CLASSIFY_HH

#include <click/config.h>

#include "../common/Flow.hh"
#include "../common/FlowClassify.hh"

CLICK_DECLS

class VcNoClassify : public VcFlowClassify {
public:
        VcNoClassify();
        ~VcNoClassify();
public:

        int classify(const Packet* p, struct FlowID* fid);
        int to_string(const struct FlowID* fid, char* buf, int len);
};


CLICK_ENDDECLS

#endif
