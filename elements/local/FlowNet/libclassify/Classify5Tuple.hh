#ifndef _CLASSIFY_5TUPLE_HH_
#define _CLASSIFY_5TUPLE_HH_

#include <click/config.h>

#include "../common/FlowCommon.hh"
#include "../common/Flow.hh"
#include "../common/FlowClassify.hh"

CLICK_DECLS

class Vc5TupleClassify : public VcFlowClassify {
public:
        Vc5TupleClassify();
        ~Vc5TupleClassify();
public:

        int classify(const Packet* p, struct FlowID* fid);
        int to_string(const struct FlowID* fid, char* buf, int len);

};

CLICK_ENDDECLS

#endif
