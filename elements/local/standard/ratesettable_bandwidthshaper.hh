// -*- c-basic-offset: 4 -*-
#ifndef CLICK_RATESETTABLE_BANDWIDTHSHAPER_HH
#define CLICK_RATESETTABLE_BANDWIDTHSHAPER_HH
#include "../../standard/shaper.hh"
CLICK_DECLS

/*
 * =c
 * RateSettableBandwidthShaper(RATE)
 * =s shaping
 * shapes traffic to maximum rate (bytes/s)
 * =processing
 * Pull
 * =d
 *
 * RateSettableBandwidthShaper is a pull element that allows a maximum bandwidth of
 * RATE to pass through.  That is, output traffic is shaped to RATE.
 * If a RateSettableBandwidthShaper receives a large number of
 * evenly-spaced pull requests, then it will emit packets at the specified
 * RATE with low burstiness.
 *
 * =h rate read/write
 *
 * Returns or sets the RATE parameter.
 *
 * =a Shaper, BandwidthRatedSplitter, BandwidthRatedUnqueue */

class RateSettableBandwidthShaper : public Shaper { public:

    RateSettableBandwidthShaper();
    ~RateSettableBandwidthShaper();

    const char *class_name() const	{ return "RateSettableBandwidthShaper"; }

    void set_rate(int r) { _rate.set_rate(r, NULL); }


    Packet *pull(int);

};

CLICK_ENDDECLS
#endif
