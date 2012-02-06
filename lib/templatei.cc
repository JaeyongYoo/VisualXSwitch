/*
 * templatei.cc -- Template instantiations
 * Eddie Kohler
 *
 * Copyright (c) 1999-2000 Massachusetts Institute of Technology
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, subject to the conditions
 * listed in the Click LICENSE file. These conditions include: you must
 * preserve this copyright notice, and you cannot mention the copyright
 * holders in advertising related to the Software without their permission.
 * The Software is provided WITHOUT ANY WARRANTY, EXPRESS OR IMPLIED. This
 * notice is a summary of the Click LICENSE file; the license in that file is
 * legally binding.
 */

#include <click/config.h>

#include <click/string.hh>
#include <click/router.hh>
#include <click/ipflowid.hh>
#include <click/ipaddress.hh>
#include <click/etheraddress.hh>
CLICK_DECLS

template class Vector<int>;
template class Vector<unsigned int>;
template class Vector<String>;
template class Vector<IPAddress>;

CLICK_ENDDECLS
