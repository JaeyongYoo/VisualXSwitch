#ifndef DHPARAMS_H
#define DHPARAMS_H 1
#include <click/config.h>

CLICK_DECLS
#include <openssl/dh.h>

DH *get_dh1024(void);
DH *get_dh2048(void);
DH *get_dh4096(void);

CLICK_ENDDECLS
#endif /* dhparams.h */
