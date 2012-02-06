#ifndef __MULTIOP_UTILS_H__
#define __MULTIOP_UTILS_H__

CLICK_DECLS


#include <click/ipaddress.hh>
#include <click/etheraddress.hh>
#include <click/string.hh>
#include <click/confparse.hh>
#include <click/timer.hh>
#include <clicknet/ip.h>
#include <clicknet/ether.h>
#include <clicknet/wifi.h>
#include <clicknet/udp.h>
#include <sys/socket.h>
#include <arpa/inet.h>


/* just for compilation of multiop_utils.hh */
class DummyClass {
};


/* reset UDP checksum */
void checksumUDP( Packet* p, int offset = 0 ); /* supposed to receive IP hdr including packet */
void checksumIP( Packet* p, int offset = 0 ); /* supposed to receive IP hdr including packet */

void sprintf_time( char*buf, struct timeval* tv );

/* time difference between timeval */
int64_t timevaldiff( struct timeval* starttime, struct timeval *finishtime );

void sprintf_flow( char* buf, uint32_t src, uint32_t dst, uint16_t sport, uint16_t dport );
void sprintf_flow( char* buf, struct in_addr src, struct in_addr dst, uint16_t sport, uint16_t dport );
void sprintf_flow( char* buf, IPAddress src, IPAddress dst, uint16_t sport, uint16_t dport );

void sprintf_ether( char* buf, click_ether* );
void sprintf_ip( char* buf, click_ip* );
void sprintf_udp( char* buf, click_udp* );

void printf_ether( click_ether* );
void printf_ip( click_ip* );
void printf_udp( click_udp* );

void sprintf_mac( char* buf, uint8_t* macaddr);

void print_now();

CLICK_ENDDECLS
#endif
