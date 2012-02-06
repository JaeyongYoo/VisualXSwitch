#ifndef __SCHED_OUTER_VIRTUALPRESSURE_H__
#define __SCHED_OUTER_VIRTUALPRESSURE_H__

#include <click/config.h>
#include <sys/ioctl.h>	/* for using ioctl */
#include <net/if.h>

#include "../../common/FlowCommon.hh"
#include "../../common/ScheduleOuter.hh"

#define	SIOCGATHSTATS		(SIOCDEVPRIVATE+0)

CLICK_DECLS

#define JYD_FUNCTION_CALL_FREQUENCY 1

/*
 * XXX: this should be always be synchronized to the
 * file of madwifi 0.9.4 "ath/if_athioctl.h" 
 */
struct ath_stats {
	u_int32_t ast_watchdog;		/* device reset by watchdog */
	u_int32_t ast_hardware;		/* fatal hardware error interrupts */
	u_int32_t ast_bmiss;		/* beacon miss interrupts */
	u_int32_t ast_rxorn;		/* rx overrun interrupts */
	u_int32_t ast_rxeol;		/* rx eol interrupts */
	u_int32_t ast_txurn;		/* tx underrun interrupts */
	u_int32_t ast_mib;		/* mib interrupts */
	u_int32_t ast_tx_packets;	/* packet sent on the interface */
	u_int32_t ast_tx_mgmt;		/* management frames transmitted */
	u_int32_t ast_tx_discard;	/* frames discarded prior to assoc */
	u_int32_t ast_tx_invalid;	/* frames discarded due to is device gone */
	u_int32_t ast_tx_qstop;		/* tx queue stopped because it's full */
	u_int32_t ast_tx_encap;		/* tx encapsulation failed */
	u_int32_t ast_tx_nonode;		/* tx failed due to of no node */
	u_int32_t ast_tx_nobuf;		/* tx failed due to of no tx buffer (data) */
	u_int32_t ast_tx_nobufmgt;	/* tx failed due to of no tx buffer (mgmt)*/
	u_int32_t ast_tx_xretries;	/* tx failed due to of too many retries */
	u_int32_t ast_tx_fifoerr;	/* tx failed due to of FIFO underrun */
	u_int32_t ast_tx_filtered;	/* tx failed due to xmit filtered */
	u_int32_t ast_tx_shortretry;	/* tx on-chip retries (short) */
	u_int32_t ast_tx_longretry;	/* tx on-chip retries (long) */
	u_int32_t ast_tx_badrate;	/* tx failed due to of bogus xmit rate */
	u_int32_t ast_tx_noack;		/* tx frames with no ack marked */
	u_int32_t ast_tx_rts;		/* tx frames with rts enabled */
	u_int32_t ast_tx_cts;		/* tx frames with cts enabled */
	u_int32_t ast_tx_shortpre;	/* tx frames with short preamble */
	u_int32_t ast_tx_altrate;	/* tx frames with alternate rate */
	u_int32_t ast_tx_protect;	/* tx frames with protection */
	u_int32_t ast_rx_orn;		/* rx failed due to of desc overrun */
	u_int32_t ast_rx_crcerr;		/* rx failed due to of bad CRC */
	u_int32_t ast_rx_fifoerr;	/* rx failed due to of FIFO overrun */
	u_int32_t ast_rx_badcrypt;	/* rx failed due to of decryption */
	u_int32_t ast_rx_badmic;		/* rx failed due to of MIC failure */
	u_int32_t ast_rx_phyerr;		/* rx PHY error summary count */
	u_int32_t ast_rx_phy[32];	/* rx PHY error per-code counts */
	u_int32_t ast_rx_tooshort;	/* rx discarded due to frame too short */
	u_int32_t ast_rx_toobig;		/* rx discarded due to frame too large */
	u_int32_t ast_rx_nobuf;		/* rx setup failed due to of no skbuff */
	u_int32_t ast_rx_packets;	/* packet recv on the interface */
	u_int32_t ast_rx_mgt;		/* management frames received */
	u_int32_t ast_rx_ctl;		/* control frames received */
	int8_t ast_tx_rssi;		/* tx rssi of last ack */
	int8_t ast_rx_rssi;		/* rx rssi from histogram */
	u_int32_t ast_be_xmit;		/* beacons transmitted */
	u_int32_t ast_be_nobuf;		/* no skbuff available for beacon */
	u_int32_t ast_per_cal;		/* periodic calibration calls */
	u_int32_t ast_per_calfail;	/* periodic calibration failed */
	u_int32_t ast_per_rfgain;	/* periodic calibration rfgain reset */
	u_int32_t ast_rate_calls;	/* rate control checks */
	u_int32_t ast_rate_raise;	/* rate control raised xmit rate */
	u_int32_t ast_rate_drop;		/* rate control dropped xmit rate */
	u_int32_t ast_ant_defswitch;	/* rx/default antenna switches */
	u_int32_t ast_ant_txswitch;	/* tx antenna switches */
	u_int32_t ast_ant_rx[8];		/* rx frames with antenna */
	u_int32_t ast_ant_tx[8];		/* tx frames with antenna */

#ifdef JYD_FUNCTION_CALL_FREQUENCY
        /* jyyoo add to debug function call frequencies */
        u_int32_t ast_jyd_ath_hardstart;
        u_int32_t ast_jyd_ath_hardstart_drop;
        u_int32_t ast_jyd_ath_hardstart_silent_drop;
        u_int32_t ast_jyd_ath_hardstart_raw;
        u_int32_t ast_jyd_ieee80211_hardstart;
        u_int32_t ast_jyd_ieee80211_hardstart_drop;
        u_int32_t ast_jyd_ieee80211_hardstart_silent_drop;

        u_int32_t ast_jyd_ath_priority_0;
        u_int32_t ast_jyd_ath_priority_1;
        u_int32_t ast_jyd_ath_priority_2;
        u_int32_t ast_jyd_ath_priority_3;
        u_int32_t ast_jyd_ath_priority_4;
        u_int32_t ast_jyd_ath_priority_5;
        u_int32_t ast_jyd_ath_priority_6;
        u_int32_t ast_jyd_ath_priority_7;
#endif

};



class PFSchedFW;

/* 
 * WBS stands for Weighted Backpressure Scheduling 
 */
class VcScheduleOuterVirtualpressure : public VcScheduleOuter {
public:
	VcScheduleOuterVirtualpressure();
	~VcScheduleOuterVirtualpressure();
public:

        void periodic_monitor( int *next_period );
        void act();
        int bind( VcSchedule *, PFSchedFW *);

private:
	VcSchedule *_inner_scheduler;
	PFSchedFW *_sched;

	/* for getting information from madwifi */
	int _socket;
	struct ifreq _ifr;
	struct ath_stats _cur_ath_stats;

	int _operation_counter;

	int _no_closs_interval_counter;

};


CLICK_ENDDECLS

#endif
