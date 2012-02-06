// -*- c-basic-offset: 4 -*-
/*
 * 
 * Jae-Yong Yoo
 *
 * Copyright (c) 2010 Gwangju Institute of Science and Technology, Korea
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, subject to the conditions listed in the Click LICENSE
 * file. These conditions include: you must preserve this copyright
 * notice, and you cannot mention the copyright holders in advertising
 * related to the Software without their permission.  The Software is
 * provided WITHOUT ANY WARRANTY, EXPRESS OR IMPLIED. This notice is a
 * summary of the Click LICENSE file; the license in that file is
 * legally binding.
 */
#include <click/config.h>
#include <clicknet/udp.h>

#include "FlowMpeg2AdaptEncap.hh"
#include "../utils/FlowNetUtils.hh"
CLICK_DECLS

/*
 * jyyoo debugging : indenting stack depth 
 */
D_DEFINE_EXTERN;

static uint16_t network_header_size;
static uint16_t ethernet_header_size;

/* XXX: If input packet is ethernet or ip, change this functino */
static int extract_headers( Packet* p, click_ether** e, click_ip** i, click_udp** u, struct bpadapt_header ** n )
{

	*e = (click_ether*)p->data();
	*i = (click_ip*)(*e + 1);
	*u = (click_udp*)(*i + 1);
	if( n ) *n = (struct bpadapt_header*)(*u + 1);

	return 0;
}

/* 
 * error code strings
 * has to be corresponding to the error codes defined in "multiop_mpeg2_streamingproxy.hh"
 */
const char* const error_code_to_string [] = {
	"mpeg2 success: parse",
	"mpeg2 warning: pid not correct",
	"mpeg2 warning: no es header detected",
	"mpeg2 ERROR: sync not found",
	"mpeg2 ERROR: adaptation field too long",
	"mpeg2 ERROR: es too long",
	"mpeg2 ERROR: udp packet size not correct",
	"mpeg2 ERROR: out of memory" };


void FlowMpeg2AdaptEncap::clear()
{
        /* 
         * description: 
         * in order to use the original demux from shpark and yongjae,
         * port the original code and use wrapping 
	 * function to interface harmonically
	 * please DO NOT DEAL WITH THESE WITHOUT COMPLETE KNOWLEDGE OF MPEG2PARSER PART
         */
	state = DEMUX_SKIP; 
	state_bytes=0; 
	framenum=0;

	/* to set follow-up ts packet's frame type */
	frameindex = 0;
	prev_frame_type=0;
	curr_enque_frametype = 0;
	pkts_per_frame = 0;
	flush_frame = 0;

	last_receiving_frametype = -1;

	network_header_size = sizeof(click_ether) + sizeof(click_ip) + sizeof(click_udp);
	ethernet_header_size = sizeof(click_ether);

	Flow::clear();
}

const struct bpadapt_header* FlowMpeg2AdaptEncap::get_bpadapt_header_readonly(const Packet* p)
{
	D_START_FUNCTION;
	const struct bpadapt_header* b = (const struct bpadapt_header*)(p->data() + network_header_size);
	D_END_FUNCTION;
	return b;
}

void FlowMpeg2AdaptEncap::print_stat()
{
	char buf[1024];

        /* print the sent video frame index */
        stat.print_frame_index( NULL, buf );
        printf("\t%s\n", buf );

        stat.flush_frame_index();
}

void FlowMpeg2AdaptEncap::print_error_message(int errorcode)
{
	printf("Error! Code:%d [%s]\n", errorcode, error_code_to_string[errorcode] );
}



int FlowMpeg2AdaptEncap::enque( Packet* p, const Element* e )
{
	D_START_FUNCTION;
	int re=0;
	/* extract header */
	click_ether* ethdr;
	click_ip* iphdr;
	click_udp* udphdr;
	struct bpadapt_header* bphdr;

	extract_headers( p, &ethdr, &iphdr, &udphdr, &bphdr );

	if( bphdr->frametype == MPEG2_NON_VIDEO )
	{
		/* in this case, send immediately */
		e->output(0).push( p );

	} else {

		re = q.push( p );
		if( re == 0 ) { /* enqueue succeeded */

			if( flush_frame )
			{
				fprintf(stderr, "Error! FlowMpeg2AdaptEncap::enque() Structure failure\n");
			}

			/* if curr_enque_frametype == 0, it means start of the video */
			if( curr_enque_frametype == 0 )
				curr_enque_frametype = bphdr->frametype;

			/* new frame is enqueued, then flush the previous frame */
			if( curr_enque_frametype != bphdr->frametype ) {
				frameindex ++;
				flush_frame = 1;
				curr_enque_frametype = bphdr->frametype;

			} else {
				pkts_per_frame ++;
			}
		} else {
			fprintf(stderr, "FlowMpeg2AdaptEncap::enque() packet dropped.\n");
		}
	}
	D_END_FUNCTION;
	return re;
}

int FlowMpeg2AdaptEncap::deque_and_send(const Element* e, papmo* papmo)
{
	D_START_FUNCTION;
	unsigned int counter = 0;
	unsigned int frametype=0;

	if( flush_frame )
	{
		Packet* p;
		while( 1 )
		{ 	
			/* at the moment, just see if it is eligible packet */
			p = q.observe();

			if( p == NULL ) {
		
				fprintf(stderr, "FlowMpeg2AdaptEncap::deque_and_send() queue empty.\n");
				flush_frame = 0;
				pkts_per_frame=0;
				break;
			} else {
				click_ether* ethdr;
				click_ip* iphdr;
				click_udp* udphdr;
				struct bpadapt_header* bphdr;

				extract_headers( p, &ethdr, &iphdr, &udphdr, &bphdr );


				if( counter == 0 ) frametype = bphdr->frametype;

				if( bphdr->frametype == frametype ) { 
					/* we got the right frame */	
					/* then do the real pop */

					if( q.pop() != p ) {
						fprintf(stderr, "Error! Queue inconsistency\n");
					}

					/* setup magic header */
					bphdr->magicHeader = BPADAPT_MAGIC_HEADER; 

					/* fill up the additional bphdr info */
					bphdr->pkts_per_frame = pkts_per_frame;
					bphdr->pkts_index = counter;
					bphdr->frameindex = frameindex;
					counter ++;

					/* set stat for debug */
					stat.add_frame_index( frameindex );

					if( papmo ) papmo->do_monitor(
							COMPOSED_TRACE_TAG_MPEG ,
							COMPOSED_TRACE_POS_L4_OUT,
							p,
							this,
							NULL,
							NULL,
							NULL );


					/* and send the packet to next element */
					e->output(0).push( p );

				} else { /* we reach another frame, just stop sending out */

					/* at the moment, we are supposed to have 1-packet queued in the buffer */
					if( q.length() != 1 )
					{
						fprintf(stderr, 
						"FlowMpeg2AdaptEncap::deque_and_send() internal structure messed (queue size:%d)\n", q.length() );
					}

					if( pkts_per_frame != counter )
					{ /* this is probably internal structure messup */
						fprintf(stderr, "FlowMpeg2AdaptEncap::deque_and_send() internal structure messed (%d-%d)\n", pkts_per_frame, counter );
					}
					/* reset some important values to keep the frame statistics */
					/* why should it be 1 rather than 0? */
					/* this is a bit trick, we miss counting one, cause we have to know the new frame
					 * from enque, and we can not count the new frame as 1 for pkts_per_frame 
					 * so we count it here */
					pkts_per_frame = 1;
					/* waiting for next flush signal from enque */		
					flush_frame = 0;
					break;
				}
			}
		}

	} else {



		D_END_FUNCTION;
		return 0;
	}

	D_END_FUNCTION;
	return counter;
}

/* repacketize @p_in into @p_sep1, within @start and @end ts packets with @frametype1 */
int FlowMpeg2AdaptEncap::repacketization( Packet* p_in, WritablePacket** p_sep1, int start, int end, int frametype1 )
{
        D_START_FUNCTION;


	int ts_num;
	uint8_t* payload;
	uint8_t* packet_head;

	assert( p_in && p_sep1 );

	/* first derive information to allocate two packets */
	ts_num = end - start;



	packet_head = (uint8_t*) p_in->data();
	payload = packet_head + network_header_size;

	/* create two packets */
	*p_sep1 = Packet::make( 0, /* head room */
			NULL, /* data */
			network_header_size +
			sizeof(struct bpadapt_header) + ts_num * MPEG2_TS_SIZE, /* data size */
			0); /* tailroom */

	if( *p_sep1 == NULL  ) return MPEG2_ERROR_OUTOFMEMORY; 

	/* copy the data */
	memcpy( (*p_sep1)->data() + network_header_size + sizeof(struct bpadapt_header),
			p_in->data() + network_header_size + start * MPEG2_TS_SIZE,
			ts_num * MPEG2_TS_SIZE );


	/* copy the data */
	/* set frame type */
	struct bpadapt_header* bphdr1;
	bphdr1 = (struct bpadapt_header*)( (*p_sep1)->data() + network_header_size );

	bphdr1->frametype = frametype1;

	/* setup magic header */
	bphdr1->magicHeader = BPADAPT_MAGIC_HEADER; 


	/* copy the ip and udp header */
	memcpy( (*p_sep1)->data(), p_in->data(), network_header_size );

	/* checksum again */
	checksumIP( *p_sep1, ethernet_header_size );
	checksumUDP( *p_sep1, ethernet_header_size );

	/* also, we have to set desitnation annotation */
	click_ip* iphdr_sep1 = (click_ip*)((*p_sep1)->data() + ethernet_header_size);
	(*p_sep1)->set_dst_ip_anno( iphdr_sep1->ip_dst );


        D_END_FUNCTION;
	return 0;
}


/* TODO: This function should be decomposed... */
int FlowMpeg2AdaptEncap::separate_packet( Packet* p_in, WritablePacket** p_sep1, 
						WritablePacket** p_sep2, int separation_point, int frametype1, int frametype2 )
{
        D_START_FUNCTION;


	int first_packet_ts_num;
	int second_packet_ts_num;
	uint8_t* payload;
	uint8_t* packet_head;

	assert( p_in && p_sep1 && p_sep2 );

	/* first derive information to allocate two packets */
	first_packet_ts_num = separation_point;
	second_packet_ts_num = MPEG2_TS_PER_UDP - separation_point;
	packet_head = (uint8_t*) p_in->data();
	payload = packet_head + network_header_size;

	/* create two packets */
	*p_sep1 = Packet::make( 0, /* head room */
			NULL, /* data */
			network_header_size +
			sizeof(struct bpadapt_header) + first_packet_ts_num * MPEG2_TS_SIZE, /* data size */
			0); /* tailroom */
	*p_sep2 = Packet::make( 0, /* head room */
			NULL, /* data */
			network_header_size +
			sizeof(struct bpadapt_header) + second_packet_ts_num * MPEG2_TS_SIZE, /* data size */
			0); /* tailroom */

	if( *p_sep1 == NULL || *p_sep2 == NULL ) return MPEG2_ERROR_OUTOFMEMORY; 

	/* copy the data */
	memcpy( (*p_sep1)->data() + network_header_size + sizeof(struct bpadapt_header),
			p_in->data() + network_header_size,
			first_packet_ts_num * MPEG2_TS_SIZE );
	memcpy( (*p_sep2)->data() + network_header_size + sizeof(struct bpadapt_header),
			p_in->data() + network_header_size + first_packet_ts_num * MPEG2_TS_SIZE,
			second_packet_ts_num * MPEG2_TS_SIZE );

	/* set frame type */
	struct bpadapt_header* bphdr1;
	struct bpadapt_header* bphdr2;
	bphdr1 = (struct bpadapt_header*)( (*p_sep1)->data() + network_header_size );
	bphdr2 = (struct bpadapt_header*)( (*p_sep2)->data() + network_header_size );

	bphdr1->frametype = frametype1;
	bphdr2->frametype = frametype2;

	/* setup magic header */
	bphdr1->magicHeader = BPADAPT_MAGIC_HEADER; 

	/* setup magic header */
	bphdr2->magicHeader = BPADAPT_MAGIC_HEADER; 


	/* copy the ip and udp header */
	memcpy( (*p_sep1)->data(), p_in->data(), network_header_size );
	memcpy( (*p_sep2)->data(), p_in->data(), network_header_size );

	/* checksum again */
	checksumIP( *p_sep1, ethernet_header_size );
	checksumUDP( *p_sep1, ethernet_header_size );
	checksumIP( *p_sep2, ethernet_header_size );
	checksumUDP( *p_sep2, ethernet_header_size );

	/* also, we have to set desitnation annotation */
	click_ip* iphdr_sep1 = (click_ip*)((*p_sep1)->data() + ethernet_header_size);
	click_ip* iphdr_sep2 = (click_ip*)((*p_sep2)->data() + ethernet_header_size);
	(*p_sep1)->set_dst_ip_anno( iphdr_sep1->ip_dst );
	(*p_sep2)->set_dst_ip_anno( iphdr_sep2->ip_dst );


	/* kill original packet */
	p_in->kill();


        D_END_FUNCTION;
	return 0;
}

WritablePacket* FlowMpeg2AdaptEncap::encapsulate_bpadapt( Packet* p, int frametype )
{
        D_START_FUNCTION;


	WritablePacket* p_out;
	struct bpadapt_header* bphdr;
	int sizeof_nh = network_header_size;
	p_out = Packet::make( 0 /* head room */
				, NULL, /* data */
				p->length() + sizeof(struct bpadapt_header), /* data size */
				0); /* tailroom */

	/* targeting bphdr */
	bphdr = (struct bpadapt_header*)(p_out->data() + sizeof_nh);
	
	/* copy head */
	memcpy( p_out->data(), p->data(), sizeof_nh );
	
	/* set frametype into the packet */
	bphdr->frametype = frametype;

	/* setup magic header */
	bphdr->magicHeader = BPADAPT_MAGIC_HEADER; 

	/* copy the ts packets */
	memcpy( bphdr+1, p->data() + sizeof_nh, p->length() - sizeof_nh);

	/* reset the Ip checksum */
	checksumIP( p_out, ethernet_header_size );

	/* reset the UDP checksum: XXX do we have to?*/
	checksumUDP( p_out, ethernet_header_size );

	/* set annotation for ARP Request Element */
	click_ip* iphdr = (click_ip*)(p_out->data() + ethernet_header_size );
	p_out->set_dst_ip_anno( iphdr->ip_dst );

	p->kill(); /* kill original packet */

        D_END_FUNCTION;
	return p_out; 
}

/* 
 * description:
 * parse a UDP packet into multiple ts packets
 * when a UDP packet is constructed by multiple frame-types of ts packets, 
 * then, separate the UDP packet into two UDP packets
 */
int FlowMpeg2AdaptEncap::parse_packet( Packet* p_in, WritablePacket** p_out, int* p_out_len, int parsemode ) 
{
        D_START_FUNCTION;

	struct tspacket_table ts_table;
	int i;
	int re;
	int last_frametype = -1;
	int last_index=0;

	memset( &ts_table, 0, sizeof(ts_table) );

	/* sanity check */
	assert( p_in && p_out && p_out_len );

	/* parse the packet with FlowNetAdaptationHeader */
	if( parsemode == MPEG2_PARSEMODE_NORMAL )
	{
		/* parse the packet into ts packets */
		if( packet_to_ts( &ts_table, p_in, 0x00000000 ) ){
			fprintf( stderr, "FlowNet Error! mpeg2 parse failed at FlowMpeg2AdaptEncap::parse_packet\n");
			D_END_FUNCTION;
			return re;
		}

		*p_out_len = 0;

		/* UDP packet separation to arrange same frame types in a UDP packet */
		/* and also, provides frame type*/
		for( i = 0; i<MPEG2_TS_PER_UDP; i++ )
		{

			if( last_frametype != -1 )
			{
				if( last_frametype != ts_table.ts[i].frame_type )
				{

					/* separate here */
					repacketization( p_in, &p_out[ *p_out_len ], last_index, i, last_frametype );

					last_frametype = ts_table.ts[i].frame_type;
					*p_out_len = *p_out_len + 1;
					last_index = i;
				}
			}
			else {
				last_frametype = ts_table.ts[i].frame_type;
			}
		}

		/* last packet */
		repacketization( p_in, &p_out[ *p_out_len ], last_index, i, last_frametype );
		*p_out_len = *p_out_len + 1;
	}
	/* no parsing but only with FlowNetAdaptationHeader */
	else if( parsemode == MPEG2_PARSEMODE_NULL )
	{
		repacketization( p_in, &p_out[ *p_out_len ], last_index, MPEG2_TS_PER_UDP, last_frametype );
		*p_out_len = *p_out_len + 1;
	} else {
		fprintf(stderr, "Error! Unknown parsing mode: %d\n", parsemode );
	}

	p_in->kill();

	D_END_FUNCTION;
	return 0;
}


/*
 * print header messages of TS 
 *
 * 1. Expects ethernet-headered packet 
 *
 * 2. Strongly assumes little endian 
 */

struct mpeg2ts {
        uint8_t ts_sync;
	
	/* Note that bit-field is reverse order */
        uint8_t         ts_high_pid : 5,
                        ts_priority : 1,
                        ts_payload_unit_start : 1,
                        ts_error : 1;

	uint8_t		ts_low_pid;

	/* Note that bit-field is reverse order */
        uint8_t         ts_continuity_cnt : 4,
                        ts_adapt_field : 2,
                        ts_scramble_ctrl : 2;

} __attribute__((packed));


struct mpeg2ts_adapt {
	uint8_t a_field_length;

	/* Note that bit-field is reverse order */
	uint8_t		a_flag_extension : 1,
			a_flag_private_data : 1,
			a_flag_splicing_point : 1,
			a_flag_OPCR : 1,
			a_flag_PCR : 1,
			a_ind_es_priority : 1,
			a_ind_random_access : 1,
		 	a_ind_discontinuity : 1;

} __attribute__((packed));


struct mpeg2psi {
	uint8_t		psi_pointer_field;
} __attribute__((packed));

struct mpeg2psi_pat {
	uint8_t		pat_table_id;

	uint8_t 	pat_high_section_length : 4,
			pat_reserved : 2,
			pat_zero : 1,
			pat_ind_section_syntax : 1;
	uint8_t		pat_low_section_length;
	uint16_t	pat_ts_id;
	uint8_t		pat_ind_current_next : 1,
			pat_version : 5,
			pat_reserved2 : 2;
	uint8_t		pat_section;
	uint8_t		pat_last_section;
} __attribute__((packed));


struct mpeg2psi_pat_loop {
	uint16_t pat_program_number;
	uint8_t pat_high_PID : 5,
		pat_reserved : 3;
	uint8_t pat_low_PID;
} __attribute__((packed));

struct mpeg2psi_pmt {
	uint8_t		pmt_table_id;
	uint8_t 	pmt_high_section_length : 4,
			pmt_reserved : 2,
			pmt_zero : 1,
			pmt_ind_section_syntax : 1;
	uint8_t		pmt_low_section_length;
	uint16_t	pmt_program_number;		/* note the difference to @mpeg2psi_pat */
	uint8_t		pmt_ind_current_next : 1,
			pmt_version : 5,
			pmt_reserved2 : 2;
	uint8_t		pmt_section;
	uint8_t		pmt_last_section;
	uint8_t		pmt_high_PCR_PID : 5,
			pmt_reserved3 : 3;
	uint8_t		pmt_low_PCR_PID;
	uint8_t		pmt_high_program_info_length : 4,
			pmt_reserved4 : 4;
	uint8_t		pmt_low_program_info_length;

} __attribute__((packed));

struct mpeg2psi_pmt_loop { 
	uint8_t		pmt_stream_type;
	uint8_t		pmt_high_elementary_PID : 5,
			pmt_reserved : 3;
	uint8_t		pmt_low_elementary_PID;
	uint8_t		pmt_high_ES_info_length : 4,
			pmt_reserved2 : 4;
	uint8_t		pmt_low_ES_info_length;
	
} __attribute__((packed));


struct mpeg2pes {
	uint32_t	pes_start_prefix : 24,
			pes_stream_id : 8;

	uint16_t	pes_packet_length;

} __attribute__((packed));

struct mpeg2pes_optional {
	uint8_t		pes_original_or_copy	: 1,
			pes_copyright		: 1,
			pes_ind_data_align	: 1,
			pes_priority		: 1,
			pes_scramble_ctrl	: 2,
			pes_10			: 2;

	uint8_t		pes_flag_extension	: 1,
			pes_flag_CRC		: 1,
			pes_flag_additional_copy_info : 1,
			pes_flag_DSM_trick_mode : 1,
			pes_flag_ES_rate	: 1,
			pes_flag_ESCR_flag	: 1,
			pes_flag_PTS_DTS	: 2;

	uint8_t		pes_header_data_length;

} __attribute__((packed));


/*
 * list of table IDs: jyyoo thinks this ID 
 * is redundant since PID can also indicate 
 * the identity of the table)
 */
#define PSI_TID_PAT	0x00	// table id of program association section
#define PSI_TID_CAT	0x01	// table id of conditional access section
#define PSI_TID_PMT	0x02	// table id of TS program map section
// intermediate table ids: ITU-T Rec. H.222.0, or user private
#define PSI_TID_FBD	0xFF	// table id of forbidden


/*
 * MPEG2 start code prefix
 */
#define MPEG2_START_CODE_PREFIX	0x000001


/*
 * list of stream IDs used by MPEG2 (do not confuse to program ID)
 * codes 00 ~ B8 are video stream start codes and codes B9 ~ FF are stream ID
 */
#define MPEG2_SID_PICTURE			0x00
#define MPEG2_SID_SLICES			0x01 // ~ 0xAF
#define MPEG2_SID_SEQUENCE_HEADER		0xB3
#define MPEG2_SID_EXTENSION			0xB5
#define MPEG2_SID_GOP				0xB8

#define MPEG2_SID_PROGRAM_STREAM_MAP		0xBC 
#define MPEG2_SID_PRIVATE_STREAM_1		0xBD
#define MPEG2_SID_PADDING_STREAM		0xBE
#define MPEG2_SID_PRIVATE_STREAM_2		0xBF
#define MPEG2_SID_ECM_STREAM 			0xF0
#define MPEG2_SID_EMM_STREAM			0xF1
#define MPEG2_SID_PROGRAM_STREAM_DIRECTORY 	0xFF
#define MPEG2_SID_ITU_T_REC_H_222_0		0xF2
#define MPEG2_SID_ITU_T_REC_H_222_1_TYPE_E	0xF8


/*
 * constants for TS header 
 */
#define MPEG2_TS_SYNC 0x47
#define MPEG2_TS_ADAPTFIELD 0x2
#define MPEG2_TS_PAYLOADFIELD 0x1

/*
 * searches the stard-code-prefix starting from @start_point 
 * upto @size bytes and returns the next byte of the start-code-prefix
 */
static uint8_t* search_start_code( uint8_t* start_point, int size )
{
	uint8_t *p = start_point;
	for( int i = 0; i<size-2 /* subtracts 2 since search code prefix is 3 bytes */; i++ )
	{
		/* decomposed start-code-prefix 0x000001 */
		if(	p[0] == 0x00 &&
			p[1] == 0x00 &&
			p[2] == 0x01 ) {
			return &p[3];
		}
		p++;
	}
	return NULL;
}


/*
 * print TS and PES infos stored in Packet @p according to the following @print_option
 *
 * @print_option == 0b0000000000000001 : binary-print the TS data
 * 
 * @print_option == 0b0000000000000010 : print human-readable TS info
 * 
 * @print_option == 0b0000000000000100 : print time-serised TS-centered info
 *
 * @print_option == 0b0000000000001000 : print time-serised FrameIndex-centered info
 *
 */
int FlowMpeg2AdaptEncap::print_mpeg2ts_udp_packet( Packet* p, int print_option )
{
	click_ether *ether = (click_ether*) p->data();
	click_ip *ip = (click_ip*) (ether+1);
	click_udp *udp = (click_udp*) (ip+1);
	struct mpeg2ts *ts;
	uint8_t *raw = (uint8_t*)(udp+1);
	uint8_t *ptr;

	for( int q = 0; q<7; q++ )
	{
		ts = (struct mpeg2ts*)(raw + q * 188);

		if( print_option & 0x00000001 ) {
		/*
		 * print binary of TS data 
		 */
			int cnt=0;
			// print binary first
			printf("==Print Binary==\n");
			for( int l = 0; l<12; l++ ) {
				for( int j = 0; j<16; j++  ) {
					uint8_t v = *((uint8_t*)ts+j+l*16);
					int c=0;
					if( cnt < 188 ) {
						for( uint32_t i = (1<<7); i != 0x0; i >>= 1)
						{
							printf("%d", (v & i) ? 1 : 0);
							c++;
							if( c % 8 == 0 ) printf(" ");
						}
						cnt++;
					}
				}
				printf("\n");
			}
			printf("\n");
		}
	
		uint16_t ts_pid = (ts->ts_high_pid << 8) + ts->ts_low_pid;
		if( print_option & 0x00000002 ) {
			printf("\t==TS HEADER==\n");
			printf("\tts_sync= %x\n", ts->ts_sync);
			printf("\tts_error= %x\n", ts->ts_error);
			printf("\tts_payload_unit_start= %x\n", ts->ts_payload_unit_start);
			printf("\tts_priority= %x\n", ts->ts_priority);
			printf("\tts_TS_PID= %x\n", ts_pid);
			printf("\tts_scramble_ctrl= %x\n", ts->ts_scramble_ctrl);
			printf("\tts_adpat_field= %x\n", ts->ts_adapt_field);
			printf("\tts_continuity_cnt= %x\n", ts->ts_continuity_cnt);
		}

		if( print_option & 0x00000004 ) {
			printf("TS ");
		}

		if( print_option & 0x00000008 && ts_pid == 0x0044 && ts->ts_payload_unit_start ) {
			printf("\n");
		}


		ptr = (uint8_t*)(ts+1);

		/* adaptation field exist */
		if( ts->ts_adapt_field & MPEG2_TS_ADAPTFIELD ) {
			struct mpeg2ts_adapt *adapt = (struct mpeg2ts_adapt*)(ts+1);
			if( print_option & 0x00000002 ) {
				printf("\t\t==ADAPT HEADER==\n");
				printf("\t\tadapt_len: %d\n", adapt->a_field_length );
				printf("\t\tadapt_discon: %d\n", adapt->a_ind_discontinuity);
				printf("\t\tadapt_randomaccess: %d\n", 	adapt->a_ind_random_access);
				printf("\t\tadapt_priority:%d\n",	adapt->a_ind_es_priority);
				printf("\t\tadapt_PCR:%d\n", 	adapt->a_flag_PCR);
				printf("\t\tadapt_OPCR:%d\n", 	adapt->a_flag_OPCR);
				printf("\t\tadapt_splicing_point:%d\n",	adapt->a_flag_splicing_point);
				printf("\t\tadapt_pd:%d\n",	adapt->a_flag_private_data);
				printf("\t\tadapt_extension:%d\n",	adapt->a_flag_extension);
			}

			if( print_option & 0x0000000C ) {
				printf("ADAPT ");
			}

			/* a_field_length stores the length ahead of the first byte */
			ptr = ptr + 1 + adapt->a_field_length;
		}

		/* classify payload */
		if( ts_pid == 0x0000 ) /* the payload is Program Association Table */ {

			struct mpeg2psi *psi = (struct mpeg2psi*)(ptr);
			struct mpeg2psi_pat *pat = (struct mpeg2psi_pat*)(ptr + sizeof(struct mpeg2psi) + psi->psi_pointer_field);

			uint16_t section_length = (pat->pat_high_section_length << 8) + pat->pat_low_section_length;
			int residual_bytes = section_length - 5 /* offset to other headers  */ - sizeof(uint32_t) /* sizeof CRC */;
			int pat_loop_cnt = residual_bytes / sizeof(mpeg2psi_pat_loop);

			if( print_option & 0x00000002 ) {
				printf("\t\t==PAT HEADER\n");
				printf("\t\tpat_table_id=%x\n", pat->pat_table_id );
				printf("\t\tpat_section_indicator=%x\n", pat->pat_ind_section_syntax);
				printf("\t\tpat_section_length=%x\n", section_length );
				printf("\t\tpat_ts_id=%x\n", ntohs(pat->pat_ts_id));
				printf("\t\tpat_ind_current_next=%x\n", pat->pat_ind_current_next);
				printf("\t\tpat_version=%x\n", pat->pat_version);
				printf("\t\tpat_section=%x\n", pat->pat_section);
				printf("\t\tpat_last_section=%x\n", pat->pat_last_section);
				printf("\t\tpat loop count=%d (residual bytes=%d)\n", pat_loop_cnt, residual_bytes );
			}

			if( print_option & 0x0000000C ) {
				printf("PAT ");
			}


			struct mpeg2psi_pat_loop *loop = (struct mpeg2psi_pat_loop*)(pat+1);

			/* since we know we only have 1 PID */

			for( int i = 0; i<pat_loop_cnt; i++ ) {
				uint16_t pid2 = (loop->pat_high_PID << 8) + loop->pat_low_PID;

				if( print_option & 0x00000002 ) {

					printf("\t\t\t[loop index=%d\n", i);
					printf("\t\t\tprogram_number= %x\n", ntohs(loop->pat_program_number) );

					/* TODO: this @pid2 should be globally stored and be applied
					 * to the below @ts_pid comparison and get PMT info */
					printf("\t\t\tTS_PID= %x\n", pid2 );
					loop ++;
				}
			}


		} else if( ts_pid == 0x0042 ) {
			/* TODO: we should not use constant 0x0042 
			 * and use globally stored @pid2 info */
			struct mpeg2psi *psi = (struct mpeg2psi*)(ptr);
			struct mpeg2psi_pmt *pmt = (struct mpeg2psi_pmt*)(ptr + sizeof(struct mpeg2psi) + psi->psi_pointer_field);
			uint16_t section_length = (pmt->pmt_high_section_length << 8) + pmt->pmt_low_section_length;
			uint16_t program_info_length = (pmt->pmt_high_program_info_length << 8) + pmt->pmt_low_program_info_length;


			if( print_option & 0x00000002 ) {
				printf("\t\tpmt_table_id=%x\n", pmt->pmt_table_id );
				printf("\t\tpmt_section_indicator=%x\n", pmt->pmt_ind_section_syntax);
				printf("\t\tpmt_section_length=%x\n", section_length );
				printf("\t\tpmt_program_number=%x\n", ntohs(pmt->pmt_program_number));
				printf("\t\tpmt_ind_current_next=%x\n", pmt->pmt_ind_current_next);
				printf("\t\tpmt_version=%x\n", pmt->pmt_version);
				printf("\t\tpmt_section=%x\n", pmt->pmt_section);
				printf("\t\tpmt_last_section=%x\n", pmt->pmt_last_section);
				printf("\t\tpmt_program_info_length=%x\n", program_info_length);
			}

			if( print_option & 0x0000000C ) {
				printf("PMT ");
			}


			/* FIXME: assume there is no descriptor() */
			int pmt_loop_cnt = ntohs(pmt->pmt_program_number);
			
			struct mpeg2psi_pmt_loop *loop = (struct mpeg2psi_pmt_loop*)(pmt+1);
			for( int i = 0; i<pmt_loop_cnt; i++ ) {

				uint16_t elementary_PID;
				uint16_t ES_info_length;
				elementary_PID = (loop->pmt_high_elementary_PID << 8) + loop->pmt_low_elementary_PID;
				ES_info_length = (loop->pmt_high_ES_info_length << 8) + loop->pmt_low_ES_info_length;

				if( print_option & 0x00000002 ) {

					/* TODO: this @elementary_PID should be globally stored and be applied
					 * to the below @ts_pid comparison and get PMT info */
					printf("\t\t\tpmt_stream_type=%x\n", loop->pmt_stream_type );
					printf("\t\t\tpmt_elementary_PID=%x\n", elementary_PID );
					printf("\t\t\tES_info_length=%x\n", ES_info_length);
				}
				loop ++;

			}
		
		} else if( ts_pid == 0x0044 ) {
			/* TODO: we should not use constant 0x0042 
			 * and use globally stored @elementary_PID info */
			if( ts->ts_payload_unit_start ) {
				struct mpeg2pes *pes = (struct mpeg2pes*)(ptr);


				if( print_option & 0x00000002 ) {
					printf("\t\t==PES header==\n");
					printf("\t\tpes_start_prefix= %x\n", pes->pes_start_prefix);
					printf("\t\tpes_stream_id= %x\n", pes->pes_stream_id );
					printf("\t\tpes_length= %d\n", ntohs(pes->pes_packet_length) );
				}

				if( 
						pes->pes_stream_id !=  MPEG2_SID_PROGRAM_STREAM_MAP	&&
						pes->pes_stream_id !=  MPEG2_SID_PADDING_STREAM		&&
						pes->pes_stream_id !=  MPEG2_SID_PRIVATE_STREAM_2	&&
						pes->pes_stream_id !=  MPEG2_SID_ECM_STREAM		&&
						pes->pes_stream_id !=  MPEG2_SID_EMM_STREAM		&&
						pes->pes_stream_id !=  MPEG2_SID_ITU_T_REC_H_222_0	&& 
						pes->pes_stream_id !=  MPEG2_SID_PROGRAM_STREAM_DIRECTORY &&
						pes->pes_stream_id !=  MPEG2_SID_ITU_T_REC_H_222_1_TYPE_E ) {
					struct mpeg2pes_optional *opt = (struct mpeg2pes_optional*)(pes+1);
					if( print_option & 0x00000002 ) {
						printf("\t\t\tpes_option_10= %x\n", opt->pes_10);
					}
				} 

				if( print_option & 0x0000000C ) {
					printf("NEW_FRAME ");
				}

			}
			/* 
			 * ptr is the starting byte of PES 
			 * this searching is regardless of @ts_payload_unit_start
			 * since start-code can happen any point any where
			 */
			int residual_bytes = 188 - ( ((uint8_t*)ptr) - ((uint8_t*)ts) );
			uint8_t *next;

			while( (next = search_start_code( ptr, residual_bytes ) ))  {
				int associated_frame_type=0;
				if( *next == 0 /* it means that it is Picture */ ) {
					/* TODO: make this as structure */
					associated_frame_type = (*(next+2) & 0x1c)>>3;
				}
				if( print_option & 0x0000000C ) {
					printf("SID= %x ", *next );
					if( *next == 0 ) {
						printf("[%d] ", associated_frame_type );
					}
				}

				residual_bytes -= (next - ptr);
				ptr = next;
			}

		} else {
			printf("Error! Unrecognized PID is found: %x\n", ts_pid );
		}	

		if( print_option & 0x00000004 ) {
			printf("\n");
		}
	}
	return 0;
}



/*
 * parse a packet into ts packets
 *
 * @vervose == 0b0000000000000001 : binary-print the TS data
 * 
 * @vervose == 0b0000000000000010 : print human-readable TS info
 * 
 * @vervose == 0b0000000000000100 : print time-serised TS-centered info
 *
 * @vervose == 0b0000000000001000 : print time-serised FrameIndex-centered info
 *
 */
int FlowMpeg2AdaptEncap::packet_to_ts( struct tspacket_table *ts_table, Packet* p, int vervose )
{
	click_ether *ether = (click_ether*) p->data();
	click_ip *ip = (click_ip*) (ether+1);
	click_udp *udp = (click_udp*) (ip+1);
	struct mpeg2ts *ts;
	uint8_t *raw = (uint8_t*)(udp+1);
	uint8_t *ptr;

	for( int q = 0; q<7; q++ )
	{
		ts = (struct mpeg2ts*)(raw + q * 188);

		if( vervose & 0x00000001 ) {
		/*
		 * print binary of TS data 
		 */
			int cnt=0;
			// print binary first
			printf("==Print Binary==\n");
			for( int l = 0; l<12; l++ ) {
				for( int j = 0; j<16; j++  ) {
					uint8_t v = *((uint8_t*)ts+j+l*16);
					int c=0;
					if( cnt < 188 ) {
						for( uint32_t i = (1<<7); i != 0x0; i >>= 1)
						{
							printf("%d", (v & i) ? 1 : 0);
							c++;
							if( c % 8 == 0 ) printf(" ");
						}
						cnt++;
					}
				}
				printf("\n");
			}
			printf("\n");
		}
	
		uint16_t ts_pid = (ts->ts_high_pid << 8) + ts->ts_low_pid;
		if( vervose & 0x00000002 ) {
			printf("\t==TS HEADER==\n");
			printf("\tts_sync= %x\n", ts->ts_sync);
			printf("\tts_error= %x\n", ts->ts_error);
			printf("\tts_payload_unit_start= %x\n", ts->ts_payload_unit_start);
			printf("\tts_priority= %x\n", ts->ts_priority);
			printf("\tts_TS_PID= %x\n", ts_pid);
			printf("\tts_scramble_ctrl= %x\n", ts->ts_scramble_ctrl);
			printf("\tts_adpat_field= %x\n", ts->ts_adapt_field);
			printf("\tts_continuity_cnt= %x\n", ts->ts_continuity_cnt);
		}

		if( vervose & 0x00000004 ) {
			printf("TS ");
		}

		if( vervose & 0x00000008 && ts_pid == 0x0044 && ts->ts_payload_unit_start ) {
			printf("\n");
		}


		ptr = (uint8_t*)(ts+1);

		/* adaptation field exist */
		if( ts->ts_adapt_field & MPEG2_TS_ADAPTFIELD ) {
			struct mpeg2ts_adapt *adapt = (struct mpeg2ts_adapt*)(ts+1);
			if( vervose & 0x00000002 ) {
				printf("\t\t==ADAPT HEADER==\n");
				printf("\t\tadapt_len: %d\n", adapt->a_field_length );
				printf("\t\tadapt_discon: %d\n", adapt->a_ind_discontinuity);
				printf("\t\tadapt_randomaccess: %d\n", 	adapt->a_ind_random_access);
				printf("\t\tadapt_priority:%d\n",	adapt->a_ind_es_priority);
				printf("\t\tadapt_PCR:%d\n", 	adapt->a_flag_PCR);
				printf("\t\tadapt_OPCR:%d\n", 	adapt->a_flag_OPCR);
				printf("\t\tadapt_splicing_point:%d\n",	adapt->a_flag_splicing_point);
				printf("\t\tadapt_pd:%d\n",	adapt->a_flag_private_data);
				printf("\t\tadapt_extension:%d\n",	adapt->a_flag_extension);
			}

			if( vervose & 0x0000000C ) {
				printf("ADAPT ");
			}

			/* a_field_length stores the length ahead of the first byte */
			ptr = ptr + 1 + adapt->a_field_length;
		}

		/* classify payload */
		if( ts_pid == 0x0000 ) /* the payload is Program Association Table */ {

			struct mpeg2psi *psi = (struct mpeg2psi*)(ptr);
			struct mpeg2psi_pat *pat = (struct mpeg2psi_pat*)(ptr + sizeof(struct mpeg2psi) + psi->psi_pointer_field);

			uint16_t section_length = (pat->pat_high_section_length << 8) + pat->pat_low_section_length;
			int residual_bytes = section_length - 5 /* offset to other headers  */ - sizeof(uint32_t) /* sizeof CRC */;
			int pat_loop_cnt = residual_bytes / sizeof(mpeg2psi_pat_loop);

			if( vervose & 0x00000002 ) {
				printf("\t\t==PAT HEADER\n");
				printf("\t\tpat_table_id=%x\n", pat->pat_table_id );
				printf("\t\tpat_section_indicator=%x\n", pat->pat_ind_section_syntax);
				printf("\t\tpat_section_length=%x\n", section_length );
				printf("\t\tpat_ts_id=%x\n", ntohs(pat->pat_ts_id));
				printf("\t\tpat_ind_current_next=%x\n", pat->pat_ind_current_next);
				printf("\t\tpat_version=%x\n", pat->pat_version);
				printf("\t\tpat_section=%x\n", pat->pat_section);
				printf("\t\tpat_last_section=%x\n", pat->pat_last_section);
				printf("\t\tpat loop count=%d (residual bytes=%d)\n", pat_loop_cnt, residual_bytes );
			}

			if( vervose & 0x0000000C ) {
				printf("PAT ");
			}


			struct mpeg2psi_pat_loop *loop = (struct mpeg2psi_pat_loop*)(pat+1);

			/* since we know we only have 1 PID */

			for( int i = 0; i<pat_loop_cnt; i++ ) {
				uint16_t pid2 = (loop->pat_high_PID << 8) + loop->pat_low_PID;

				if( vervose & 0x00000002 ) {

					printf("\t\t\t[loop index=%d\n", i);
					printf("\t\t\tprogram_number= %x\n", ntohs(loop->pat_program_number) );

					/* TODO: this @pid2 should be globally stored and be applied
					 * to the below @ts_pid comparison and get PMT info */
					printf("\t\t\tTS_PID= %x\n", pid2 );
					loop ++;
				}
			}
		
			/* XXX: assuming that PAT TS-packet belongs to the frametype of the past frames. */

		} else if( ts_pid == 0x0042 ) {
			/* TODO: we should not use constant 0x0042 
			 * and use globally stored @pid2 info */
			struct mpeg2psi *psi = (struct mpeg2psi*)(ptr);
			struct mpeg2psi_pmt *pmt = (struct mpeg2psi_pmt*)(ptr + sizeof(struct mpeg2psi) + psi->psi_pointer_field);
			uint16_t section_length = (pmt->pmt_high_section_length << 8) + pmt->pmt_low_section_length;
			uint16_t program_info_length = (pmt->pmt_high_program_info_length << 8) + pmt->pmt_low_program_info_length;


			if( vervose & 0x00000002 ) {
				printf("\t\tpmt_table_id=%x\n", pmt->pmt_table_id );
				printf("\t\tpmt_section_indicator=%x\n", pmt->pmt_ind_section_syntax);
				printf("\t\tpmt_section_length=%x\n", section_length );
				printf("\t\tpmt_program_number=%x\n", ntohs(pmt->pmt_program_number));
				printf("\t\tpmt_ind_current_next=%x\n", pmt->pmt_ind_current_next);
				printf("\t\tpmt_version=%x\n", pmt->pmt_version);
				printf("\t\tpmt_section=%x\n", pmt->pmt_section);
				printf("\t\tpmt_last_section=%x\n", pmt->pmt_last_section);
				printf("\t\tpmt_program_info_length=%x\n", program_info_length);
			}

			if( vervose & 0x0000000C ) {
				printf("PMT ");
			}


			/* FIXME: assume there is no descriptor() */
			int pmt_loop_cnt = ntohs(pmt->pmt_program_number);
			
			struct mpeg2psi_pmt_loop *loop = (struct mpeg2psi_pmt_loop*)(pmt+1);
			for( int i = 0; i<pmt_loop_cnt; i++ ) {

				uint16_t elementary_PID;
				uint16_t ES_info_length;
				elementary_PID = (loop->pmt_high_elementary_PID << 8) + loop->pmt_low_elementary_PID;
				ES_info_length = (loop->pmt_high_ES_info_length << 8) + loop->pmt_low_ES_info_length;

				if( vervose & 0x00000002 ) {

					/* TODO: this @elementary_PID should be globally stored and be applied
					 * to the below @ts_pid comparison and get PMT info */
					printf("\t\t\tpmt_stream_type=%x\n", loop->pmt_stream_type );
					printf("\t\t\tpmt_elementary_PID=%x\n", elementary_PID );
					printf("\t\t\tES_info_length=%x\n", ES_info_length);
				}
				loop ++;

			}

			/* XXX: assuming that PAT TS-packet belongs to the frametype of the past frames. */

		} else if( ts_pid == 0x0044 ) {
			/* TODO: we should not use constant 0x0042 
			 * and use globally stored @elementary_PID info */
			if( ts->ts_payload_unit_start ) {
				struct mpeg2pes *pes = (struct mpeg2pes*)(ptr);


				if( vervose & 0x00000002 ) {
					printf("\t\t==PES header==\n");
					printf("\t\tpes_start_prefix= %x\n", pes->pes_start_prefix);
					printf("\t\tpes_stream_id= %x\n", pes->pes_stream_id );
					printf("\t\tpes_length= %d\n", ntohs(pes->pes_packet_length) );
				}

				if( 
						pes->pes_stream_id !=  MPEG2_SID_PROGRAM_STREAM_MAP	&&
						pes->pes_stream_id !=  MPEG2_SID_PADDING_STREAM		&&
						pes->pes_stream_id !=  MPEG2_SID_PRIVATE_STREAM_2	&&
						pes->pes_stream_id !=  MPEG2_SID_ECM_STREAM		&&
						pes->pes_stream_id !=  MPEG2_SID_EMM_STREAM		&&
						pes->pes_stream_id !=  MPEG2_SID_ITU_T_REC_H_222_0	&& 
						pes->pes_stream_id !=  MPEG2_SID_PROGRAM_STREAM_DIRECTORY &&
						pes->pes_stream_id !=  MPEG2_SID_ITU_T_REC_H_222_1_TYPE_E ) {
					struct mpeg2pes_optional *opt = (struct mpeg2pes_optional*)(pes+1);
					if( vervose & 0x00000002 ) {
						printf("\t\t\tpes_option_10= %x\n", opt->pes_10);
					}
				} 

				if( vervose & 0x0000000C ) {
					printf("NEW_FRAME ");
				}

			}
			/* 
			 * ptr is the starting byte of PES 
			 * this searching is regardless of @ts_payload_unit_start
			 * since start-code can happen any point any where
			 */
			int residual_bytes = 188 - ( ((uint8_t*)ptr) - ((uint8_t*)ts) );
			uint8_t *next;

			while( (next = search_start_code( ptr, residual_bytes ) ))  {
				int associated_frame_type=0;
				if( *next == 0 /* it means that it is Picture */ ) {
					/* TODO: make this as structure */
					associated_frame_type = (*(next+2) & 0x1c)>>3;
					last_receiving_frametype = associated_frame_type;
				}


				if( vervose & 0x0000000C ) {
					printf("SID= %x ", *next );
					if( *next == 0 ) {
						printf("[%d] ", associated_frame_type );
					}
				}

				residual_bytes -= (next - ptr);
				ptr = next;
			}

		} else {
			printf("Error! Unrecognized PID is found: %x\n", ts_pid );
		}	

		if( vervose & 0x00000004 ) {
			printf("\n");
		}
		ts_table->ts[q].frame_type = last_receiving_frametype;

	}
	return 0;

}

/* statistics related functions for FlowMpeg2AdaptEncapStat */
void FlowMpeg2AdaptEncapStat::add_frame_type( int ft )
{
	if( total_frame_types >= STREAMINGPROXY_MAX_FRAMETYPE_BUFFER )
	{
		return;
	}
	received_frame_types[total_frame_types] = ft;
	total_frame_types ++;
}

void FlowMpeg2AdaptEncapStat::print_frame_type( FILE* fp )
{
	fprintf( fp, "FRAME_TYPE=> ");
	for( int i = 0; i<total_frame_types; i++ )
		fprintf( fp, "[%d] ", received_frame_types[i] );
	fprintf( fp, "\n");
}

void FlowMpeg2AdaptEncapStat::add_frame_index( int fi )
{
	D_START_FUNCTION;
	/* if the currently entering is the same as the previous, it is probably the same frame */
	if( total_frame_index != 0 && received_frame_index[total_frame_index-1] == fi ) 
	{
		D_END_FUNCTION;
		return;
	}

	if( total_frame_index >= STREAMINGSTUB_MAX_FRAMEINDEX_BUFFER ) 
	{
		D_END_FUNCTION;
		return;
	}
	
	/* if not, store it */
        received_frame_index[total_frame_index] = fi;
        total_frame_index ++;
	D_END_FUNCTION;
}

void FlowMpeg2AdaptEncapStat::flush_frame_index()
{
        total_frame_index = 0;
}

void FlowMpeg2AdaptEncapStat::print_frame_index( FILE* fp, char* buf )
{
        buf[0]=NULL;
        int old=-1;
        int i;
        for( i = 0; i<total_frame_index; i++ )
        {
                if( old == -1 ) {
                        old = received_frame_index[i];
                } else {
                        if( received_frame_index[i] - old != 1 )
                        {
                                if( fp )
                                        fprintf(fp, "*" );
                                if( buf )
                                        sprintf(buf+strlen(buf), "*" );
                        }
                        old = received_frame_index[i];
                }
        }
        if( fp ) fprintf(fp, " ");
        if( buf ) sprintf( buf+strlen(buf), " " );

        for( i = 0; i<total_frame_index; i++ )
        {
                if( fp )
                {
                        fprintf(fp, "%d ", received_frame_index[i] );
                }
                if( buf )
                {
                        sprintf(buf+strlen(buf), "%d ", received_frame_index[i] );
                }
        }
}


void tspacket_table::dump_table(FILE* fp)
{
	/* sanity check */
	assert(fp);
	fprintf(fp, "dumping tspacket table...\n");
	for( int i = 0; i<MPEG2_TS_PER_UDP; i++ )
	{
	}
}




CLICK_ENDDECLS
ELEMENT_PROVIDES(FlowMpeg2AdaptEncap)
