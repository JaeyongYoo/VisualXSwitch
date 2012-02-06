#ifndef __STIP_h__
#define __STIP_h__

/*
 - Source Video Type
	(RAW)
	YUV444		0x0001
	YUV422		0x0002
	YUV420		0x0004
	RGB		0x0011
	RGBA		0x0012
	(Compressed)
	MPEG2-TS	0x0021
	MPEG2-PS	0x0022
	MPEG4		0x0041
	H.263		0x0042
	H.264		0x0044

 - Processed Video Type
	DXT1/wo alpha	0x0001
	DXT1/w alpha	0x0002
	DXT5		0x0004
*/

// - Source Video Type
#define YUV444 		0x0001
#define	YUV422		0x0002
#define	YUV420		0x0004
#define	RGB		0x0011
#define	RGBA		0x0012
#define	MPEG2TS		0x0021
#define	MPEG2PS		0x0022
#define	MPEG4		0x0041
#define H263		0x0042
#define	H264		0x0044

// - Processed Video Type
#define	DXT1		0x0001
#define	DXT1A		0x0002
#define	DXT5		0x0004

struct  stip_common_header {
	unsigned char version;
	unsigned char hdr_len;
};

struct stip_initiation_header {
	unsigned char version;
	unsigned char hdr_len;
	unsigned short reservered;
	unsigned short src_video;
	unsigned short proc_video;
	unsigned char src_px_width;
	unsigned char src_px_height;
	unsigned short src_Bpb;
	unsigned char proc_px_width;
	unsigned char proc_px_height;
	unsigned short proc_Bpb;
	unsigned short frame_width;
	unsigned short frame_height;
	unsigned int max_video_frame;
	unsigned short video_fps;
};

struct stip_transport_header {
	unsigned char version;
	unsigned char hdr_len;
	unsigned short reservered;
	unsigned int frame_idx;
	unsigned int pblock_idx;
	unsigned int pblock_count;
};


#endif // __STIP_h__
