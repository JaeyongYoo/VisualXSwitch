rc :: RConn();
dp :: Datapath( HOST_ADDR	de:ad:be:ef:00:00,

	 PORT1_ADDR 00:15:17:8a:ef:f2,
	 PORT1_NAME eth3,

	 PORT2_ADDR 00:15:17:8a:ef:f3,
	 PORT2_NAME eth5,

	 PORT3_ADDR 00:00:5a:72:56:12,
	 PORT3_NAME eth1,

	 PORT4_ADDR 00:00:5a:72:56:11,
	 PORT4_NAME eth0,

	DPID 1 );

fh :: FromHost(DEVNAME tap0, DST 192.168.0.1/24, ETHER de:ad:be:ef:00:00 );

th :: ToHost(tap0);

fd0 :: FromDevice( eth3, SNIFFER false, PROMISC true, HEADROOM 34, CAPTURE LINUX );
td0 :: ToDevice( eth3, DEBUG 0 );

fd1 :: FromDevice( eth5, SNIFFER false, PROMISC true, HEADROOM 34, CAPTURE LINUX );
td1 ::ToDevice( eth5, DEBUG 0 );

fd2 :: FromDevice( eth1, SNIFFER false, PROMISC true, HEADROOM 34, CAPTURE LINUX );
td2 :: ToDevice( eth1, DEBUG 0 );

fd3 :: FromDevice( eth0, SNIFFER false, PROMISC true, HEADROOM 34, CAPTURE LINUX );
td3 :: ToDevice( eth0, DEBUG 0 );
/*
q3 :: Queue(100);
//cl :: Classifier( 26/c0a80b0bc0a80b0d 36/d903 , - );
//cl :: Classifier( 26/c0a80b0bc0a80b0d0000 36/d903 , - );

dxt :: DXTCompressor(PIXEL_BLOCK_BUFFER 8192, DO_COMPRESS 1);
rrs :: RoundRobinSched();

cl[0] -> dxt -> q3;
cl[1] -> q4;

q3 -> [0]rrs;
q4 -> BandwidthShaper(62500000) -> [1]rrs;

rrs -> td3;
*/

rc  -> [0]dp[0] -> rc;
fh  -> [1]dp[1] -> th;
fd0 -> [2]dp[2] -> Queue(3000)-> td0;
fd1 -> [3]dp[3] -> Queue(3000)-> td1;
fd2 -> [4]dp[4] -> Queue(3000)-> td2;
fd3 -> [5]dp[5] -> Queue(3000)-> td3; 


