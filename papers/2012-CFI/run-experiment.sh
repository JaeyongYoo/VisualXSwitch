# written by jaeyong 
# desc: a script that runs an click-based openflow experiment

if ! [ `whoami` == "root" ]; then
        echo "should be run by root"
        exit -1
fi


__CLICK=/home/netcs/packages/icc-sdn/click-2.0-FlowNet-v0.6.1/userlevel/click
__SEC_CHAN=/home/netcs/packages/icc-sdn/openflow-1.0.0/secchan/ofprotocol

# desc:
# a script that automatically generates click script for given input params
# input param 1: tap mac addr
# input param 2: tap ip addr
# input param 3: dpid
# input param ~n: interfaces 
generate_click_script() {
	TAP_MAC_ADDR=$1;
	TAP_IP_ADDR=$2;
	DPID=$3;

	# parse the next variable arguments which supposed to be 
	# interface names
	for((i=4; i<=$#; i++))
	do
		eval "interfaces=\$interfaces\" \${$i}\""
	done


	echo "rc :: RConn();"
	echo "dp :: Datapath( HOST_ADDR	$TAP_MAC_ADDR,"
	cnt=1;
	for i in $interfaces
	do
		echo ""
		sudo ifconfig $i up 0.0.0.0
		macaddr=$(ifconfig $i | awk "/$i/ { print \$5}")
		echo -e "\t PORT${cnt}_ADDR $macaddr,"
		echo -e "\t PORT${cnt}_NAME $i,"
		cnt=$(($cnt+1))
	done
	echo ""
	echo -e "\tDPID $DPID );"

	echo ""
	echo "fh :: FromHost(DEVNAME tap0, DST $TAP_IP_ADDR/24, ETHER $TAP_MAC_ADDR );"
	echo ""
	echo "th :: ToHost(tap0);"

	cnt=0;
	for i in $interfaces
	do
		echo ""
		echo "fd$cnt :: FromDevice( $i, SNIFFER false, PROMISC true, HEADROOM 34, CAPTURE LINUX );"
		echo "td$cnt ::Print(\"$i\", LENGTH 150)-> ToDevice( $i, DEBUG 0 );"

		cnt=$(($cnt+1))
	done

	echo ""
	echo "rc  -> [0]dp[0] -> rc;"
	echo "fh  -> [1]dp[1] -> th;"

	cnt=0;
	offseted_cnt=2;
	for i in $interfaces
	do
		echo "fd$cnt -> [$offseted_cnt]dp[$offseted_cnt] -> Queue(100)-> td$cnt;"

		cnt=$(($cnt+1))
		offseted_cnt=$(($offseted_cnt+1))
	done
}

###############################################################################################################
# experiment body 
###############################################################################################################

# first kill ofprotocols and click
sudo killall -9 ofprotocol click &> /dev/null

# launch ofprotocol
$__SEC_CHAN unix:/var/run/dp0.sock tcp:210.125.84.74:6655 &> /tmp/ofprotocol-output &
#$__SEC_CHAN unix:/var/run/dp0.sock tcp:210.125.84.74:6607 &> /tmp/ofprotocol-output &

sleep 2

generate_click_script de:ad:be:ef:00:00 192.168.0.1 1  eth3 eth5 eth1 eth0 > /tmp/tmp.click
$__CLICK $1 2>&1 1> output | tee /tmp/tmp
#gdb $__CLICK



