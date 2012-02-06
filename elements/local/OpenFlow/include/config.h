/* config.h.  Generated from config.h.in by configure.  */
/* config.h.in.  Generated from configure.ac by autoheader.  */

#define SIZE_MAX 65535
#define UINT16_MAX 65535
#define UINT32_MAX 65535*65535


#if __WORDSIZE == 64
#define UINT64_C(c)	c##UL
#else
#define UINT64_C(c)	c##ULL
#endif


// jyyoo debug
#define JD_DEBUG			0
#define JD_DEBUG_PARSE_CONTROL_PACKET	0


#if JD_DEBUG == 1

#include <pthread.h>

#define DEBUG_TRACE_FUNCTION_CALL click_chatter("jyyoo_debug [enter:%x] ===> %s\n", (uint32_t)pthread_self(), __PRETTY_FUNCTION__ );
#define DEBUG_TRACE_FUNCTION_END click_chatter("jyyoo_debug [leave:%x] ===> %s\n", (uint32_t)pthread_self(),  __PRETTY_FUNCTION__);
#define JD(n) click_chatter("\t%s %f\n", __FUNCTION__, n);
#define PRINT_OFP(buffer) {\
        click_chatter("=> Packet OutBuffer [size:%d]", buffer->size);  \
        uint32_t i;\
        for(i=0; i < (buffer->size < 80 ? buffer->size : 80); i++) { \
                if( i%16==0 ) click_chatter("\n\t");  \
                click_chatter("%x ", *( ((uint8_t*)buffer->data)+i) );  \
        } \
        click_chatter("\n");}
#define PRINT_OFP_HEADER(buffer) {\
	click_chatter("Buffer [base:%x l2:%x l3:%x l4:%x l7:%x alloc:%d]\n", buffer->base, buffer->l2, buffer->l3, buffer->l4, buffer->l7, buffer->allocated);\
	click_chatter("BUffer [data:%x size:%d]\n", buffer->data, buffer->size); \
	}

#else

#define DEBUG_TRACE_FUNCTION_CALL
#define DEBUG_TRACE_FUNCTION_END 
#define JD(n) 
#define PRINT_OFP(buffer) 
#define PRINT_OFP_HEADER(buffer) 


#endif


/* Official build number as a VERSION suffix string, e.g. "+build123", or ""
   if this is not an official build. */
#define BUILDNR ""

/* Define to 1 if net/if_packet.h is available. */
#define HAVE_IF_PACKET 1

/* Define to 1 if you have the <inttypes.h> header file. */
#define HAVE_INTTYPES_H 1

/* Define to 1 if you have the `socket' library (-lsocket). */
/* #undef HAVE_LIBSOCKET */

/* Define to 1 if you have __malloc_hook, __realloc_hook, and __free_hook in
   <malloc.h>. */
#define HAVE_MALLOC_HOOKS 1

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H 1

/* Define to 1 if Netlink protocol is available. */
#define HAVE_NETLINK 1

/* Define to 1 if OpenSSL is installed. */
/* #undef HAVE_OPENSSL */

/* Define to 1 if you have the <stdint.h> header file. */
#define HAVE_STDINT_H 1

/* Define to 1 if you have the <stdlib.h> header file. */
#define HAVE_STDLIB_H 1

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H 1

/* Define to 1 if you have the <string.h> header file. */
#define HAVE_STRING_H 1

/* Define to 1 if you have the `strlcpy' function. */
/* #undef HAVE_STRLCPY */

/* Define to 1 if you have the `strsignal' function. */
#define HAVE_STRSIGNAL 1

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H 1

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define to 1 if your C compiler doesn't accept -c and -o together. */
/* #undef NO_MINUS_C_MINUS_O */

/* Name of package */
#define PACKAGE "openflow"

/* Define to the address where bug reports for this package should be sent. */
#define PACKAGE_BUGREPORT "openflow-discuss@openflowswitch.org"

/* Define to the full name of this package. */
#define PACKAGE_NAME "openflow"

/* Define to the full name and version of this package. */
#define PACKAGE_STRING "openflow 1.0.0"

/* Define to the one symbol short name of this package. */
#define PACKAGE_TARNAME "openflow"

/* Define to the version of this package. */
#define PACKAGE_VERSION "1.0.0"

/* Define to 1 if you have the ANSI C header files. */
#define STDC_HEADERS 1

/* Enable extensions on AIX 3, Interix.  */
#ifndef _ALL_SOURCE
# define _ALL_SOURCE 1
#endif
/* Enable GNU extensions on systems that have them.  */
#ifndef _GNU_SOURCE
# define _GNU_SOURCE 1
#endif
/* Enable threading extensions on Solaris.  */
#ifndef _POSIX_PTHREAD_SEMANTICS
# define _POSIX_PTHREAD_SEMANTICS 1
#endif
/* Enable extensions on HP NonStop.  */
#ifndef _TANDEM_SOURCE
# define _TANDEM_SOURCE 1
#endif
/* Enable general extensions on Solaris.  */
#ifndef __EXTENSIONS__
# define __EXTENSIONS__ 1
#endif


/* Version number of package */
#define VERSION "1.0.0"

/* Number of bits in a file offset, on hosts where this is settable. */
#define _FILE_OFFSET_BITS 64

/* Define for large files, on AIX-style hosts. */
/* #undef _LARGE_FILES */

/* Define to 1 if on MINIX. */
/* #undef _MINIX */

/* Define to 2 if the system does not provide POSIX.1 features except with
   this defined. */
/* #undef _POSIX_1_SOURCE */

/* Define to 1 if you need to in order for `stat' and other things to work. */
/* #undef _POSIX_SOURCE */
