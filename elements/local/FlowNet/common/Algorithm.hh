#ifndef __ALGORITHM_H__
#define __ALGORITHM_H__

#define FLOWNET_ALGORITHM_NAME_SIZE 64

/*
 * base class for algorithms 
 */
class Algorithm {
public:

	Algorithm() {};
	~Algorithm() {};
	inline bool isThisAlgorithm(const char*) const;
	inline const char* name() const;

protected:
	inline void set_name(const char *buf);

private:

	char _name[FLOWNET_ALGORITHM_NAME_SIZE];
};

inline void Algorithm::set_name(const char *buf)
{
	int len = strlen(buf);
	assert( len < FLOWNET_ALGORITHM_NAME_SIZE );
	
	strncpy( _name, buf, FLOWNET_ALGORITHM_NAME_SIZE );
}

inline const char* Algorithm::name() const
{
	return _name;
}

inline bool Algorithm::isThisAlgorithm(const char* sn) const
{
	return strncmp( _name, sn, FLOWNET_ALGORITHM_NAME_SIZE ) == 0;
}


#endif
