#include <map>

class Counters
{
public:
	Counters(); 
	~Counters();

	void inc(const char* s);

	void print();

	void clear();

private:
	std::map<std::string, int>* Counts;
	int total;
};
