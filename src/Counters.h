#pragma once

#include <map>
#include <string>

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
