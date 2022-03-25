#include "Counters.h"

#include <iostream>
#include <map>
#include <string>

Counters::Counters() : total(0) { Counts = new std::map<std::string, int>; }

Counters::~Counters()
{
    if (Counts) delete Counts;
}

void Counters::inc(const char* s)
{
    (*Counts)[std::string(s)]++;
    total++;

    if (total == 1000000) {
        print();
        clear();
    }
}

void Counters::print()
{
    for (auto kv : *Counts) std::cout << "Counter," << kv.first << "," << kv.second << '\n';
    std::cout << "Counter,Total," << total << '\n';
}

void Counters::clear()
{
    Counts->clear();
    total = 0;
}
