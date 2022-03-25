#pragma once

#include <string>
#include <vector>

struct interval {
    float lower, upper;

    interval();
    interval(const float v);
    interval(const float v0, const float v1); // Takes the two extents of the range, not necessarily in order

    void set(const float v);                  // Set both extents to v
    void set(const float v1, const float v2); // Takes the two extents of the range, not necessarily in order
    void extend(const float v);               // Extend the interval to include this value
    void extend(const interval ivl);          // Extend the interval to include this interval
    void set_infinite();                      // Set the interval to -inf .. inf

    bool contains(const float v) const;      // True if the interval contains v
    bool contains(const interval ivl) const; // True if the interval fully contains ivl
    bool overlaps(const interval ivl) const; // True if the interval at least partially contains ivl
    bool empty() const;                      // True if the interval is empty
    bool finite() const;                     // True if both lower and upper are finite
    bool isnan() const;                      // True if either lower or upper is NaN

    float span() const; // Absolute distance between lower and upper

    float lower_finite() const; // Return lower, but with inf converted to max float
    float upper_finite() const; // Return upper, but with inf converted to max float

    static float min_float(); // Returns a finite number near +0
    static float infinity();  // Returns +inf
};

interval operator-(const interval& ivl);
interval operator+(const interval& ivl, const interval& ivr);
interval operator*(const interval& ivl, const interval& ivr);
interval intersect(const interval& ivl, const interval& ivr); // The intersection of the two intervals

std::string tostring(const interval& ival);
