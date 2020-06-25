#ifndef STOPWATCH_20200619_H
#define STOPWATCH_20200619_H

#include <time.h>

namespace tndm {

class Stopwatch {
public:
    void reset() { time_ = 0; }

    void start() { clock_gettime(CLOCK_MONOTONIC, &start_); }

    double split() {
        timespec end;
        clock_gettime(CLOCK_MONOTONIC, &end);
        return seconds(difftime(end));
    }

    double pause() {
        timespec end;
        clock_gettime(CLOCK_MONOTONIC, &end);
        time_ += difftime(end);
        return seconds(time_);
    }

    double stop() {
        double time_ = pause();
        reset();
        return time_;
    }

private:
    timespec start_;
    long long time_ = 0LL;

    long long difftime(struct timespec const& end) {
        return 1000000000LL * (end.tv_sec - start_.tv_sec) + end.tv_nsec - start_.tv_nsec;
    }

    double seconds(long long time) { return 1.0e-9 * time; }
};

} // namespace tndm

#endif // STOPWATCH_20200619_H
