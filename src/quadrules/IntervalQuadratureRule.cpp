#include "IntervalQuadratureRule.h"

namespace tndm {

void IntervalQuadratureRule::changeInterval(double start, double stop) {
    double diff = 0.5 * (b - a);
    double avg = 0.5 * (a + b);
    a = start;
    b = stop;
    double newDiff = 0.5 * (b - a);
    double newAvg = 0.5 * (a + b);
    for (auto& p : points_) {
        p = newDiff * (p - avg) / diff + newAvg;
    }
    for (auto& w : weights_) {
        w *= newDiff / diff;
    }
}

} // namespace tndm
