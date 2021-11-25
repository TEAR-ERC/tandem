#ifndef TYPE_20211125_H
#define TYPE_20211125_H

namespace tndm {

enum class SeasMode {
    Unknown = 0,
    QuasiDynamic = 1,
    QuasiDynamicDiscreteGreen = 2,
    FullyDynamic = 3
};
enum class LocalOpType { Unknown = 0, Poisson = 1, Elasticity = 2 };

} // namespace tndm

#endif // TYPE_20211125_H
