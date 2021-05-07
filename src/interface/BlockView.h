#ifndef BLOCKVIEW_20210507_H
#define BLOCKVIEW_20210507_H

#include "tensor/Tensor.h"

#include <cstddef>

namespace tndm {

class BlockView {
public:
    virtual ~BlockView() {}

    virtual bool has_block(std::size_t idx) const = 0;
    virtual Vector<const double> get_block(std::size_t idx) const = 0;
};

} // namespace tndm

#endif // BLOCKVIEW_20210507_H
