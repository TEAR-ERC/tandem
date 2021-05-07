#ifndef LOCALGHOSTCOMPOSITEVIEW_20210507_H
#define LOCALGHOSTCOMPOSITEVIEW_20210507_H

#include "interface/BlockView.h"
#include "parallel/SparseBlockVector.h"
#include "tensor/Tensor.h"

#include <cstddef>

namespace tndm {

class LocalGhostCompositeView : public BlockView {
public:
    LocalGhostCompositeView(Matrix<const double> const& local,
                            SparseBlockVector<double> const& ghost)
        : local_(local), ghost_(ghost) {}

    virtual ~LocalGhostCompositeView() {}

    bool has_block(std::size_t idx) const {
        if (idx < local_.shape(1)) {
            return true;
        } else {
            return ghost_.has_block(idx);
        }
    }
    Vector<const double> get_block(std::size_t idx) const {
        if (idx < local_.shape(1)) {
            return local_.subtensor(slice{}, idx);
        } else {
            return ghost_.get_block(idx);
        }
    }

private:
    Matrix<const double> const& local_;
    SparseBlockVector<double> const& ghost_;
};

} // namespace tndm

#endif // LOCALGHOSTCOMPOSITEVIEW_20210507_H
