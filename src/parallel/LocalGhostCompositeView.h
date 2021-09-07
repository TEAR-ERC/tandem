#ifndef LOCALGHOSTCOMPOSITEVIEW_20210507_H
#define LOCALGHOSTCOMPOSITEVIEW_20210507_H

#include "interface/BlockVector.h"
#include "interface/BlockView.h"
#include "parallel/SparseBlockVector.h"
#include "tensor/Tensor.h"

#include <cstddef>
#include <utility>

namespace tndm {

class LocalGhostCompositeView : public BlockView {
public:
    LocalGhostCompositeView(BlockVector const& local, SparseBlockVector<double> const& ghost)
        : local_(&local), ghost_(&ghost) {
        handle_ = local.begin_access_readonly();
    }

    virtual ~LocalGhostCompositeView() {
        if (local_) {
            local_->end_access_readonly(handle_);
        }
    }

    // no copy
    LocalGhostCompositeView(LocalGhostCompositeView const&) = delete;
    LocalGhostCompositeView& operator=(LocalGhostCompositeView const&) = delete;

    // move
    LocalGhostCompositeView(LocalGhostCompositeView&& other) noexcept
        : local_(std::exchange(other.local_, nullptr)),
          ghost_(std::exchange(other.ghost_, nullptr)), handle_(std::move(other.handle_)) {}
    LocalGhostCompositeView& operator=(LocalGhostCompositeView&& other) {
        local_ = std::exchange(other.local_, nullptr);
        ghost_ = std::exchange(other.ghost_, nullptr);
        handle_ = std::move(other.handle_);
        return *this;
    }

    bool has_block(std::size_t idx) const {
        if (idx < handle_.shape(1)) {
            return true;
        } else {
            return ghost_->has_block(idx);
        }
    }
    Vector<const double> get_block(std::size_t idx) const {
        if (idx < handle_.shape(1)) {
            return handle_.subtensor(slice{}, idx);
        } else {
            return ghost_->get_block(idx);
        }
    }

private:
    BlockVector const* local_;
    SparseBlockVector<double> const* ghost_;
    Matrix<const double> handle_;
};

} // namespace tndm

#endif // LOCALGHOSTCOMPOSITEVIEW_20210507_H
