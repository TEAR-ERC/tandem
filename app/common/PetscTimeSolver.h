#ifndef PETSCTS_20201001_H
#define PETSCTS_20201001_H

#include "common/PetscUtil.h"
#include "common/PetscVector.h"
#include "tandem/SeasConfig.h"
extern "C" {
#include "ts_util.h"
#include "vecnest_util.h"
}
#include <petscsystypes.h>
#include <petscts.h>
#include <petscvec.h>

#include <array>
#include <cassert>
#include <memory>
#include <tuple>

namespace tndm {

class PetscTimeSolverBase {
public:
    PetscTimeSolverBase(MPI_Comm comm, Config const& cfg);
    ~PetscTimeSolverBase();

    std::size_t get_step_number() const;
    std::size_t get_step_rejections() const;
    inline bool fsal() const { return fsal_; }
    void set_max_time_step(double dt);

protected:
    TS ts_ = nullptr;
    bool fsal_;
};

template <std::size_t NumStateVecs> class PetscTimeSolver : public PetscTimeSolverBase {
public:
    template <typename TimeOp>
    PetscTimeSolver(TimeOp& timeop, std::array<std::unique_ptr<PetscVector>, NumStateVecs> state,
                    Config const& cfg)
        : PetscTimeSolverBase(timeop.comm(), cfg), state_(std::move(state)),
          ts_checkpoint_load_directory(std::move(cfg.ts_checkpoint_load_directory)),
          comm(timeop.comm()) {

        Vec x[NumStateVecs];
        for (std::size_t n = 0; n < NumStateVecs; ++n) {
            x[n] = state_[n]->vec();
        }
        MPI_Comm comm;
        CHKERRTHROW(VecCreateNest(timeop.comm(), NumStateVecs, nullptr, x, &ts_state_));
        CHKERRTHROW(VecNestUpgradeOperations(ts_state_));

        std::apply([&timeop](auto&... x) { timeop.initial_condition((*x)...); }, state_);

        CHKERRTHROW(TSSetSolution(ts_, ts_state_));
        CHKERRTHROW(TSSetRHSFunction(ts_, nullptr, RHSFunction<TimeOp>, &timeop));
    }
    ~PetscTimeSolver() { VecDestroy(&ts_state_); }

    void solve(double upcoming_time) {
        CHKERRTHROW(TSSetUp(ts_));

        if (ts_checkpoint_load_directory.has_value()) {
            int rank;
            MPI_Comm_rank(comm, &rank);
            std::string sload;
            if (rank == 0) {
                sload = ts_checkpoint_load_directory.value();
                if (std::filesystem::is_regular_file(sload)) {
                    std::cout << "Retrieving the name of the last checkpoint from " << sload
                              << std::endl;
                    std::ifstream file(sload);
                    if (file.is_open()) {
                        if (std::getline(file, sload)) {
                            if (not std::filesystem::is_directory(sload)) {
                                throw std::runtime_error(
                                    "The first line of the file does not point to an existing "
                                    "directory.");
                            }
                        } else {
                            throw std::runtime_error("The file is empty.");
                        }
                        file.close();
                    } else {
                        throw std::runtime_error("Failed to open the file.");
                    }
                }
            } else {
                MPI_Bcast(&sload[0], sload.size(), MPI_CHAR, 0, MPI_COMM_WORLD);
            }
            const char* loadDirectory = sload.c_str();
            CHKERRTHROW(ts_checkpoint_restart(ts_, loadDirectory));
        }
        CHKERRTHROW(TSSetMaxTime(ts_, upcoming_time));
        CHKERRTHROW(TSSolve(ts_, ts_state_));
    }

    auto& state(std::size_t idx) {
        assert(idx < NumStateVecs);
        return *state_[idx];
    }
    auto const& state(std::size_t idx) const {
        assert(idx < NumStateVecs);
        return *state_[idx];
    }

    template <class Monitor> void set_monitor(Monitor& monitor) {
        CHKERRTHROW(TSMonitorSet(ts_, &MonitorFunction<Monitor>, &monitor, nullptr));
    }

private:
    template <typename TimeOp>
    static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec u, Vec F, void* ctx) {
        TimeOp* self = reinterpret_cast<TimeOp*>(ctx);

        std::array<Vec, 2 * NumStateVecs> x;
        for (std::size_t n = 0; n < NumStateVecs; ++n) {
            CHKERRTHROW(VecNestGetSubVec(u, n, &x[n]));
            CHKERRTHROW(VecNestGetSubVec(F, n, &x[NumStateVecs + n]));
        }
        auto x_view = std::apply(
            [](auto&... x) -> std::array<PetscVectorView, 2 * NumStateVecs> {
                return {PetscVectorView(x)...};
            },
            x);

        std::apply([&self, &t](auto&... xv) { self->rhs(t, xv...); }, x_view);
        return 0;
    }

    template <class Monitor>
    static PetscErrorCode MonitorFunction(TS ts, PetscInt steps, PetscReal time, Vec u, void* ctx) {
        Monitor* self = reinterpret_cast<Monitor*>(ctx);

        std::array<Vec, NumStateVecs> x;
        for (std::size_t n = 0; n < NumStateVecs; ++n) {
            CHKERRTHROW(VecNestGetSubVec(u, n, &x[n]));
        }
        auto x_view = std::apply(
            [](auto&... x) -> std::array<PetscVectorView, NumStateVecs> {
                return {PetscVectorView(x)...};
            },
            x);

        std::apply([&self, &time](auto&... xv) { self->monitor(time, xv...); }, x_view);
        return 0;
    }

    std::array<std::unique_ptr<PetscVector>, NumStateVecs> state_;
    Vec ts_state_ = nullptr;
    std::optional<std::string> ts_checkpoint_load_directory;
    MPI_Comm comm;
};

} // namespace tndm

#endif // PETSCTS_20201001_H
