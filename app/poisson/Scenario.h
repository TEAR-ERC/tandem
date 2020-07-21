#ifndef SCENARIO_20200627_H
#define SCENARIO_20200627_H

#include "Poisson.h"
#include "config.h"

#include "form/Error.h"
#include "geometry/Curvilinear.h"
#include "mesh/GenMesh.h"
#include "mesh/GlobalSimplexMesh.h"

namespace tndm {

class Scenario {
public:
    using transform_t = Curvilinear<DomainDimension>::transform_t;
    using functional_t = Poisson::functional_t;

    virtual ~Scenario() {}
    virtual transform_t transform() const = 0;
    virtual functional_t force() const = 0;
    virtual functional_t dirichlet() const = 0;
    virtual std::unique_ptr<SolutionInterface> reference() const = 0;
    virtual functional_t K() const = 0;

    virtual std::unique_ptr<GlobalSimplexMesh<DomainDimension>>
    getGlobalMesh(unsigned long n, MPI_Comm comm) const = 0;
};

class MyScenario : public Scenario {
public:
    using reference_t = std::function<std::array<double, 1>(Vector<double> const&)>;

    MyScenario(
        transform_t transform, functional_t force, functional_t dirichlet, reference_t reference,
        functional_t K = [](auto) { return 1.0; })
        : trans_(std::move(transform)), force_(std::move(force)), diri_(std::move(dirichlet)),
          ref_(std::move(reference)), K_(std::move(K)) {}
    Curvilinear<DomainDimension>::transform_t transform() const override { return trans_; }
    functional_t force() const override { return force_; }
    functional_t dirichlet() const override { return diri_; }
    std::unique_ptr<SolutionInterface> reference() const override {
        return std::make_unique<LambdaSolution<decltype(ref_)>>(ref_);
    }
    functional_t K() const override { return K_; }

    void setPointsAndBCs(std::array<std::vector<double>, DomainDimension> const& points,
                         std::array<std::vector<BC>, DomainDimension> const& BCs) {
        points_ = points;
        bcs_ = BCs;
    }

    std::unique_ptr<GlobalSimplexMesh<DomainDimension>>
    getGlobalMesh(unsigned long n, MPI_Comm comm) const override {
        auto meshGen = meshGenerator(n, comm);
        auto globalMesh = meshGen.uniformMesh();
        globalMesh->repartition();
        return globalMesh;
    }

private:
    GenMesh<DomainDimension> meshGenerator(unsigned long n, MPI_Comm comm) const {
        if (points_ && bcs_) {
            std::array<double, DomainDimension> h;
            h.fill(1.0 / n);
            return GenMesh(*points_, h, *bcs_, comm);
        }
        std::array<uint64_t, DomainDimension> size;
        size.fill(n);
        std::array<std::pair<BC, BC>, DomainDimension> BCs;
        BCs.fill(std::make_pair(BC::Dirichlet, BC::Dirichlet));
        return GenMesh(size, BCs, comm);
    }

    transform_t trans_;
    functional_t force_, diri_;
    reference_t ref_;
    functional_t K_;

    std::optional<std::array<std::vector<double>, DomainDimension>> points_ = std::nullopt;
    std::optional<std::array<std::vector<BC>, DomainDimension>> bcs_ = std::nullopt;
};

} // namespace tndm

#endif // SCENARIO_20200627_H
