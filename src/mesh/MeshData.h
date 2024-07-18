#ifndef MESHDATA_H
#define MESHDATA_H

#include "basis/NumberingConvention.h"
#include "form/BC.h"
#include "parallel/CommPattern.h"
#include "parallel/MPITraits.h"
#include "tensor/Managed.h"
#include "tensor/Tensor.h"
#include "util/Algorithm.h"

#include <mpi.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <memory>
#include <vector>

namespace tndm {

class MeshData {
public:
    virtual ~MeshData() {}
    virtual std::size_t size() const = 0;
    virtual std::unique_ptr<MeshData> redistributed(std::vector<std::size_t> const& lids,
                                                    AllToAllV const& a2a) const = 0;
    virtual void permute(std::vector<std::size_t> const& permutation) = 0;


    virtual std::tuple<std::unique_ptr<MeshData>, std::vector<std::size_t>> redistributedWithMapping (std::vector<std::size_t> const& lids,
                                            AllToAllV const& a2a) const =0 ;
/*
    std::vector<std::vector<std::size_t>> generateMappingMatrix
    (const std::vector<std::size_t>& originalIndices, 
    const std::vector<std::size_t>& newIndices) const {

        std::vector<std::vector<std::size_t>> mappingMatrix;

        for (std::size_t i = 0; i < newIndices.size(); ++i) {
            if (newIndices[i] != std::numeric_limits<std::size_t>::max()) {
                mappingMatrix.push_back({newIndices[i], i});
            }
        }

        return mappingMatrix;
    }
    */
};

template <std::size_t SpaceD> class VertexData : public MeshData {
public:
    using vertex_t = std::array<double, SpaceD>;
    
    VertexData(std::vector<vertex_t>&& vertices) : vertices(std::move(vertices)) {}
    virtual ~VertexData() {}

    std::size_t size() const override { return vertices.size(); }

    std::unique_ptr<MeshData> redistributed(std::vector<std::size_t> const& lids,
                                            AllToAllV const& a2a) const override {
        std::vector<vertex_t> requestedVertices;
        requestedVertices.reserve(lids.size());
        for (auto& lid : lids) {
            assert(lid < vertices.size());
            requestedVertices.emplace_back(vertices[lid]);
        }

        mpi_array_type<double> mpi_t(SpaceD);
        auto newVertices = a2a.exchange(requestedVertices, mpi_t.get());
        return std::make_unique<VertexData>(std::move(newVertices));
    }

    std::tuple<std::unique_ptr<MeshData>, std::vector<std::size_t>> redistributedWithMapping(std::vector<std::size_t> const& lids,
                                                       AllToAllV const& a2a) const override {
        // Call redistributed and return its result
        auto redistributedData =  redistributed(lids, a2a);

        std::vector<std::size_t> dummpymapping = {1, 2, 3};

        return std::make_tuple(std::move(redistributedData), std::move(dummpymapping));
    
    }

    void permute(std::vector<std::size_t> const& permutation) override {
        apply_permutation(vertices, permutation);
    }

    std::vector<vertex_t> const& getVertices() const { return vertices; }

private:
    std::vector<vertex_t> vertices;
    
};


class BoundaryData : public MeshData {
public:
    BoundaryData(std::vector<BC>&& BCs) : boundaryConditions(std::move(BCs)) {}
    virtual ~BoundaryData() {}

    std::size_t size() const override { return boundaryConditions.size(); }

    std::unique_ptr<MeshData> redistributed(std::vector<std::size_t> const& lids,
                                            AllToAllV const& a2a) const override {
        std::vector<BC> requestedBCs;
        requestedBCs.reserve(lids.size());
        for (auto& lid : lids) {
            if (lid == std::numeric_limits<std::size_t>::max()) {
                requestedBCs.emplace_back(BC::None);
            } else {
                requestedBCs.emplace_back(boundaryConditions[lid]);
            }
        }

        static_assert(sizeof(BC::None) == sizeof(int));
        auto newBCs = a2a.exchange(requestedBCs, mpi_type_t<int>());
        return std::make_unique<BoundaryData>(std::move(newBCs));
    }

    std::tuple<std::unique_ptr<MeshData>, std::vector<std::size_t>> redistributedWithMapping (std::vector<std::size_t> const& lids,
                                            AllToAllV const& a2a) const  {
        std::vector<BC> requestedBCs;
        std::vector<std::size_t> newOriginalIndices;
        std::vector<std::size_t> originalIndices(boundaryConditions.size());
        std::iota(originalIndices.begin(), originalIndices.end(), 0);

        requestedBCs.reserve(lids.size());
        newOriginalIndices.reserve(lids.size());

        for (auto& lid : lids) {
            if (lid == std::numeric_limits<std::size_t>::max()) {
                requestedBCs.emplace_back(BC::None);
                newOriginalIndices.emplace_back(std::numeric_limits<std::size_t>::max());
            } else {
                requestedBCs.emplace_back(boundaryConditions[lid]);
                newOriginalIndices.emplace_back(originalIndices[lid]);
            }
        }

        static_assert(sizeof(BC::None) == sizeof(int));
        auto newBCs = a2a.exchange(requestedBCs, mpi_type_t<int>());
        
        //auto mapping = generateMappingMatrix(originalIndices,newOriginalIndices);
        
        //auto newIndices = a2a.exchange(newOriginalIndices, mpi_type_t<std::size_t>());
       // return std::make_unique<BoundaryData>(std::move(newBCs));
        return {std::make_unique<BoundaryData>(std::move(newBCs)), std::move(newOriginalIndices)};
    }



    void permute(std::vector<std::size_t> const& permutation) override {
        apply_permutation(boundaryConditions, permutation);
    }

    std::vector<BC> const& getBoundaryConditions() const { return boundaryConditions; }

private:
    std::vector<BC> boundaryConditions;
};



class ElementData : public MeshData {
public:
    using nodes_t = Managed<Tensor<double, 3u>>;
    constexpr static std::size_t ElementMode = 2;

    ElementData(nodes_t&& nodes, NumberingConvention convention)
        : nodes_(std::move(nodes)), convention_(convention) {}
    virtual ~ElementData() {}

    std::size_t size() const override { return nodes_.size(); }

    std::unique_ptr<MeshData> redistributed(std::vector<std::size_t> const& lids,
                                            AllToAllV const& a2a) const override {
        auto shape = nodes_.shape();
        shape[ElementMode] = lids.size();
        auto requestedNodes = nodes_t(shape);
        std::size_t I = nodes_.shape(0), J = nodes_.shape(1), K = nodes_.shape(2);
        std::size_t k = 0;
        for (auto& lid : lids) {
            assert(lid < K);
            for (std::size_t j = 0; j < J; ++j) {
                for (std::size_t i = 0; i < I; ++i) {
                    requestedNodes(i, j, k) = nodes_(i, j, lid);
                }
            }
            ++k;
        }



        auto newNodes = a2a.exchange(requestedNodes);
        return std::make_unique<ElementData>(std::move(newNodes), convention_);
    }

    std::tuple<std::unique_ptr<MeshData>, std::vector<std::size_t>> redistributedWithMapping(
        std::vector<std::size_t> const& lids, AllToAllV const& a2a) const override {
        // Call redistributed and return its result
        auto redistributedData = redistributed(lids, a2a);

        // a dummy vector 
        std::vector<std::size_t> dummpymapping = {1, 2, 3};

        return std::make_tuple(std::move(redistributedData), std::move(dummpymapping));
    
    }


    void permute(std::vector<std::size_t> const& permutation) override {
        apply_permutation(nodes_, permutation, ElementMode);
    }

    nodes_t const& getNodes() const { return nodes_; }
    NumberingConvention getNumberingConvention() const { return convention_; }

private:
    nodes_t nodes_;
    NumberingConvention convention_;
};

} // namespace tndm

#endif // MESHDATA_H


/*  

old stuff that might become handy later on




class BoundaryDataWithMapping : public BoundaryData {
public:
    BoundaryDataWithMapping(std::vector<BC>&& BCs,  std::vector<std::size_t>&& newIndices)
        : BoundaryData(std::move(BCs)),  newIndices(std::move(newIndices)) {}

    std::unique_ptr<MeshData> redistributed(std::vector<std::size_t> const& lids,
                                            AllToAllV const& a2a) const override {
        std::vector<BC> requestedBCs;
        std::vector<std::size_t> newOriginalIndices;
    


        requestedBCs.reserve(lids.size());
        newOriginalIndices.reserve(lids.size());

        for (auto& lid : lids) {
            if (lid == std::numeric_limits<std::size_t>::max()) {
                requestedBCs.emplace_back(BC::None);
                newOriginalIndices.emplace_back(std::numeric_limits<std::size_t>::max());
            } else {
                requestedBCs.emplace_back(boundaryConditions[lid]);
                newOriginalIndices.emplace_back(originalIndices[lid]);
            }
        }

        static_assert(sizeof(BC::None) == sizeof(int));
        auto newIndices = a2a.exchange(newOriginalIndices, mpi_type_t<std::size_t>());
        auto newBCs = a2a.exchange(requestedBCs, mpi_type_t<int>());
        return std::make_unique<BoundaryDataWithMapping>(std::move(newBCs), std::move(newIndices));
    }

    std::vector<BC> const& getBoundaryConditions() const { return boundaryConditions; }
    std::vector<std::size_t> const& getOriginalIndices() const { return originalIndices; }
    std::vector<std::size_t> const& getNewIndices() const { return newIndices; }

    std::unordered_map<std::size_t, std::size_t> generateIndexMapping() const {
        std::unordered_map<std::size_t, std::size_t> indexMapping;
        for (std::size_t i = 0; i < newIndices.size(); ++i) {
            indexMapping[newIndices[i]] = i;
        }
        return indexMapping;
    }

    std::size_t getNewIndexPosition(std::size_t originalIndex) const {
        auto indexMapping = generateIndexMapping();
        auto it = indexMapping.find(originalIndex);
        if (it != indexMapping.end()) {
            return it->second;
        } else {
            throw std::runtime_error("Original index not found in new indices.");
        }
    }

    std::vector<std::size_t> getNewIndexPositions(const std::vector<std::size_t>& originalIndices) const {
        auto indexMapping = generateIndexMapping();
        std::vector<std::size_t> newIndexPositions;
        newIndexPositions.reserve(originalIndices.size());

        for (auto originalIndex : originalIndices) {
            auto it = indexMapping.find(originalIndex);
            if (it != indexMapping.end()) {
                newIndexPositions.push_back(it->second);
            } else {
                throw std::runtime_error("Original index " + std::to_string(originalIndex) + " not found in new indices.");
            }
        }
        return newIndexPositions;
    }

private:
    std::vector<std::size_t> originalIndices;
    std::vector<std::size_t> newIndices;
};






*/