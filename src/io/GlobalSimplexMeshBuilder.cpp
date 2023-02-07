#include "GlobalSimplexMeshBuilder.h"

#include "basis/Equidistant.h"
#include "basis/NumberingConvention.h"
#include "basis/Util.h"
#include "mesh/MeshData.h"
#include "parallel/MPITraits.h"
#include "util/Math.h"
#include "util/Permutation.h"

#include <algorithm>
#include <cassert>
#include <exception>
#include <limits>
#include <sstream>
#include <type_traits>

namespace tndm {

template <std::size_t D>
void GlobalSimplexMeshBuilder<D>::preparePermutationTable(std::size_t numNodes) {
    auto N = num_nodes_to_degree<D>(numNodes);
    assert(N);

    node_permutations_ = Managed<Matrix<unsigned>>(numNodes, factorial(NumVerts));

    auto nodes = EquidistantNodesFactory<D>(NumberingConvention::GMSH)(*N);
    auto p = std::array<unsigned, NumVerts>{};
    std::iota(p.begin(), p.end(), 0);
    do {
        auto np = Lehmer_to_integer(permutation_to_Lehmer(p));
        auto P = node_permutation(nodes, p);
        assert(P);
        for (std::size_t k = 0; k < P->size(); ++k) {
            node_permutations_(k, np) = (*P)[k];
        }
    } while (std::next_permutation(p.begin(), p.end()));
}

template <std::size_t D>
void GlobalSimplexMeshBuilder<D>::addElement(long type, int ptag, int etag, long* node,
                                             std::size_t numNodes) {
    if (is_gmsh_simplex<D>(type)) {
        if (type_ == 0) {
            type_ = type;
        }
        if (type_ != type) {
            std::stringstream s;
            s << "Element of type " << type
              << " in GMSH mesh conflicts with previous element of type " << type_
              << ". (Mixed meshes are not supported.)";
            throw std::runtime_error(s.str());
        }
        assert(numNodes >= NumVerts);
        std::array<uint64_t, NumVerts> elem;
        std::copy(node, node + NumVerts, elem.begin());
        elements.emplace_back(Simplex<D>(elem));
		element_ptags.emplace_back(ptag);
		element_etags.emplace_back(etag);
        if (numNodes > NumVerts) {
            if (high_order_nodes.size() == 0) {
                preparePermutationTable(numNodes);
                high_order_nodes = Managed<Matrix<long>>(numNodes - NumVerts, elements.capacity());
            }
            auto p = std::array<unsigned, NumVerts>{};
            std::iota(p.begin(), p.end(), 0);
            std::sort(p.begin(), p.end(),
                      [&elem](unsigned a, unsigned b) { return elem[a] < elem[b]; });
            auto np = Lehmer_to_integer(permutation_to_Lehmer(p));
            std::size_t elNo = elements.size() - 1u;
            for (std::size_t i = NumVerts; i < numNodes; ++i) {
                auto p_i = node_permutations_(i, np);
                high_order_nodes(i - NumVerts, elNo) = node[p_i];
            }
        }
    } else if (is_gmsh_simplex<D - 1u>(type)) {
        assert(numNodes >= D);
        std::array<uint64_t, D> elem;
        std::copy(node, node + D, elem.begin());
        facets.emplace_back(Simplex<D - 1u>(elem));
		facet_ptags.emplace_back(ptag);
		facet_etags.emplace_back(etag);
		// BC bc = BC::None;
		//         switch (tag) {
		//         case static_cast<long>(BC::None):
		//             bc = BC::None;
		//             break;
		//         case static_cast<long>(BC::Dirichlet):
		//             bc = BC::Dirichlet;
		//             break;
		//         case static_cast<long>(BC::Fault):
		//             bc = BC::Fault;
		//             break;
		//         case static_cast<long>(BC::Natural):
		//             bc = BC::Natural;
		//             break;
		//         default:
		//             ++unknownBC;
		//             break;
		//         }
		//         bcs.push_back(bc);
    } else {
        if (is_lower_dimensional_gmsh_simplex_v<D - 1u>(type)) {
            ++ignoredElems;
        } else {
            std::stringstream s;
            s << "GMSH element of type " << type << " is not a 0,...," << D
              << "-simplex. Does the domain dimension match the dimension of the mesh?";
            throw std::runtime_error(s.str());
        }
    }
}

template <std::size_t D>
std::unique_ptr<GlobalSimplexMesh<D>> GlobalSimplexMeshBuilder<D>::create(MPI_Comm comm) {
    auto check_mesh = std::array<std::size_t, 3>{elements.size(), facets.size()};
    MPI_Allreduce(MPI_IN_PLACE, &check_mesh, 2, mpi_type_t<std::size_t>(), MPI_SUM, comm);
    if (check_mesh[0] == 0) {
        std::stringstream s;
        s << "The mesh does not contain any " << D
          << "-simplex. Does the domain dimension match the dimension of the mesh?";
        throw std::runtime_error(s.str());
    }
    if (check_mesh[1] == 0) {
        std::stringstream s;
        s << "The mesh does not contain any " << (D-1)
          << "-simplex. Does the domain dimension match the dimension of the mesh?";
        throw std::runtime_error(s.str());
    }

    std::unique_ptr<GlobalSimplexMesh<D>> mesh;
    auto high_order_shape = high_order_nodes.shape();
    MPI_Allreduce(MPI_IN_PLACE, &high_order_shape, 2,
                  mpi_type_t<std::decay_t<decltype(high_order_shape[0])>>(), MPI_MAX, comm);
    if (high_order_shape[1] > 0) {
        std::size_t num_nodes = high_order_shape[0];
        std::size_t num_elements = elements.size();
        auto high_order_verts = Managed<Tensor<double, 3u>>(D, num_nodes, num_elements);
        for (std::size_t k = 0; k < num_elements; ++k) {
            for (std::size_t j = 0; j < num_nodes; ++j) {
                auto const& verts = vertices[high_order_nodes(j, k)];
                for (std::size_t i = 0; i < D; ++i) {
                    high_order_verts(i, j, k) = verts[i];
                }
            }
        }

        // We need to drop all the high order nodes for GlobalSimplexMesh
        constexpr uint64_t Invalid = std::numeric_limits<uint64_t>::max();
        auto map = std::vector<uint64_t>(vertices.size(), Invalid);
        uint64_t new_id = 0;
        for (auto& elem : elements) {
            for (auto& e : elem) {
                if (map[e] == Invalid) {
                    map[e] = new_id++;
                }
                e = map[e];
            }
        }
        for (auto& fct : facets) {
            for (auto& f : fct) {
                f = map[f];
                assert(f != Invalid);
            }
        }

        auto new_vertices = std::vector<std::array<double, D>>(new_id);
        uint64_t old_id = 0;
        for (auto id : map) {
            if (id != Invalid) {
                new_vertices[id] = vertices[old_id];
            }
            ++old_id;
        }
        auto vtexData = std::make_unique<VertexData<D>        >(std::move(new_vertices));
        auto elemData = std::make_unique<ElementData          >(std::move(high_order_verts), NumberingConvention::GMSH);
	    auto pTagData = std::make_unique<ScalarMeshData<int>  >(std::move(element_ptags));
		auto eTagData = std::make_unique<ScalarMeshData<int>  >(std::move(element_etags));
    	mesh          = std::make_unique<GlobalSimplexMesh<D> >(std::move(elements), std::move(vtexData),
                                                                std::move(elemData), std::move(pTagData), std::move(eTagData), comm);
    } else {
        auto vtexData = std::make_unique<VertexData<D>        >(std::move(vertices));
	    auto elemData = std::unique_ptr<MeshData>(nullptr);
		auto pTagData = std::make_unique<ScalarMeshData<int>  >(std::move(element_ptags));
		auto eTagData = std::make_unique<ScalarMeshData<int>  >(std::move(element_etags));
    	mesh          = std::make_unique<GlobalSimplexMesh<D> >(std::move(elements), std::move(vtexData),
                                                                std::move(elemData), std::move(pTagData), std::move(eTagData), comm);
    }

    // boundary mesh
    auto vtexData = std::unique_ptr<MeshData>(nullptr);
	auto elemData = std::unique_ptr<MeshData>(nullptr);
	auto pTagData = std::make_unique<ScalarMeshData<int>      >(std::move(facet_ptags));
	auto eTagData = std::make_unique<ScalarMeshData<int>      >(std::move(facet_etags));
    auto bMesh    = std::make_unique<GlobalSimplexMesh<D - 1u>>(std::move(facets),   std::move(vtexData),
                                                                std::move(elemData), std::move(pTagData), std::move(eTagData), comm);
    mesh->setBoundaryMesh(std::move(bMesh));
    return mesh;
}

template class GlobalSimplexMeshBuilder<2ul>;
template class GlobalSimplexMeshBuilder<3ul>;

} // namespace tndm
