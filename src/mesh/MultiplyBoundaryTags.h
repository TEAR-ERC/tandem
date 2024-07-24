#ifndef MULTIPLYBOUNDARYTAGS_H
#define MULTIPLYBOUNDARYTAGS_H

#include "mesh/Simplex.h"
#include "form/BC.h"
#include <array>
#include <vector>
#include <string>
#include <stdexcept>


namespace tndm {

class baseBoundaryTag {
public:
    baseBoundaryTag(std::string tagLabel, long tagID, int dimension, BC boundaryType)
        : tagLabel(tagLabel), tagID(tagID), dimension(dimension), boundaryType(boundaryType) {}

    long getTagID() const { return tagID; }
    std::string getLabel() const { return tagLabel; }
    BC getBoundaryType() const { return boundaryType; }
    int getDimension() const { return dimension;}


    std::string tagLabel;
    long tagID;
    int dimension;
    BC boundaryType;
};


class localBoundaryTag : public baseBoundaryTag {
public:
    localBoundaryTag(std::string tagLabel, long tagID, int dimension, BC boundaryType)
        : baseBoundaryTag(tagLabel, tagID, dimension, boundaryType) {}


    std::vector<std::size_t> getFacesInLocalMesh() const { return facesInLocalMesh; }
    const std::vector<std::size_t>& returnfacesInLocalMesh() const { return facesInLocalMesh; }
    void setFacesInLocalBoundaryMap(std::vector<std::size_t> facesInBoundaryMap) { facesInLocalBoundaryMap = facesInBoundaryMap; }
    std::vector<std::size_t> getFacesInLocalBoundaryMap() const { return facesInLocalBoundaryMap; }
    void resizeFacesInLocalBoundaryMap() { facesInLocalBoundaryMap.resize(facesInLocalMesh.size()); }
    void setfacesInLocalBoundaryMap(std::size_t index, std::size_t value) { facesInLocalBoundaryMap[index] = value; }
    std::vector<std::size_t> returnfacesInLocalBoundaryMap() const { return facesInLocalBoundaryMap; }

    void addFacesInLocalMesh(const std::vector<std::size_t>& boundaryFacesInLocalMesh) {
        facesInLocalMesh = boundaryFacesInLocalMesh;
    }

private:
    std::vector<std::size_t> facesInLocalMesh;
    std::vector<std::size_t> facesInLocalBoundaryMap;
    
};


template <std::size_t D>
class globalBoundaryTag : public baseBoundaryTag {
public:
    globalBoundaryTag(std::string tagLabel, long tagID, int dimension, BC boundaryType)
        : baseBoundaryTag(tagLabel, tagID, dimension, boundaryType) {}

    
    void addElements(Simplex<D - 1> elements) { elementBoundary.push_back(elements); }

    localBoundaryTag returnLocalBoundaryTags(const LocalFaces<D-1u>& localMeshFaces) const {
        std::vector<std::size_t> boundaryFacesInLocalMesh;

        for (const auto& elementBoundary_i : elementBoundary) {
            auto localIndex = localMeshFaces.getLocalIndex(elementBoundary_i);
            boundaryFacesInLocalMesh.push_back(localIndex);
        }

        localBoundaryTag boundaryTagInLocalMesh(tagLabel, tagID, dimension, boundaryType);
        boundaryTagInLocalMesh.addFacesInLocalMesh(boundaryFacesInLocalMesh);

        return boundaryTagInLocalMesh;
    }



    //std::vector<std::size_t> facesInLocalMesh;
    //std::vector<std::size_t> facesInLocalBoundaryMap;
    std::vector<Simplex<D - 1>> elementBoundary;
};

template <std::size_t D> 
inline std::vector<localBoundaryTag> getLocalBoundaryTagFromGlobal(const std::vector<globalBoundaryTag<D>>& globalBoundaryTags, const LocalFaces<D-1u>& localFacesRecivied){
    
    std::vector<localBoundaryTag> localTagsVector;

        for (auto& globalboundaryTag_i :globalBoundaryTags ){
        localTagsVector.emplace_back(globalboundaryTag_i.returnLocalBoundaryTags(localFacesRecivied));
    
    }

    return localTagsVector;

}
template <std::size_t D> 
std::vector<globalBoundaryTag<D>> ReturnBoundariesOfType(const std::vector<globalBoundaryTag<D>>& globalBoundaryTags, BC bcType) {
        std::vector<globalBoundaryTag<D>> result;
        for (const auto& tag : globalBoundaryTags) {
            if (tag.getBoundaryType() == bcType) {
                result.push_back(tag);
            }
        }
    return result;
    }

template <std::size_t D> 
std::vector<globalBoundaryTag<D>> returnFaultTypeBoundaries(const std::vector<globalBoundaryTag<D>>& globalBoundaryTags) {
    return ReturnBoundariesOfType(globalBoundaryTags,BC::Fault);
}


/*
template <std::size_t D> 
inline void setSizeForFacesInLocalBoundaryMap(std::vector<boundaryTag<D>>& boundaryTags){
    for (auto& boundaryTag_i :boundaryTags ){
        boundaryTag_i.resizeFacesInLocalBoundaryMap();
    }
}




template <std::size_t D> 
inline void computeMappingForBoundaries(std::vector<boundaryTag<D>>& boundaryTags, std::vector<std::vector<std::size_t>>& bcMapping){
    for (auto& boundaryTag_i : boundaryTags ){
        boundaryTag_i.computeMappingForLocalMesh(bcMapping);
    }
}

template <std::size_t D> 
inline std::vector<std::vector<size_t>> create2DMatrixFromBoundaryTags(const std::vector<boundaryTag<D>>& boundaryTags) {
    std::vector<std::vector<size_t>> matrix;

    for (size_t i = 0; i < boundaryTags.size(); ++i) {
        const auto& facesInLocalMesh = boundaryTags[i].getFacesInLocalMesh();

        for (size_t j = 0; j < facesInLocalMesh.size(); ++j) {
            matrix.push_back({ i, j, facesInLocalMesh[j] });
        }
    }

    return matrix;
}

*/
} //end tndm namespace

#endif // MULTIPLYBOUNDARYTAGS_H

