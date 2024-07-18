#ifndef MULTIPLYBOUNDARYTAGS_H
#define MULTIPLYBOUNDARYTAGS_H


#include "form/BC.h"
#include <array>
#include <vector>
#include <string>
#include <stdexcept>


namespace tndm {



class boundaryTag {
public:
    boundaryTag(std::string tagLabel, long tagID, int dimension, BC boundaryType)
        : tagLabel(tagLabel), tagID(tagID), dimension(dimension), boundaryType(boundaryType) {}

    long getTagID() const {return tagID;}
    std::string getLabel() const {return tagLabel;}
    inline BC getBoundaryType() const {return boundaryType;}
    std::vector<size_t> getFacesInLocalMesh() const { return facesInLocalMesh; }
    inline void addFaceInGlobalMesh(std::size_t faceIndex) {originalFaces.push_back(faceIndex);}
    const std::vector<size_t>& returnfacesInLocalMesh() const { return facesInLocalMesh; }
    inline void setFacesInLocalBoundaryMap(std::vector<size_t> facesInBoundaryMap ) {facesInLocalBoundaryMap=facesInBoundaryMap;}
    inline std::vector<size_t> getFacesInLocalBoundaryMap( ) {return facesInLocalBoundaryMap ;}
    inline void resizeFacesInLocalBoundaryMap (){facesInLocalBoundaryMap.resize(facesInLocalMesh.size());}
    inline void setfacesInLocalBoundaryMap(std::size_t index, std::size_t value) { facesInLocalBoundaryMap[index]=value; }
    inline std::vector<std::size_t> returnfacesInLocalBoundaryMap(){return facesInLocalBoundaryMap;}
    
    inline void computeMappingForLocalMesh(std::vector<std::vector<std::size_t>>& bcMapping){
    

        for (const auto& mapping_i: bcMapping){
            std::size_t globalMesh_i = mapping_i[0];
            std::size_t localMesh_i = mapping_i[1];

            for (const auto& originalFaces_i : originalFaces ){
                if (originalFaces_i==globalMesh_i){
                    facesInLocalMesh.push_back(localMesh_i);
                }

            }
        }
    }


private:

    std::string tagLabel;
    long tagID;
    int dimension;
    BC boundaryType; 
    std::vector<std::size_t> originalFaces;
    std::vector<std::size_t> facesInLocalMesh;
    std::vector<std::size_t> facesInLocalBoundaryMap;



};

 

inline std::vector<std::vector<std::size_t>> generateMappingMatrix
    ( const std::vector<std::size_t>& newIndices)  {

    std::vector<std::size_t> originalIndices;
    std::size_t maxValue = 0;
    bool foundValid = false;

    // Iterate through newOriginalIndices to find the maximum value excluding null values
    for (auto value : newIndices) {
        if (value != std::numeric_limits<std::size_t>::max()) {
            if (!foundValid || value > maxValue) {
                maxValue = value;
                foundValid = true;
            }
        }
    }

    // Generate originalIndices vector ranging from 0 to maxValue
    if (foundValid) {
        originalIndices.resize(maxValue + 1);
        std::iota(originalIndices.begin(), originalIndices.end(), 0);
    } else {
        originalIndices.clear();
    }

    std::vector<std::vector<std::size_t>> mappingMatrix;

    for (std::size_t i = 0; i < newIndices.size(); ++i) {
        if (newIndices[i] != std::numeric_limits<std::size_t>::max()) {
            mappingMatrix.push_back({newIndices[i], i});
        }
    }

    return mappingMatrix;
}

inline void setSizeForFacesInLocalBoundaryMap(std::vector<boundaryTag>& boundaryTags){
    for (auto& boundaryTag_i :boundaryTags ){
        boundaryTag_i.resizeFacesInLocalBoundaryMap();
    }
}




inline void computeMappingForBoundaries(std::vector<boundaryTag>& boundaryTags, std::vector<std::vector<std::size_t>>& bcMapping){
    for (auto& boundaryTag_i :boundaryTags ){
        boundaryTag_i.computeMappingForLocalMesh(bcMapping);
    }
}

inline std::vector<std::vector<size_t>> create2DMatrixFromBoundaryTags(const std::vector<boundaryTag>& boundaryTags) {
    std::vector<std::vector<size_t>> matrix;

    for (size_t i = 0; i < boundaryTags.size(); ++i) {
        const auto& facesInLocalMesh = boundaryTags[i].getFacesInLocalMesh();

        for (size_t j = 0; j < facesInLocalMesh.size(); ++j) {
            matrix.push_back({ i, j, facesInLocalMesh[j] });
        }
    }

    return matrix;
}


} //end tndm namespace

#endif // MULTIPLYBOUNDARYTAGS_H

