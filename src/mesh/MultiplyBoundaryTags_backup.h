#ifndef MULTIPLYBOUNDARYTAGS_H
#define MULTIPLYBOUNDARYTAGS_H


#include "form/BC.h"
#include <array>
#include <vector>
#include <string>
#include <stdexcept>

namespace tndm {



class boundaryTag1 {
public:
    boundaryTag1(std::string tagLabel, long tagID, int dimension, BC boundaryType)
        : tagLabel(tagLabel), tagID(tagID), dimension(dimension), boundaryType(boundaryType) {}

    long getTagID() const {return tagID;}
    std::string getLabel() const {return tagLabel;}
    BC getBoundaryType() const {return boundaryType;}
    void addFaceInGlobalMesh(std::size_t faceIndex) {originalFaces.push_back(faceIndex);}

private:

    std::string tagLabel;
    long tagID;
    int dimension;
    BC boundaryType; 
    std::vector<std::size_t> originalFaces;
    std::vector<std::size_t> facesInLocalMesh;


};

struct boundary {
    std::vector<BC> bc;
    std::string boundaryName;
};

class BoundaryTag {
private:
    long tagID;
    std::string tagLabel;
    std::vector<long> elementTagIds; 

public:
    BoundaryTag(const std::string& label, long tagID)
        : tagLabel(label), tagID(tagID) { }

    void addElementTagID(long elementID) {
        elementTagIds.push_back(elementID);
    }

    const std::string& getTagLabel() const {
        return tagLabel;
    }

    long getTagID() const {
        return tagID ;
    }

    long getTagIdByIndex(size_t index) const {
        if (index < elementTagIds.size()) {
            return elementTagIds[index];
        } else {
            throw std::out_of_range("Index out of range");
        }
    }

    size_t getNumberOfIds() const {
        return elementTagIds.size();
    }
};



class BoundaryManger {
private:
    std::vector<BoundaryTag> boundaries;
    BC bcType;

    void addElementIDToLastBoundary(long elementID) {
        boundaries.back().addElementTagID(elementID);
    }

    void addBoundary(long id, const std::string& label) {
        boundaries.emplace_back(label, id);  // Parameter order corrected
    }

public:
    void setBCType(BC bcType){bcType=bcType;}
    void returnVectorOfBC(std::vector<long> tagIDs ){
        

        for (auto& manager : boundaries){
            std::vector<bool> vectorOfBCs(tagIDs.size(), false);
            std::vector<BC> bcVector(vectorOfBCs.size(), BC::None);

            for (std::size_t i = 0; i < tagIDs.size(); ++i) {

                long tagID_i=tagIDs[i];
                if (manager.getTagID() == tagID_i){
                    vectorOfBCs[i]=true;
                }

            }
            for (std::size_t i = 0; i < vectorOfBCs.size(); ++i) {
                bcVector[i] = vectorOfBCs[i] ? bcType : BC::None;
            }

        }
    }

    void addTagIdToBoundaryManager(const std::string& boundaryName, long tagID, long elementID) {
        for (auto& manager : boundaries) {
            if (manager.getTagID() == tagID) {
                manager.addElementTagID(elementID);  // Corrected function name
                return;
            }
        }
        // If BoundaryTag with the given tagID does not exist, create a new one
        addBoundary(tagID, boundaryName);
        addElementIDToLastBoundary(elementID);
    }

    const BoundaryTag& getBoundaryManager(size_t index) const {
        if (index < boundaries.size()) {
            return boundaries[index];
        } else {
            throw std::out_of_range("Index out of range");
        }
    }

    size_t getNumberOfBoundaryManagers() const {
        return boundaries.size();
    }
};



class BoundaryCEO {

private:
    BoundaryManger faultsTags;
    BoundaryManger dieterichBoundaryTags;
    BoundaryManger normalBoundaryTags;
    std::vector<long> boundaryCounter;
    std::vector<long> tagIDsCounter;


public:
void addTagID(long tagID){tagIDsCounter.push_back(tagID);}
void addNewEntry(BC bcType,const std::string& boundaryName, long tagID, long elementID,long boundryCounter_i){

    boundaryCounter.push_back(boundryCounter_i);
    switch (bcType)
    {
        case BC::Dirichlet:
            dieterichBoundaryTags.addTagIdToBoundaryManager(boundaryName,  tagID,  elementID);
            tagIDsCounter.push_back(tagID);
            break;
        case BC::Fault:
            faultsTags.addTagIdToBoundaryManager( boundaryName,  tagID,  elementID);
            tagIDsCounter.push_back(tagID);
            break;
        case BC::Natural:
            normalBoundaryTags.addTagIdToBoundaryManager( boundaryName,  tagID,  elementID);
            tagIDsCounter.push_back(tagID);
            break;
        default:
            throw std::runtime_error("Not sure what I should do now??");
    }
}

void generateBoolVector(BC bcType){

    std::vector<bool> vectorOfBCs(tagIDsCounter.size(), false);
    switch (bcType)
    {
        case BC::Fault:
            faultsTags.setBCType(BC::Fault);
            faultsTags.returnVectorOfBC(tagIDsCounter);
            break;
           
    }

}


};


} //end tndm namespace

#endif // MULTIPLYBOUNDARYTAGS_H

