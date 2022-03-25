#pragma once

#include "Individual.h"

#include <vector>

class Style;

enum SortCriterion { SORTBY_SCORE = 0, SORTBY_IDNUM = 1, SORTBY_COLORSPACE = 2, SORTBY_RENDERTIME = 3, SORTBY__MAX = 4 };

class Population {
public:
    Population();
    ~Population() {}

    typedef std::vector<Individual::shp>::iterator iterator;
    typedef std::vector<Individual::shp>::reverse_iterator reverse_iterator;
    iterator beginChildren() { return Children.begin(); }
    iterator endChildren() { return Children.end(); }
    size_t sizeChildren() { return Children.size(); }
    Individual::shp getChildren(size_t i) { return Children[i]; }
    void insertChildren(Individual::shp i) { Children.push_back(i); }

    iterator beginParents() { return Parents.begin(); }
    iterator endParents() { return Parents.end(); }
    reverse_iterator rbeginParents() { return Parents.rbegin(); }
    reverse_iterator rendParents() { return Parents.rend(); }
    size_t sizeParents() { return Parents.size(); }
    Individual::shp getParents(size_t i) { return Parents[i]; }
    void insertParents(Individual::shp i) { Parents.push_back(i); }

    iterator beginZoo() { return Zoo.begin(); }
    iterator endZoo() { return Zoo.end(); }
    reverse_iterator rbeginZoo() { return Zoo.rbegin(); }
    reverse_iterator rendZoo() { return Zoo.rend(); }
    size_t sizeZoo() { return Zoo.size(); }
    Individual::shp getZoo(size_t i) { return Zoo[i]; }
    void insertZoo(Individual::shp i) { Zoo.push_back(i); }

    void MoveParentToZoo(size_t i);
    void MoveChildToZoo(size_t i);
    void MoveChildToParent(size_t i);
    void MoveZooToParent(size_t i);

    void ClearImages(); // Clear the images of all individuals

    void DeleteSpanOfIndividuals(size_t index, size_t count = 1, bool AmParent = false, bool AmChild = false); // Delete count individuals, starting with index

    void SavePopulation(const std::string& outFNameBase, Style* style);

    void Sort(const SortCriterion SortBy, bool Z = true, bool P = true, bool C = false);

    std::string getFNameBase() const { return m_FNameBase; }
    void setFNameBase(std::string val) { m_FNameBase = val; }

private:
    // The three components of the population. Every item in the vector is valid. Cannot be null.
    // TODO Can we unify these in some way? What if everything lived in the Zoo, but some were also in Parents or Children?
    // Would have to worry about dangling pointers if deleted from Zoo.
    std::vector<Individual::shp> Children;
    std::vector<Individual::shp> Parents;
    std::vector<Individual::shp> Zoo; // All of the non-child, non-parent individuals.

    std::string m_FNameBase; // Filename to save to, without the suffix
};
