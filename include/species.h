#ifndef __SPECIES_H__
#define __SPECIES_H__

#include "genome.h"

#include <vector>

namespace neat
{

class Species
{
private:
    Genome m_leader;

    SpeciesID m_id;

    std::vector<Genome*> m_members;
    // number of generations this species has not experienced any improvement
    // in fitness
    unsigned int m_gens_no_improvement;
    // total number of generations this species has been alive
    unsigned int m_age;

    double m_spawns_required;


public:
    Species(Genome& originator, SpeciesID id);

    void AdjustFitness();

    void AddMember(Genome& new_member);

    void Purge();

    void CalculateSpawnAmount();

    Genome Spawn();

    friend bool operator<(const Species& lhs, const Species& rhs)
    {
        return lhs.m_leader.Fitness() > rhs.m_leader.Fitness();
    }

};

}
#endif
