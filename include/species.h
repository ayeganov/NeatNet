#ifndef __SPECIES_H__
#define __SPECIES_H__

#include "genome.h"
#include "params.h"

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
    std::size_t m_gens_no_improvement;
    // total number of generations this species has been alive
    std::size_t m_age;

    double m_spawns_required;

    Params* m_params;


public:
    Species(Genome& originator, SpeciesID id, Params* params);

    void AdjustFitness();

    void AddMember(Genome& new_member);

    void Purge();

    void CalculateSpawnAmount();

    Genome* Spawn();

    friend bool operator<(const Species& lhs, const Species& rhs)
    {
        return lhs.m_leader.Fitness() > rhs.m_leader.Fitness();
    }

    // Member getters and setters
    std::size_t GensNoImprovement() const { return m_gens_no_improvement; }
    std::size_t Age() const { return m_age; }
    double SpawnsRequired() const { return m_spawns_required; }
    double LeaderFitness() const { return m_leader.Fitness(); }

    const Genome& Leader() const { return m_leader; }
    std::size_t Size() const { return m_members.size(); }

    SpeciesID ID() const { return m_id; }
};

}
#endif
