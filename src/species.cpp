#include <cassert>

#include "cpplinq.hpp"
#include "species.h"
#include "utils.h"

namespace neat
{

Species::Species(Genome& originator, SpeciesID id, Params* params): m_leader(originator),
                                                                    m_id(id),
                                                                    m_members(),
                                                                    m_gens_no_improvement(0),
                                                                    m_age(0),
                                                                    m_spawns_required(0),
                                                                    m_params(params)
{
    m_members.push_back(&originator);
}

//==============================PUBLIC METHODS=================================
void Species::AddMember(Genome& new_member)
{
    if(new_member.Fitness() > m_leader.Fitness())
    {
        m_leader = new_member;
        m_gens_no_improvement = 0;
    }
    m_members.push_back(&new_member);
}

void Species::Purge()
{
    m_members.clear();
    ++m_age;
    ++m_gens_no_improvement;
    m_spawns_required = 0;
}

void Species::AdjustFitness()
{
    for(auto genome : m_members)
    {
        double fitness = genome->Fitness();
        assert(fitness >= 0);

        if(m_age < m_params->YoungBonusThreshold())
        {
            fitness *= m_params->YoungBonusScaler();
        }
        if(m_age > m_params->OldPenaltyThreshold())
        {
            fitness *= m_params->OldPenaltyScaler();
        }
        double shared_fitness = fitness / m_members.size();

        assert(shared_fitness >= 0);
        genome->SetAjustedFitness(shared_fitness);
    }
}

void Species::CalculateSpawnAmount()
{
    m_spawns_required = cpplinq::from(m_members)
        >> cpplinq::sum([](const Genome* const g)
            {
                return g->AmountToSpawn();
            });
    assert(m_spawns_required >= 0);
}

Genome* Species::Spawn()
{
    auto& random = Utils::DefaultRandom::Instance();

    assert(m_members.size() > 0);
    if(m_members.size() == 1)
    {
        return m_members[0];
    }
    else
    {
        int max_idx = m_members.size() * m_params->SurvivalRate() + 1;
        int the_one = random.RandomClamped(0, max_idx);
        return m_members[the_one];
    }
}

//=============================PRIVATE METHODS=================================


};
