#include "cpplinq.hpp"
#include "species.h"
#include "utils.h"

namespace neat
{

Species::Species(Genome& originator, SpeciesID id): m_leader(originator),
                                                    m_id(id),
                                                    m_members(),
                                                    m_gens_no_improvement(0),
                                                    m_age(0),
                                                    m_spawns_required(0)
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

        if(m_age < YoungBonusThreshold)
        {
            fitness *= 1.3;
        }
        if(m_age > OldPenaltyThreshold)
        {
            fitness *= .7;
        }
        double shared_fitness = fitness / m_members.size();
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
        int max_idx = m_members.size() * SURVIVAL_RATE;
        int the_one = random.RandomClamped(0, max_idx);
        return cpplinq::from(m_members) >> cpplinq::element_at_or_default(the_one);
    }
}

//=============================PRIVATE METHODS=================================


};
