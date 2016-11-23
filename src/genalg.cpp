#include "cpplinq.hpp"

#include "genalg.h"
#include "utils.h"

namespace neat
{

using namespace cpplinq;

/**
 * IMPORTANT: Initial population of genomes is structurally identical - the
 *            only thing that is different is the weights. As a consequence the
 *            innovation database gets created from a temporary genome to
 *            record all connections that were used in the original populaiton.
 */
GenAlg::GenAlg(std::size_t pop_size,
               std::size_t num_inputs,
               std::size_t num_outputs)
{
    m_genomes = range(0, pop_size) >> select(
        [&](int _)
            {
                return Genome(m_next_genome_id++, num_inputs, num_outputs);
            })
        >> to_vector();
    Genome tmp(1, num_inputs, num_outputs);
    m_inno_db = InnovationDB(tmp.NeuronGenes(), tmp.NeuronLinks());
}


std::vector<SNeuralNetPtr> GenAlg::Epoch(const std::vector<double>& fitness_scores)
{
    // Remove species that have not been improving for configured number of
    // generations
    PurgeSpecies();
    UpdateGenomeScores(fitness_scores);
    // keep in mind the m_genomes vector will be sorted ascendingly after the next
    // step
    UpdateBestGenomes();
    SpeciateGenomes();
    UpdateSpeciesFitness();
    CalculateSpeciesSpawnAmounts();
}


//=================================PRIVATE METHODS==============================
void GenAlg::PurgeSpecies()
{
    auto current_species = m_species.begin();
    while(current_species != m_species.end())
    {
        current_species->Purge();

        if(current_species->GensNoImprovement() > NUM_GENS_ALLOWED_NO_IMPROV &&
            current_species->LeaderFitness() < m_best_ever_fitness)
        {
            current_species = m_species.erase(current_species);
            continue;
        }
        ++current_species;
    }
}

void GenAlg::UpdateGenomeScores(const std::vector<double>& fitness_scores)
{
    from(m_genomes) >> zip_with(fitness_scores)
    >> for_each([this](std::pair<Genome, double>& p)
        {
            p.first.SetFitness(p.second);
        });
}

void GenAlg::UpdateBestGenomes()
{
    std::sort(m_genomes.begin(), m_genomes.end());
    m_best_ever_fitness = std::max(m_best_ever_fitness, m_genomes[0].Fitness());

    m_best_genomes = from(m_genomes) >> take(NUM_BEST_GENOMES)
                        >> select([](Genome& g) { return &g; }) >> to_vector();
}

void GenAlg::SpeciateGenomes()
{
    for(const auto& genome : m_genomes)
    {
        bool is_new_species = true;
        for(auto& species : m_species)
        {
            double compat_score = genome.CalculateCompatabilityScore(species.Leader());
            if(compat_score > COMPATIBILITY_THRESHOLD)
            {
                species.AddMember(genome);
                is_new_species = false;
            }
        }

        if(is_new_species)
        {
            Species new_species(genome, m_next_species_id++);
            m_species.push_back(new_species);
        }
    }
}

void GenAlg::UpdateSpeciesFitness()
{
    from(m_species) >> for_each([](Species& s) { s.AdjustFitness(); });
}

/**
 * This method depends on UpdateSpeciesFitness implicitly. Updating species
 * fitness actually adjusts the fitness of each member genome. Class Species
 * holds a vector of POINTERS to genomes within GenAlg, so by the time this
 * method should be called all genomes must contain their new adjusted fitness.
 */
void GenAlg::CalculateSpeciesSpawnAmounts()
{
    Utils::RunningStat rs;
    from(m_genomes) >> for_each( [&rs](const Genome& g) { rs.Push(g.GetAdjustedFitness()); });
    from(m_genomes) >> for_each( [&rs](Genome& g)
        {
            double to_spawn = g.GetAdjustedFitness() / rs.Mean();
            g.SetAmountToSpawn(to_spawn);
        });
    from(m_species) >> for_each( [](Species& s) { s.CalculateSpawnAmount(); });
}

};
