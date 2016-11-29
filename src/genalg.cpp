#include "cpplinq.hpp"

#include "genalg.h"
#include "phenotype.h"
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
               std::size_t num_outputs): m_generation_count(0),
                                         m_next_species_id(2),
                                         m_best_genomes(),
                                         m_best_ever_fitness(0.0),
                                         m_random()
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
    m_genomes = CreateNewPopulation();
    return CreateNeuralNetworks();
}


std::vector<SNeuralNetPtr> GenAlg::CreateNeuralNetworks()
{
    return from(m_genomes) >> select([](const Genome& g)
        {
            return std::make_shared<NeuralNet>(g.NeuronGenes(), g.NeuronLinks(), 1);
        })
        >> to_vector();
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
    from(m_genomes) >> zip_with(from(fitness_scores))
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

/**
 * 1. Grab leader from species
 * 2. Spawn two unique genomes from species
 * 3. Crossover two spawns
 * 4. Set baby ID
 * 5. Call AddNeuron, AddLink, MutateWeights, MutateActivationResponse
 * 6. Sort baby genes
 * 7. Check if the new population contains enough genomes
 * 8. Tournament select in case not enough genomes were crossed over
 */
std::vector<Genome> GenAlg::CreateNewPopulation()
{
    std::vector<Genome> new_pop{m_genomes.size()};
    // grab all the leaders from each species
    from(m_species) >> for_each([&new_pop](const Species& s) { new_pop.push_back(s.Leader()); });

    auto total_num_spawned = new_pop.size();
    for(auto species : m_species)
    {
        // break out if the population is large enough
        if(total_num_spawned >= POPULATION_SIZE) break;

        int species_num_to_spawn = std::round(species.SpawnsRequired()) - 1;
        while(species_num_to_spawn > 0)
        {
            Genome baby;
            if(species.Size() > 1)
            {
                baby = MakeCrossoverBaby(species, m_next_genome_id++);
            }
            else
            {
                baby = *species.Spawn();
                baby.SetID(m_next_genome_id++);
            }

            if(baby.NumGenes() < MAX_NEURONS) baby.AddNeuron(ADD_NEURON_CHANCE,
                                                             m_inno_db,
                                                             NUM_FIND_OLD_LINK_ATTEMPTS);
            baby.AddLink(ADD_LINK_CHANCE,
                         ADD_RECUR_LINK_CHANCE,
                         m_inno_db,
                         NUM_ADD_RECUR_LINK_ATTEMPTS,
                         NUM_ADD_LINK_ATTEMPTS);

            baby.MutateWeights(MUTATION_CHANCE, NEW_WEIGHT_CHANCE, MAX_PERTURBATION);
            baby.MutateActivationResponse(ACTIVATION_MUTATION_CHANCE, MAX_ACTIVATION_PERTURBATION);

            baby.SortLinks();

            new_pop.push_back(baby);
            --species_num_to_spawn;
            ++total_num_spawned;

            if(total_num_spawned >= POPULATION_SIZE)
            {
                break;
            }
        }
    }

    if(new_pop.size() < POPULATION_SIZE)
    {
        auto rqrd = POPULATION_SIZE - new_pop.size();
        while(rqrd > 0)
        {
            new_pop.push_back(TournamentSelect(m_genomes.size() / 5));
            --rqrd;
        }
    }
    return new_pop;
}

Genome GenAlg::MakeCrossoverBaby(Species& species, GenomeID next_id) const
{
    Genome baby;
    Genome* mom = species.Spawn();
    if(m_random.RandomDouble() < CROSSOVER_CHANCE)
    {
        // find dad
        int num_attempts = 5;
        Genome* dad = range(0, num_attempts) >> select([&species](int _)
            {
                return species.Spawn();
            })
        >> cpplinq::first_or_default([&mom](const Genome* dad)
            {
                return dad->ID() != mom->ID();
            });

        if(dad)
        {
            baby = mom->Crossover(*dad, m_inno_db, next_id);
        }
        else
        {
            // couldn't find dad, we'll have to settle for mom alone
            baby = *mom;
        }
    }
    else
    {
        baby = *mom;
    }

    baby.SetID(next_id);
    return baby;
}

Genome GenAlg::TournamentSelect(int num_battles)
{
    int lower = 0;
    int upper = m_genomes.size() - 1;
    int chosen_one = 0;
    double best_fitness_so_far = 0;

    for(int b = 0; b < num_battles; b++)
    {
        int contender = m_random.RandomClamped(lower, upper);
        if(m_genomes[contender].Fitness() > best_fitness_so_far)
        {
            chosen_one = contender;
            best_fitness_so_far = m_genomes[contender].Fitness();
        }
    }

    return m_genomes[chosen_one];
}

};
