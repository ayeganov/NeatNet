#include <cassert>
#include <stdexcept>

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
GenAlg::GenAlg(std::size_t num_inputs,
               std::size_t num_outputs,
               const Params& params): m_generation_count(0),
                                         m_next_genome_id(0),
                                         m_next_species_id(0),
                                         m_best_genomes(),
                                         m_best_ever_fitness(0.0),
                                         m_params(params)
{
    m_genomes = range(0, m_params.PopulationSize()) >> select(
        [&](int _)
            {
                return Genome(m_next_genome_id++, num_inputs, num_outputs, &m_params);
            })
        >> to_vector();
    Genome tmp(1, num_inputs, num_outputs, &m_params);
    m_inno_db = InnovationDB(tmp.NeuronGenes(), tmp.NeuronLinks());
}

GenAlg::GenAlg(std::size_t num_inputs,
               std::size_t num_outputs): GenAlg(num_inputs, num_outputs, Params())
{}


std::vector<SNeuralNetPtr> GenAlg::Epoch(const std::vector<double>& fitness_scores)
{
    if(m_genomes.size() != fitness_scores.size())
    {
//      std::cerr << "Error: GenAlg::Epoch number of scores doesn't match number of genomes" << std::endl;
      throw std::invalid_argument("GenAlg::Epoch number of scores doesn't match number of genomes");
    }

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

    RunEpochStatistics();
    RunLongTermStatistics();
    ++m_generation_count;

    return CreateNeuralNetworks();
}


std::vector<SNeuralNetPtr> GenAlg::CreateNeuralNetworks()
{
    return from(m_genomes) >> select([](const Genome& g)
        {
            return std::make_shared<NeuralNet>(g.NeuronGenes(), g.NeuronLinks());
        })
        >> to_vector();
}


SNeuralNetPtr GenAlg::BestNN() const
{
    using std::to_string;
    if(m_best_genomes.size() > 0)
    {
        return std::make_shared<NeuralNet>(m_best_genomes[0]);
    }
    else
    {
        return std::shared_ptr<NeuralNet>(nullptr);
    }
}


const Species* const GenAlg::GetSpecie(SpeciesID id) const
{
    for(auto& s : m_species)
    {
        if(s.ID() == id)
        {
            return &s;
        }
    }
    throw new std::invalid_argument("Invalid SpeciesID provided: " + std::to_string(id));
}


//=================================PRIVATE METHODS==============================

void GenAlg::RunEpochStatistics()
{
    m_genome_links_stat.Clear();
    m_genome_neuron_stat.Clear();
    for(auto& g : m_genomes)
    {
        m_genome_links_stat.Push(g.NumLinks());
        m_genome_neuron_stat.Push(g.NumNeurons());
    }
}

void GenAlg::RunLongTermStatistics()
{
    m_num_species_stat.Push(m_species.size());
}

void GenAlg::PurgeSpecies()
{
    auto current_species = m_species.begin();
    auto num_gens_allowed_no_improv = m_params.NumGensAllowedNoImprov();
    while(current_species != m_species.end())
    {
        current_species->Purge();

        if(current_species->GensNoImprovement() > num_gens_allowed_no_improv &&
            current_species->LeaderFitness() < m_best_ever_fitness)
        {
            current_species = m_species.erase(current_species);
        }
        else
        {
            ++current_species;
        }
    }
}

void GenAlg::UpdateGenomeScores(const std::vector<double>& fitness_scores)
{
    for(int i = 0; i < m_genomes.size(); ++i)
    {
        assert(fitness_scores[i] >= 0);
        m_genomes[i].SetFitness(fitness_scores[i]);
    }
    std::sort(m_genomes.begin(), m_genomes.end());
}

void GenAlg::UpdateBestGenomes()
{
    m_best_genomes.clear();
    m_best_ever_fitness = std::max(m_best_ever_fitness, m_genomes[0].Fitness());

    auto num_best_genomes = m_params.NumBestGenomes();
    for(int i = 0; i < num_best_genomes; ++i)
    {
        m_best_genomes.push_back(m_genomes[i]);
    }
}

void GenAlg::SpeciateGenomes()
{
    bool added = false;
    auto compatibility_threshold = m_params.CompatibilityThreshold();
    for(auto& genome : m_genomes)
    {
        for(auto& species : m_species)
        {
            double diff_score = genome.CalculateDifferenceScore(species.Leader());
            if(diff_score <= compatibility_threshold)
            {
                species.AddMember(genome);
                genome.SetSpeciesID(species.ID());
                added = true;
                break;
            }
        }

        if(!added)
        {
            Species new_species(genome, m_next_species_id++, &m_params);
            genome.SetSpeciesID(new_species.ID());
            m_species.push_back(new_species);
        }
        added = false;
    }
}

void GenAlg::UpdateSpeciesFitness()
{
    for(auto& s : m_species)
    {
        s.AdjustFitness();
    }
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
    from(m_genomes) >> for_each( [&rs](const Genome g) { rs.Push(g.GetAdjustedFitness()); });

    for(auto& g : m_genomes)
    {
        double to_spawn = g.GetAdjustedFitness() / rs.Mean();
        g.SetAmountToSpawn(to_spawn);
    }

    for(auto& s : m_species)
    {
        s.CalculateSpawnAmount();
    }
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
    std::vector<Genome> new_pop;
    auto population_size = m_params.PopulationSize();

    Genome baby;
    auto total_num_spawned = new_pop.size();
    for(auto& species : m_species)
    {
        // break out if the population is large enough
        if(total_num_spawned >= population_size) break;

        bool leader_taken = false;
        int species_num_to_spawn = std::round(species.SpawnsRequired());
        while(species_num_to_spawn--)
        {
            if(!leader_taken)
            {
                baby = species.Leader();
                leader_taken = true;
            }
            else
            {
                if(species.Size() == 1)
                {
                    baby = *species.Spawn();
                    baby.SetID(m_next_genome_id++);
                }
                else if(species.Size() > 1)
                {
                    baby = MakeCrossoverBaby(species, m_next_genome_id++, baby);
                }
                else
                {
                    baby = *species.Spawn();
                    baby.SetID(m_next_genome_id++);
                }

                if(baby.NumHiddenNeurons() < m_params.MaxNeurons())
                {
                    baby.AddNeuron(m_params.AddNeuronChance(),
                                   m_inno_db,
                                   m_params.NumFindOldLinkAttempts());
                }

                baby.AddLink(m_params.AddLinkChance(),
                             m_params.AddRecurLinkChance(),
                             m_inno_db,
                             m_params.NumAddRecurLinkAttempts(),
                             m_params.NumAddLinkAttempts());

                baby.MutateWeights(m_params.MutationChance(),
                                   m_params.NewWeightChance(),
                                   m_params.MaxPerturbation());
                baby.MutateActivationResponse(m_params.ActivationMutationChance(),
                                              m_params.MaxActivationPerturbation());
            }
            baby.SortLinks();

            new_pop.push_back(baby);
            ++total_num_spawned;

            if(total_num_spawned >= population_size)
            {
                break;
            }
        }
    }

    if(new_pop.size() < population_size)
    {
        auto rqrd = population_size - new_pop.size();
        while(rqrd > 0)
        {
            new_pop.push_back(TournamentSelect(m_genomes.size() / 5));
            --rqrd;
        }
    }
    return new_pop;
}

Genome GenAlg::MakeCrossoverBaby(Species& species, GenomeID next_id, Genome& prev_baby)
{
    Genome baby;
    Genome* mom = species.Spawn();
    auto& random = Utils::DefaultRandom::Instance();

    if(random.RandomDouble() < m_params.CrossoverChance())
    {
        // find dad
        int num_attempts = 5;
        Genome* dad = species.Spawn();
        while(dad->ID() == mom->ID() && num_attempts--)
        {
            dad = species.Spawn();
        }

        if(dad->ID() != mom->ID())
        {
            baby = dad->Crossover(*mom, m_inno_db, next_id);
        }
        else
        {
            baby = prev_baby;
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
    auto& random = Utils::DefaultRandom::Instance();

    for(int b = 0; b < num_battles; b++)
    {
        int contender = random.RandomClamped(lower, upper);
        if(m_genomes[contender].Fitness() > best_fitness_so_far)
        {
            chosen_one = contender;
            best_fitness_so_far = m_genomes[contender].Fitness();
        }
    }

    return m_genomes[chosen_one];
}

};
