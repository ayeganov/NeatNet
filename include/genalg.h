#ifndef __GENALG_H__
#define __GENALG_H__

#include <vector>

#include "genes.h"
#include "genome.h"
#include "species.h"
#include "innovation.h"
#include "phenotype.h"
#include "utils.h"

namespace neat
{

const int NUM_GENS_ALLOWED_NO_IMPROV = 10;
const int NUM_BEST_GENOMES = 5;
const double COMPATIBILITY_THRESHOLD = 0.25;
const std::size_t POPULATION_SIZE = 50;
const std::size_t MAX_NEURONS = 100;
const double ADD_NEURON_CHANCE = 0.08;
const double ADD_LINK_CHANCE = 0.07;
const double ADD_RECUR_LINK_CHANCE = 0.05;
const double MUTATION_CHANCE = 0.8;
const double CROSSOVER_CHANCE = 0.7;
const double MAX_PERTURBATION = 0.3;
const double NEW_WEIGHT_CHANCE = 0.1;
const double ACTIVATION_MUTATION_CHANCE = 0.1;
const double MAX_ACTIVATION_PERTURBATION = 0.1;
const std::size_t NUM_ADD_LINK_ATTEMPTS = 5;
const std::size_t NUM_FIND_OLD_LINK_ATTEMPTS = 5;
const std::size_t NUM_ADD_RECUR_LINK_ATTEMPTS = 5;


class GenAlg
{
private:
    std::vector<Genome> m_genomes;
    InnovationDB m_inno_db;

    std::size_t m_population_size;
    std::size_t m_generation_count;
    GenomeID m_next_genome_id;
    SpeciesID m_next_species_id;
    std::vector<Genome*> m_best_genomes;
    std::vector<Species> m_species;
    double m_best_ever_fitness;

    Utils::RunningStat m_num_species_stat;
    Utils::RunningStat m_genome_links_stat;
    Utils::RunningStat m_genome_neuron_stat;

    void PurgeSpecies();
    void UpdateGenomeScores(const std::vector<double>& fitness_scores);
    void UpdateBestGenomes();
    void SpeciateGenomes();
    void UpdateSpeciesFitness();
    void CalculateSpeciesSpawnAmounts();
    std::vector<Genome> CreateNewPopulation();
    Genome TournamentSelect(int num_battles);
    Genome MakeCrossoverBaby(Species& s, GenomeID next_id);

    void RunEpochStatistics();
    void RunLongTermStatistics();

public:
    GenAlg(std::size_t pop_size,
           std::size_t num_inputs,
           std::size_t num_outputs);

    std::vector<SNeuralNetPtr> Epoch(const std::vector<double>& fitness_scores);
    std::vector<SNeuralNetPtr> CreateNeuralNetworks();


    // Getters and setters
    double BestEverFitness() const { return m_best_ever_fitness; }
    const Utils::RunningStat& SpeciesStats() { return m_num_species_stat; }
    const Utils::RunningStat& GenomeLinksStats() { return m_genome_links_stat; }
    const Utils::RunningStat& GenomeNeuronStats() { return m_genome_neuron_stat; }

    std::size_t Generation() { return m_generation_count; }

    SNeuralNetPtr BestNN() const;
    Genome BestGenome() const { return m_genomes[0]; }

};


};
#endif
