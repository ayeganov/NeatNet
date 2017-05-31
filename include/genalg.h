#ifndef __GENALG_H__
#define __GENALG_H__

#include <vector>

#include "genes.h"
#include "genome.h"
#include "species.h"
#include "innovation.h"
#include "phenotype.h"
#include "utils.h"
#include "params.h"

namespace neat
{


class GenAlg
{
private:
    std::vector<Genome> m_genomes;
    InnovationDB m_inno_db;

    std::size_t m_generation_count;
    GenomeID m_next_genome_id;
    SpeciesID m_next_species_id;
    std::vector<Genome> m_best_genomes;
    std::vector<Species> m_species;
    double m_best_ever_fitness;
    Params m_params;

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
    GenAlg(std::size_t num_inputs,
           std::size_t num_outputs);

    GenAlg(std::size_t num_inputs,
           std::size_t num_outputs,
           const Params& params);

    std::vector<SNeuralNetPtr> Epoch(const std::vector<double>& fitness_scores);
    std::vector<SNeuralNetPtr> CreateNeuralNetworks();


    // Getters and setters
    double BestEverFitness() const { return m_best_ever_fitness; }
    const Utils::RunningStat& SpeciesStats() { return m_num_species_stat; }
    const Utils::RunningStat& GenomeLinksStats() { return m_genome_links_stat; }
    const Utils::RunningStat& GenomeNeuronStats() { return m_genome_neuron_stat; }

    std::size_t Generation() { return m_generation_count; }

    SNeuralNetPtr BestNN() const;
    Genome BestGenome() const { return m_best_genomes[0]; }
    const std::vector<Genome>& BestGenomes() const { return m_best_genomes; }
    const std::vector<Genome>& GetGenomes() const { return m_genomes; }

    const std::vector<Species>& GetSpecies() { return m_species; }
    const Species* const GetSpecie(SpeciesID id) const;

};


};
#endif
