#ifndef __GENOME_H__
#define __GENOME_H__

/**
 * Author: Aleksandr Yeganov
 *
 * Definition of a genome used in the NEAT algorithm.
 */

#include <vector>
#include <memory>

#include "genes.h"
#include "innovation.h"
#include "utils.h"
#include "params.h"

namespace neat
{

class Genome
{
private:
    GenomeID m_genome_id;
    std::vector<NeuronGene> m_neuron_genes;
    std::vector<LinkGene> m_link_genes;

    double m_fitness;
    double m_adjusted_fitness;
    double m_amount_to_spawn;

    std::size_t m_num_inputs;
    std::size_t m_num_outputs;

    SpeciesID m_species_id;

    Params* m_params;

    /**
     * Returns true if the given link is already part of the genome
     * @param neuron_from_id - from neuron id aka neuron providing input
     * @param neuron_to_id - neuron that will receive the output
     */
    bool IsExistingLink(NeuronID neuron_from_id, NeuronID neuron_to_id);

    /**
     * Given neuron id finds the index of said neuron in the m_neuron_genes
     * vector.
     * @param neuron_id - id of the neuron to find
     */
    int GetNeuronIndex(NeuronID neuron_id);

    bool FindNonRecurrentNeuron(NeuronID& neuron_id_from, NeuronID& neuron_id_to, double prob, int num_trys);
    bool FindUnlinkedNeurons(NeuronID& neuron_id_from, NeuronID& neuron_id_to, int num_trys);

public:
    Genome() {}

    Genome(GenomeID id, std::size_t num_inputs, std::size_t num_outputs, Params* params);
    Genome(GenomeID id,
           std::vector<NeuronGene> neurons,
           std::vector<LinkGene> links,
           std::size_t num_inputs,
           std::size_t num_outputs,
           Params* params);

    Genome(const Genome& g)
    {
        m_genome_id = g.m_genome_id;
        m_neuron_genes = g.m_neuron_genes;
        m_link_genes = g.m_link_genes;
        m_fitness = g.m_fitness;
        m_adjusted_fitness = g.m_adjusted_fitness;
        m_amount_to_spawn = g.m_amount_to_spawn;
        m_num_inputs = g.m_num_inputs;
        m_num_outputs = g.m_num_outputs;
        m_species_id = g.m_species_id;
        m_params = g.m_params;
    }

    Genome(Genome&& g)
    {
        m_genome_id = g.m_genome_id;
        m_neuron_genes = g.m_neuron_genes;
        m_link_genes = g.m_link_genes;
        m_fitness = g.m_fitness;
        m_adjusted_fitness = g.m_adjusted_fitness;
        m_amount_to_spawn = g.m_amount_to_spawn;
        m_num_inputs = g.m_num_inputs;
        m_num_outputs = g.m_num_outputs;
        m_species_id = g.m_species_id;
        m_params = g.m_params;
    }

    Genome& operator=(const Genome& g) = default;
    Genome& operator=(Genome&& g) = default;

    bool AddLink(double mutation_prob,
                 double recurrent_prob,
                 InnovationDB& innovation,
                 int num_trys_recurrent,
                 int num_trys_add_link);

    void ConnectNeurons(NeuronID neuron_from, NeuronID neuron_to, InnovationDB& inno_db);

    bool AddNeuron(double mutation_prob,
                   InnovationDB& innovation,
                   int num_trys_to_find_old_link);

    void AddNeuronToLink(LinkGene& link, InnovationDB& inno_db);

    void MutateWeights(double mutation_prob,
                       double prob_new_weight,
                       double max_perturbation);

    void MutateActivationResponse(double mutation_prob,
                                  double max_perturbation);

    double CalculateDifferenceScore(const Genome& other) const;

    Genome Crossover(const Genome& other, const InnovationDB& inno_db, GenomeID genome_id);

    LinkGene* FindLinkConnectingNeurons(NeuronID from, NeuronID to);

    void SortLinks();

    // overload '<' operator for sorting by fitness - fittest to weakest
    friend bool operator<(const Genome& lhs, const Genome& rhs)
    {
        return lhs.m_fitness > rhs.m_fitness;
    }

    //============================accessor methods============================
    const std::vector<NeuronGene>& NeuronGenes() const
    {
        return m_neuron_genes;
    }

    inline std::size_t NumNeurons() const
    {
        return m_neuron_genes.size();
    }

    inline std::size_t NumHiddenNeurons() const
    {
        // don't count input, output and bias neurons
        return m_neuron_genes.size() - m_num_inputs - m_num_outputs - 1;
    }

    inline std::size_t NumLinks() const
    {
        return m_link_genes.size();
    }

    const std::vector<LinkGene>& NeuronLinks() const
    {
        return m_link_genes;
    }

    GenomeID ID() const { return m_genome_id; }
    void SetID(const GenomeID id) { m_genome_id = id; }

    double Fitness() const { return m_fitness; }
    void SetFitness(double fitness) { m_fitness = fitness; }
    void SetAjustedFitness(double adjusted_fitness) { m_adjusted_fitness = adjusted_fitness; }
    double GetAdjustedFitness() const { return m_adjusted_fitness; }

    double AmountToSpawn() const { return m_amount_to_spawn; }
    void SetAmountToSpawn(double to_spawn) { m_amount_to_spawn = to_spawn; }

    void SetSpeciesID(SpeciesID id) { m_species_id = id; }
    SpeciesID GetSpeciesID() const { return m_species_id; }
};


std::string to_string(const Genome& g);

}

#endif
