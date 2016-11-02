#ifndef __GENOME_H__
#define __GENOME_H__

/**
 * Author: Aleksandr Yeganov
 *
 * Definition of a genome used in the NEAT algorithm.
 */

#include <vector>

#include "genes.h"

class Innovation;

class Genome
{
private:
    int m_genome_id;
    std::vector<NeuronGene> m_neuron_genes;
    std::vector<LinkGene> m_link_genes;

    double m_fitness;
    double m_adjusted_fitness;
    double m_amount_to_spawn;

    int m_num_inputs;
    int m_num_outputs;

    int m_species_id;

    /**
     * Returns true if the given link is already part of the genome
     * @param neuron_from - input neuron
     * @param neuron_to - neuron that will receive the output
     */
    bool IsDuplicateLink(int neuron_from, int neuron_to);

    /**
     * Given neuron id finds the index of said neuron in the m_neuron_genes
     * vector.
     * @param neuron_id - id of the neuron to find
     */
    int GetNeuronIndex(int neuron_id);

public:
    Genome();

    Genome(int id, int num_inputs, int num_outputs);
    Genome(int id,
           std::vector<NeuronGene> neurons,
           std::vector<LinkGene> links,
           int num_inputs,
           int num_outputs);


    ~Genome();

    Genome(const Genome& g);
    Genome& operator=(const Genome& g);

    void AddLink(double mutation_prob,
                 double recurrent_prob,
                 Innovation& innovation);

    void AddNeuron(double mutation_prob,
                   Innovation& innovation);

    void MutateWeights(double mutation_prob,
                       double prob_new_weight,
                       double max_perturbation);

    void MutateActivationResponse(double mutation_prob,
                                  double max_perturbation);

    double CalculateCompatabilityScore(const Genome& other);

    const std::vector<NeuronGene>& NeuronGenes() const
    {
        return m_neuron_genes;
    }

};


#endif
