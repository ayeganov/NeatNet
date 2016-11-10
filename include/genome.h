#ifndef __GENOME_H__
#define __GENOME_H__

/**
 * Author: Aleksandr Yeganov
 *
 * Definition of a genome used in the NEAT algorithm.
 */

#include <vector>

#include "genes.h"
#include "innovation.h"
#include "utils.h"


namespace neat
{

class Genome
{
private:
    int m_genome_id;
    std::vector<NeuronGene> m_neuron_genes;
    std::vector<LinkGene> m_link_genes;

    double m_fitness;
    double m_adjusted_fitness;
    double m_amount_to_spawn;

    std::size_t m_num_inputs;
    std::size_t m_num_outputs;

    int m_species_id;

    Utils::Random<std::knuth_b> m_random;


    /**
     * Returns true if the given link is already part of the genome
     * @param neuron_from_id - from neuron id aka neuron providing input
     * @param neuron_to_id - neuron that will receive the output
     */
    bool IsExistingLink(int neuron_from_id, int neuron_to_id);

    /**
     * Given neuron id finds the index of said neuron in the m_neuron_genes
     * vector.
     * @param neuron_id - id of the neuron to find
     */
    int GetNeuronIndex(int neuron_id);

    bool FindNonRecurrentNeuron(int& neuron_id_from, int& neuron_id_to, double prob, int num_trys);
    bool FindUnlinkedNeurons(int& neuron_id_from, int& neuron_id_to, int num_trys);

public:
    Genome();

    Genome(int id, std::size_t num_inputs, std::size_t num_outputs);
    Genome(int id,
           std::vector<NeuronGene> neurons,
           std::vector<LinkGene> links,
           std::size_t num_inputs,
           std::size_t num_outputs);


    ~Genome();

    Genome(const Genome& g);
    Genome& operator=(const Genome& g);

    void AddLink(double mutation_prob,
                 double recurrent_prob,
                 InnovationDB& innovation,
                 int num_trys_recurrent,
                 int num_trys_add_link);

    void AddNeuron(double mutation_prob,
                   InnovationDB& innovation,
                   int num_trys_to_find_old_link);

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

    std::size_t NumGenes() const
    {
        return m_neuron_genes.size();
    }

    const std::vector<LinkGene>& NeuronLinks() const
    {
        return m_link_genes;
    }
};

}

#endif
