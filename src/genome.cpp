#include <iostream>
#include <random>

#include "../include/cpplinq.hpp"

#include "../include/genome.h"

Genome::Genome(int id, std::size_t inputs, std::size_t outputs):m_genome_id(id),
                                               m_fitness(0),
                                               m_adjusted_fitness(0),
                                               m_num_inputs(inputs),
                                               m_num_outputs(outputs),
                                               m_amount_to_spawn(0),
                                               m_species_id(0),
                                               m_random()
{
    double input_row_slice = 1 / (double)(inputs+2);

    m_neuron_genes = cpplinq::range(0, inputs)
        >> cpplinq::select([&](int i){ return NeuronGene(NeuronType::INPUT,
                                                        i,
                                                        0,
                                                        (i+2)*input_row_slice);
                                    })
        >> cpplinq::to_vector();

    m_neuron_genes.push_back(NeuronGene(NeuronType::BIAS, inputs, 0, input_row_slice));

    double output_row_slice = 1 / (double)(outputs+1);

    auto output_neurons = cpplinq::range(1, outputs)
        >> cpplinq::select([&](int i) { return NeuronGene(NeuronType::OUTPUT,
                                                          i + inputs + 1,
                                                          1,
                                                          (i+1)*output_row_slice);
                                      })
        >> cpplinq::to_vector();
    m_neuron_genes.insert(m_neuron_genes.end(), output_neurons.begin(), output_neurons.end());

    double lower_bound = -1.0;
    double upper_bound = 1.0;
    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    std::random_device rand_dev;
    std::shuffle_order_engine<std::minstd_rand0, 256> engine(rand_dev());

    for(int i = 0; i < inputs + 1; ++i)
    {
        for(int j = inputs + 1; j < inputs + outputs + 1; ++j)
        {
            int innovation_id = inputs + outputs + NumGenes();
            m_link_genes.push_back(LinkGene(m_neuron_genes[i].ID,
                                            m_neuron_genes[j].ID,
                                            m_random.RandomClamped<double>(lower_bound, upper_bound),
                                            true,
                                            innovation_id));
        }
    }
}


Genome::Genome(int id,
                std::vector<NeuronGene> neuron_genes,
                std::vector<LinkGene> link_genes,
                std::size_t num_inputs,
                std::size_t num_outputs):m_genome_id(id),
                                         m_neuron_genes(neuron_genes),
                                         m_link_genes(link_genes),
                                         m_fitness(0),
                                         m_adjusted_fitness(0),
                                         m_amount_to_spawn(0),
                                         m_num_inputs(num_inputs),
                                         m_num_outputs(num_outputs),
                                         m_random()
{}
Genome::~Genome()
{
}


int Genome::GetNeuronIndex(int neuron_id)
{
    for(int i = 0; i < m_neuron_genes.size(); ++i)
    {
        if(m_neuron_genes[i].ID == neuron_id)
        {
            return i;
        }
    }
    return -1;
}

void Genome::AddLink(double mutation_prob,
                     double recurrent_prob,
                     Innovation& innovation)
{
    
}
