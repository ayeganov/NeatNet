#include <cassert>
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

    for(int i = 0; i < inputs + 1; ++i)
    {
        for(int j = inputs + 1; j < inputs + outputs + 1; ++j)
        {
            int innovation_id = inputs + outputs + NumGenes();
            m_link_genes.push_back(LinkGene(m_neuron_genes[i].ID,
                                            m_neuron_genes[j].ID,
                                            m_random.RandomClamped(-1.0, 1.0),
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
                     InnovationDB& innovationDB,
                     int num_trys_recurrent,
                     int num_trys_add_link)
{
    if(m_random.RandomClamped(0.0, 1.0) > mutation_prob)
    {
        return;
    }

    int neuron_id_from = -1;
    int neuron_id_to = -1;

    // if unable to find any suitable neurons then return
    if(!FindNonRecurrentNeuron(neuron_id_from, neuron_id_to, recurrent_prob, num_trys_recurrent) &&
       !FindUnlinkedNeurons(neuron_id_from, neuron_id_to, num_trys_recurrent))
    {
        std::cout << "Could not find any suitable neurons to link" << std::endl;
        return;
    }
    else
    {
        assert(neuron_id_from >=0 && neuron_id_to >= 0);

        auto is_recurrent_link = [&](int neuron_id1, int neuron_id2) {
            auto& neuron1 = m_neuron_genes[GetNeuronIndex(neuron_id1)];
            auto& neuron2 = m_neuron_genes[GetNeuronIndex(neuron_id2)];
            return neuron1.SplitY > neuron2.SplitY;
        };
        int innovation_id = innovationDB.GetInnovationId(neuron_id_from, neuron_id_to, InnovationType::NEW_LINK);
        bool is_recurrent = is_recurrent_link(neuron_id_from, neuron_id_to);

        innovation_id = innovation_id < 0
            ? innovationDB.AddNewInnovation(neuron_id_from, neuron_id_to, InnovationType::NEW_LINK)
            : innovation_id;

        LinkGene new_link(neuron_id_from,
                         neuron_id_to,
                         m_random.RandomClamped<double>(),
                         true,
                         innovation_id,
                         is_recurrent);
        m_link_genes.push_back(new_link);
    }
}

bool Genome::FindNonRecurrentNeuron(int& neuron_id_from, int& neuron_id_to, double prob, int num_trys)
{
    bool result = false;
    auto is_acceptable = [&](NeuronGene& neuron)
    {
        return !neuron.IsRecurrent
               && neuron.Type != NeuronType::BIAS
               && neuron.Type != NeuronType::INPUT;
    };

    if(m_random.RandomDouble() < prob)
    {
        while(num_trys--)
        {
            // get random neuron
            int neuron_idx = m_random.RandomClamped(m_num_inputs+1, m_neuron_genes.size()-1);
            auto& neuron = m_neuron_genes[neuron_idx];

            if(is_acceptable(neuron))
            {
                neuron_id_from = neuron_id_to = neuron.ID;
                neuron.IsRecurrent = true;
                result = true;
                break;
            }
        }
    }
    return result;
}

bool Genome::FindUnlinkedNeurons(int& neuron_id_from, int& neuron_id_to, int num_trys)
{
    bool result = false;
    auto is_acceptable = [&](NeuronGene& neuron_from, NeuronGene& neuron_to)
    {
        return !IsExistingLink(neuron_from.ID, neuron_to.ID)
               && neuron_from.ID != neuron_to.ID
               && neuron_to.Type != NeuronType::BIAS
               && neuron_to.Type != NeuronType::INPUT;
    };

    while(num_trys--)
    {
        int from_idx = m_random.RandomClamped((std::size_t)0, m_neuron_genes.size()-1);
        int to_idx = m_random.RandomClamped(m_num_inputs+1, m_neuron_genes.size()-1);

        auto& neuron_from = m_neuron_genes[from_idx];
        auto& neuron_to = m_neuron_genes[to_idx];

        if(is_acceptable(neuron_from, neuron_to))
        {
            neuron_id_from = neuron_from.ID;
            neuron_id_to = neuron_to.ID;
            result = true;
            break;
        }
    }

    return result;
}

bool Genome::IsExistingLink(int neuron_from_id, int neuron_to_id)
{
    return cpplinq::from(m_link_genes)
        >> cpplinq::any([&](const LinkGene& link)
        {
            return link.FromNeuronID == neuron_from_id
                && link.ToNeuronID == neuron_to_id;
        });
}
