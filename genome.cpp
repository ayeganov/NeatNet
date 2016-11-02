#include <iostream>

#include "cpplinq.hpp"

#include "genome.h"

Genome::Genome(int id, int inputs, int outputs):m_genome_id(id),
                                               m_fitness(0),
                                               m_adjusted_fitness(0),
                                               m_num_inputs(inputs),
                                               m_num_outputs(outputs),
                                               m_amount_to_spawn(0),
                                               m_species_id(0)
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

}

Genome::~Genome()
{
}



int main(int argc, char* argv[])
{
    Genome g(1, 3, 2);
    auto neurons = g.NeuronGenes();
    cpplinq::from(neurons)
        >> cpplinq::for_each([](NeuronGene ng) { std::cout << ng.ID << ", " << ng.SplitX << std::endl; });
    return 0;
}
