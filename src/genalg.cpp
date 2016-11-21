#include "cpplinq.hpp"

#include "genalg.h"

namespace neat
{

/**
 * IMPORTANT: Initial population of genomes is structurally identical - the
 *            only thing that is different is the weights. As a consequence the
 *            innovation database gets created from a temporary genome to
 *            record all connections that were used in the original populaiton.
 */
GenAlg::GenAlg(std::size_t pop_size,
               std::size_t num_inputs,
               std::size_t num_outputs)
{
    m_genomes = cpplinq::range(0, pop_size) >> cpplinq::select(
        [&](int _)
            {
                return Genome(m_next_genome_id++, num_inputs, num_outputs);
            })
        >> cpplinq::to_vector();
    Genome tmp(1, num_inputs, num_outputs);
    m_inno_db = InnovationDB(tmp.NeuronGenes(), tmp.NeuronLinks());
}











};
