#ifndef __GENALG_H__
#define __GENALG_H__

#include <vector>

#include "genes.h"
#include "genome.h"
#include "species.h"
#include "innovation.h"

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
    std::vector<Genome*> m_best_genomes;
    std::vector<Species> m_species;
    double m_best_ever_fitness;


public:
    GenAlg(std::size_t pop_size,
           std::size_t num_inputs,
           std::size_t num_outputs);

};



};
#endif
