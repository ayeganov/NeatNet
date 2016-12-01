#ifndef __GENES_H__
#define __GENES_H__

/**
 * Author: Aleksandr Yeganov
 *
 * Neuron and link gene definitions used in the NEAT algorithm. These
 * structures are composed together to create a genome.
 */

#include "utils.h"

namespace neat
{

namespace internal
{
    struct NeuronIDTag {};
    struct GenomeIDTag {};
    struct InnovationIDTag {};
    struct SpeciesIDTag {};
}

typedef Utils::IDType<int, internal::NeuronIDTag> NeuronID;
typedef Utils::IDType<int, internal::GenomeIDTag> GenomeID;
typedef Utils::IDType<int, internal::InnovationIDTag> InnovationID;
typedef Utils::IDType<int, internal::SpeciesIDTag> SpeciesID;


enum class NeuronType
{
    INPUT,
    HIDDEN,
    OUTPUT,
    BIAS,
    NONE
};


std::string to_string(const NeuronType nt);

/**
 * Neuron gene definition
 */
struct NeuronGene
{
    NeuronType Type;
    NeuronID ID;
    bool IsRecurrent;
    double ActivationResponse;
    double SplitY, SplitX;

    NeuronGene(NeuronType type,
               NeuronID id,
               double y,
               double x,
               bool recurrent=false)
    : Type(type),
      ID(id),
      SplitY(y),
      SplitX(x),
      ActivationResponse(1),
      IsRecurrent(recurrent)
    {}
};


/**
 * Definition of link between neurons
 */
struct LinkGene
{
    NeuronID FromNeuronID, ToNeuronID;
    double Weight;
    bool IsEnabled;
    bool IsRecurrent;
    InnovationID InnovID;

    LinkGene()
    {
    }

    LinkGene(NeuronID from,
             NeuronID to,
             double weight,
             bool enabled,
             neat::InnovationID innovation_id,
             bool recurrent=false)
    : FromNeuronID(from),
      ToNeuronID(to),
      Weight(weight),
      IsEnabled(enabled),
      InnovID(innovation_id),
      IsRecurrent(recurrent)
    {}

    /**
     * Overload the global < operator for sorting links by their innovation id
     */
    friend bool operator<(const LinkGene& lhs, const LinkGene& rhs)
    {
        return lhs.InnovID < rhs.InnovID;
    }
};


};

#endif
