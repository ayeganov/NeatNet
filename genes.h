#ifndef __GENES_H__
#define __GENES_H__

/**
 * Author: Aleksandr Yeganov
 *
 * Neuron and link gene definitions used in the NEAT algorithm. These
 * structures are composed together to create a genome.
 */

enum class NeuronType
{
    INPUT,
    HIDDEN,
    OUTPUT,
    BIAS,
    NONE
};


/**
 * Neuron gene definition
 */
struct NeuronGene
{
    NeuronType Type;
    int ID;
    bool IsRecurrent;
    double ActivationResponse;
    double SplitY, SplitX;

    NeuronGene(NeuronType type,
               int id,
               double y,
               double x,
               bool recurrent=false)
    : Type(type),
      ID(id),
      SplitY(y),
      SplitX(x),
      ActivationResponse(1)
    {}
};


/**
 * Definition of link between neurons
 */
struct LinkGene
{
    int FromNeuronID, ToNeuronID;
    double Weight;
    bool IsEnabled;
    bool IsRecurrent;
    int InnovationID;

    LinkGene(int from,
             int to,
             double weight,
             bool enabled,
             int innovation_id,
             bool recurrent=false)
    : FromNeuronID(from),
      ToNeuronID(to),
      Weight(weight),
      IsEnabled(enabled),
      InnovationID(innovation_id),
      IsRecurrent(recurrent)
    {}

    /**
     * Overload the global < operator for sorting links by their innovation id
     */
    friend bool operator<(const LinkGene& lhs, const LinkGene& rhs)
    {
        return lhs.InnovationID < rhs.InnovationID;
    }
};



#endif
