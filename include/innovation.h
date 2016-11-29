#ifndef __INNOVATION_H__
#define __INNOVATION_H__

#include <vector>
#include <string>

#include "genes.h"


namespace neat
{

enum class InnovationType
{
    NEW_NEURON,
    NEW_LINK,
    NONE
};


struct Innovation
{
    InnovationType Type;
    InnovationID ID;
    NeuronID NeuronFromID;
    NeuronID NeuronToID;
    NeuronID NewNeuronID;
    NeuronType Neuron_Type;
    double SplitX, SplitY;

    Innovation() : Type(InnovationType::NONE),
                   ID(-1),
                   NeuronFromID(-1),
                   NeuronToID(-1),
                   NewNeuronID(-1),
                   Neuron_Type(NeuronType::NONE)
    {
    }

    Innovation(const NeuronGene& ng,
               InnovationID id): Type(InnovationType::NEW_NEURON),
                                 ID(id),
                                 NeuronFromID(-1),
                                 NeuronToID(-1),
                                 NewNeuronID(ng.ID),
                                 Neuron_Type(ng.Type),
                                 SplitX(ng.SplitX),
                                 SplitY(ng.SplitY)
    {}

    Innovation(const LinkGene& lg,
               InnovationID id): Type(InnovationType::NEW_LINK),
                                 ID(id),
                                 NeuronFromID(lg.FromNeuronID),
                                 NeuronToID(lg.ToNeuronID),
                                 NewNeuronID(-1),
                                 Neuron_Type(NeuronType::NONE)
    {}


    Innovation(InnovationType type,
               InnovationID id,
               NeuronID neuron_from_id,
               NeuronID neuron_to_id) : Type(type),
                                   ID(id),
                                   NeuronFromID(neuron_from_id),
                                   NeuronToID(neuron_to_id)
    {}
};


std::string to_string(InnovationType type);
std::string to_string(const Innovation& innov);


class InnovationDB
{
private:
    std::vector<Innovation> m_innovations;
    int m_next_neuron_id;
    int m_next_innovation_id;

public:
    InnovationDB(const std::vector<NeuronGene>& start_neuron_genes,
                 const std::vector<LinkGene>& start_link_genes);

    InnovationDB(int next_neuron_id, int next_inno_id)
        : m_next_neuron_id(next_neuron_id),
          m_next_innovation_id(next_inno_id)
    {}

    InnovationDB()
        : m_innovations(),
          m_next_neuron_id(1),
          m_next_innovation_id(1)
    {}

    InnovationID GetInnovationId(NeuronID neuron_id_from, NeuronID neuron_id_to, InnovationType type);
    InnovationID AddLinkInnovation(NeuronID neuron_id_from, NeuronID neuron_id_to);
    NeuronID AddNeuronInnovation(NeuronID neuron_id1,
                                 NeuronID neuron_id2,
                                 NeuronType neuron_type,
                                 double width,
                                 double depth);

    NeuronID GetNeuronID(InnovationID innovation_id) const;

    NeuronGene CloneNeuronFromID(NeuronID id) const;

    const std::vector<Innovation>& Innovations() const
    {
        return m_innovations;
    }

    InnovationID NextInnovationID() { return m_next_innovation_id++; }
};
}
#endif
