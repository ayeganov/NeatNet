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
    int InnovationID;
    int NeuronFromID;
    int NeuronToID;
    int NeuronID;
    NeuronType Neuron_Type;
    double SplitX, SplitY;

    Innovation() : Type(InnovationType::NONE),
                   InnovationID(-1),
                   NeuronFromID(-1),
                   NeuronToID(-1),
                   NeuronID(-1),
                   Neuron_Type(NeuronType::NONE)
    {
    }


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
    InnovationDB() : m_innovations(),
                     m_next_neuron_id(1),
                     m_next_innovation_id(1)
    {}
    int GetInnovationId(int neuron_id_from, int neuron_id_to, InnovationType type);
    int AddNewInnovation(int neuron_id1, int neuron_id2, InnovationType type);

    const std::vector<Innovation>& Innovations() const
    {
        return m_innovations;
    }
};
}
#endif
