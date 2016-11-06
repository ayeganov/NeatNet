#ifndef __INNOVATION_H__
#define __INNOVATION_H__

#include <vector>

#include "genes.h"

enum class InnovationType
{
    NEW_NEURON,
    NEW_LINK
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

};

class InnovationDB
{
private:
    std::vector<Innovation> m_innovations;
    int m_next_neuron_id;
    int m_next_innovation_id;

public:
    int GetInnovationId(int neuron_id_from, int neuron_id_to, InnovationType type);
    int AddNewInnovation(int neuron_id1, int neuron_id2, InnovationType type);
};
#endif
