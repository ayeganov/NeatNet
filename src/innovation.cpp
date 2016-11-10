#include "../include/innovation.h"
#include "../include/cpplinq.hpp"


namespace neat
{

int InnovationDB::GetInnovationId(int neuron_id_from, int neuron_id_to, InnovationType type)
{
    auto result = cpplinq::from(m_innovations)
        >> cpplinq::where([&](const Innovation& innovation)
        {
            return innovation.NeuronFromID == neuron_id_from
                   && innovation.NeuronToID == neuron_id_to
                   && innovation.Type == type;
        })
        >> cpplinq::first_or_default();

    return result.Type == InnovationType::NONE ? -1 : result.InnovationID;
}


int InnovationDB::AddNewInnovation(int neuron_from_id, int neuron_to_id, InnovationType type)
{
    Innovation new_innovation(type, m_next_innovation_id, neuron_from_id, neuron_to_id);
    if(type == InnovationType::NEW_NEURON)
    {
        new_innovation.NeuronID = m_next_neuron_id;
        ++m_next_neuron_id;
    }
    m_innovations.push_back(new_innovation);

    return m_next_innovation_id++;
}


std::string to_string(InnovationType type)
{
    switch(type)
    {
        case InnovationType::NEW_NEURON:
            return "NEW_NEURON";
        case InnovationType::NEW_LINK:
            return "NEW_LINK";
        default:
            return "NONE";
    }
}


std::string to_string(const Innovation& innov)
{
    using std::to_string;
    return "Type: " + to_string(innov.Type) + ", ID: " + to_string(innov.InnovationID)\
        + ", From: " + to_string(innov.NeuronFromID)\
        + ", To: " + to_string(innov.NeuronToID);
}


int InnovationDB::GetNeuronID(int innovation_id) const
{
    Innovation innovation = cpplinq::from(m_innovations)
        >> cpplinq::first([&](Innovation& inno)
            {
                return inno.InnovationID == innovation_id;
            });
    return innovation.NeuronID;
}

}
