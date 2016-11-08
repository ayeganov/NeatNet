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


int InnovationDB::AddNewInnovation(int neuron_id1, int neuron_id2, InnovationType type)
{
    return 0;
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

}
