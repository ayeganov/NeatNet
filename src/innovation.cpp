#include "../include/innovation.h"
#include "../include/cpplinq.hpp"


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

    // TODO: Figure out the first_or_default behavior
    return result.InnovationID;
}
