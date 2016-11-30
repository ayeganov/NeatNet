#include "innovation.h"
#include "cpplinq.hpp"


namespace neat
{

InnovationDB::InnovationDB(const std::vector<NeuronGene>& start_neuron_genes,
                            const std::vector<LinkGene>& start_link_genes)
    : InnovationDB()
{
    // first create neuron innovations
    for(auto const& ng : start_neuron_genes)
    {
        Innovation innov(ng, m_next_innovation_id++);
        m_innovations.push_back(innov);
    }

    // ... then create link innovations
    for(auto const& lg : start_link_genes)
    {
        m_innovations.push_back(Innovation(lg, m_next_innovation_id++));
    }
}

InnovationID InnovationDB::GetInnovationId(NeuronID neuron_id_from, NeuronID neuron_id_to, InnovationType type)
{
    auto result = cpplinq::from(m_innovations)
        >> cpplinq::where([&](const Innovation& innovation)
        {
            return innovation.NeuronFromID == neuron_id_from
                   && innovation.NeuronToID == neuron_id_to
                   && innovation.Type == type;
        })
        >> cpplinq::first_or_default();

    return result.Type == InnovationType::NONE ? InnovationID(-1) : result.ID;
}


InnovationID InnovationDB::AddLinkInnovation(NeuronID neuron_from_id, NeuronID neuron_to_id)
{
    Innovation new_innovation(InnovationType::NEW_LINK, m_next_innovation_id, neuron_from_id, neuron_to_id);
    m_innovations.push_back(new_innovation);
    return m_next_innovation_id++;
}


NeuronID InnovationDB::AddNeuronInnovation(NeuronID neuron_from, NeuronID neuron_to, NeuronType neuron_type, double splix_x, double split_y)
{
    Innovation new_innovation(InnovationType::NEW_NEURON, m_next_innovation_id++, neuron_from, neuron_to);
    new_innovation.NewNeuronID = m_next_neuron_id++;
    new_innovation.Neuron_Type = neuron_type;
    new_innovation.SplitX = splix_x;
    new_innovation.SplitY = split_y;
    m_innovations.push_back(new_innovation);
    return new_innovation.NewNeuronID;
}


NeuronGene InnovationDB::CloneNeuronFromID(NeuronID id) const
{
    using namespace cpplinq;
    NeuronGene ng(NeuronType::HIDDEN, 0, 0, 0);

    auto innovation = from(m_innovations) >> first([id](const Innovation& innov)
        {
            return innov.NewNeuronID == id;
        });

    ng.Type = innovation.Neuron_Type;
    ng.SplitX = innovation.SplitX;
    ng.SplitY = innovation.SplitY;
    ng.ID = innovation.NewNeuronID;

    return ng;
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
    return "Type: " + to_string(innov.Type) + ", ID: " + to_string(innov.ID)\
        + ", From: " + to_string(innov.NeuronFromID)\
        + ", To: " + to_string(innov.NeuronToID);
}


NeuronID InnovationDB::GetNeuronID(InnovationID innovation_id) const
{
    Innovation innovation = cpplinq::from(m_innovations)
        >> cpplinq::first([&](const Innovation& inno)
            {
                return inno.ID == innovation_id;
            });
    return innovation.NewNeuronID;
}

}
