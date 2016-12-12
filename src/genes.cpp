#include "genes.h"

namespace neat
{

std::string to_string(const NeuronGene& ng)
{
    using std::to_string;
    std::string result = to_string(ng.Type) + " ID: " + to_string(ng.ID) + " Recur: " + to_string(ng.IsRecurrent);
    return result;
}


std::string to_string(const LinkGene& lg)
{
    using std::to_string;
    std::string result = "From: " + to_string(lg.FromNeuronID)
                         + " To: " + to_string(lg.ToNeuronID)
                         + " W: " + to_string(lg.Weight)
                         + " Enabled: " + to_string(lg.IsEnabled)
                         + " Recur: " + to_string(lg.IsRecurrent)
                         + " InnovID: " + to_string(lg.InnovID);
    return result;
}


std::string to_string(const NeuronType& nt)
{
    switch(nt)
    {
        case NeuronType::INPUT:
            return "INPUT";
        case NeuronType::HIDDEN:
            return "HIDDEN";
        case NeuronType::OUTPUT:
            return "OUTPUT";
        case NeuronType::BIAS:
            return "BIAS";
        case NeuronType::NONE:
            return "NONE";
        default:
            return "UNKNOWN";
    }
}

};
