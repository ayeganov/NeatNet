#include <cmath>

#include "cpplinq.hpp"

#include "phenotype.h"

namespace neat
{

double sigmoid(double input, double act_response)
{
    return (1.0 / (1.0 + std::exp(-4.9 * input)));
}


NeuralNet::NeuralNet(const Genome& g) : NeuralNet(g.NeuronGenes(),
                                                  g.NeuronLinks(),
                                                  1)
{}


NeuralNet::NeuralNet(const std::vector<NeuronGene>& neuron_genes,
                     const std::vector<LinkGene>& link_genes,
                     std::size_t depth): m_neurons(),
                                         m_net_depth(depth)
{
    //first, create all the required neurons
    for(const auto& ng : neuron_genes)
    {
        m_neurons.push_back(Neuron(ng.Type, ng.ID, ng.ActivationResponse, ng.SplitX, ng.SplitY));
    }

    auto get_neuron_ptr = [this](NeuronID id)
    {
        for(auto& np : this->m_neurons)
        {
            if(np.ID == id)
                return &np;
        }
        return static_cast<Neuron*>(nullptr);
    };

    //now create the links.
    for(const auto& link_gene : link_genes)
    {
        if(link_gene.IsEnabled)
        {
            auto from_neuron = get_neuron_ptr(link_gene.FromNeuronID);
            auto to_neuron = get_neuron_ptr(link_gene.ToNeuronID);

            Link tmp_link(from_neuron, to_neuron, link_gene.Weight, link_gene.IsRecurrent);

            from_neuron->OutLinks.push_back(tmp_link);
            to_neuron->InLinks.push_back(tmp_link);
        }
    }
}


std::vector<double> NeuralNet::Update(const std::vector<double>& inputs, const UPDATE_TYPE update_type)
{
    std::vector<double> outputs;

    int flush_count = update_type == UPDATE_TYPE::SNAPSHOT ? m_net_depth : 1;

    for(int i = 0; i < flush_count; ++i)
    {
        outputs.clear();
        int neuron_idx = 0;

        // The expected order of neurons: INPUT .. INPUT, BIAS, HIDDEN .. HIDDEN
        while(m_neurons[neuron_idx].Type == NeuronType::INPUT)
        {
            m_neurons[neuron_idx].OutputSignal = inputs[neuron_idx];
            ++neuron_idx;
        }

        // set the bias neuron output to 1
        m_neurons[neuron_idx++].OutputSignal = 1;

        // step through the network a neuron at a time
        while (neuron_idx < m_neurons.size())
        {
            //sum this neuron's inputs by iterating through all the links into
            //the neuron
            double sum = cpplinq::from(m_neurons[neuron_idx].InLinks)
                >> cpplinq::select([](const Link& link)
                    {
                        return link.Weight * link.In->OutputSignal;
                    })
                >> cpplinq::sum();

            //now put the sum through the activation function and assign the
            //value to this neuron's output
            m_neurons[neuron_idx].OutputSignal =
                sigmoid(sum, m_neurons[neuron_idx].ActivationResponse);

            if (m_neurons[neuron_idx].Type == NeuronType::OUTPUT)
            {
                //add to our outputs
                outputs.push_back(m_neurons[neuron_idx].OutputSignal);
            }

            //next neuron
            ++neuron_idx;
        }
    }

    if(update_type == UPDATE_TYPE::SNAPSHOT)
    {
        for(auto& neuron : m_neurons)
        {
            neuron.OutputSignal = 0.0;
        }
    }
    return outputs;
}


std::string to_string(const NeuralNet& nn)
{
    using std::to_string;
    std::string result;
    for(auto& neuron : nn.m_neurons)
    {

    }
    return result;
}


std::size_t NeuralNet::GetDepth() const
{
    using namespace cpplinq;
    auto start_neurons = from(m_neurons) >> where([](const Neuron& n)
        {
            return n.Type == NeuronType::BIAS || n.Type == NeuronType::INPUT;
        })
    >> to_vector();

    std::size_t final_depth = 0;

    for(auto& n : start_neurons)
    {
        final_depth = std::max(final_depth, GetDepth(&n, 1));
    }
    return final_depth;
}


//=================================PRIVATE METHODS===============================
std::size_t NeuralNet::GetDepth(const Neuron* n, std::size_t depth) const
{
    if(n->Type == NeuronType::OUTPUT) return depth;

    std::size_t final_depth = 0;
    for(auto& link : n->OutLinks)
    {
        if(link.IsRecurrent) continue;
        final_depth = std::max(final_depth, GetDepth(link.Out, depth + 1));
    }
    return final_depth;
}


};
