#include <cmath>

#include "cpplinq.hpp"

#include "phenotype.h"

namespace neat
{

double sigmoid(double input, double act_response)
{
    return (1.0 / (1.0 + std::exp(-input / act_response)));
}


nlohmann::json Link::serialize() const
{
    nlohmann::json object = {
        {"InputID", (int)In->ID},
        {"OutputID", (int)Out->ID},
        {"Weight", Weight},
        {"IsRecurrent", IsRecurrent}
    };
    return std::move(object);
}


nlohmann::json Neuron::serialize() const
{
    using namespace cpplinq;
    nlohmann::json object = {
        {"Type", to_string(Type)},
        {"ID", (int)ID},
        {"ActivationResponse", ActivationResponse},
        {"SplitX", SplitX},
        {"SplitY", SplitY}
    };
    auto in_links = from(InLinks)
        >> select([](const Link& link)
        {
            return link.serialize();
        })
        >> to_vector();

    auto out_links = from(OutLinks)
        >> select([](const Link& link)
        {
            return link.serialize();
        })
        >> to_vector();

    object["InLinks"] = in_links;
    object["OutLinks"] = out_links;
    return std::move(object);
}



NeuralNet::NeuralNet(const Genome& g) : NeuralNet(g.NeuronGenes(),
                                                  g.NeuronLinks())
{}


NeuralNet::NeuralNet(nlohmann::json& object)
{
    using namespace cpplinq;
    auto type_to_enum = [](std::string&& type)
    {
        if(type == "BIAS")
        {
            return NeuronType::BIAS;
        }
        else if(type == "HIDDEN")
        {
            return NeuronType::HIDDEN;
        }
        else if(type == "INPUT")
        {
            return NeuronType::INPUT;
        }
        else if(type == "OUTPUT")
        {
            return NeuronType::OUTPUT;
        }
        else
        {
            return NeuronType::NONE;
        }
    };

    auto get_neuron_ptr = [this](NeuronID id)
    {
        for(auto& np : this->m_neurons)
        {
            if(np.ID == id)
                return &np;
        }
        return static_cast<Neuron*>(nullptr);
    };

    for(auto j : object)
    {
        NeuronType type = type_to_enum(j["Type"]);
        NeuronID id = (int)j["ID"];
        double ar = j["ActivationResponse"];
        double splitx = j["SplitX"];
        double splity = j["SplitY"];

        m_neurons.push_back(Neuron(type, id, ar, splitx, splity));
    }

    auto make_link = [&get_neuron_ptr](nlohmann::json& obj)
    {
        NeuronID input_id = (int)obj["InputID"];
        NeuronID output_id = (int)obj["OutputID"];
        double weight = obj["Weight"];
        bool is_recurrent = obj["IsRecurrent"];
        return Link(get_neuron_ptr(input_id), get_neuron_ptr(output_id), weight, is_recurrent);
    };

    // go over all links and restore them
    for(auto j : object)
    {
        NeuronID id = (int)j["ID"];
        nlohmann::json in_links = j["InLinks"];
        nlohmann::json out_links = j["OutLinks"];
        Neuron* n = get_neuron_ptr(id);

        for(auto in_link : in_links)
        {
            n->InLinks.push_back(make_link(in_link));
        }

        for(auto out_link : out_links)
        {
            n->OutLinks.push_back(make_link(out_link));
        }
    }
}


Neuron* NeuralNet::find_neuron_by_id(NeuronID id)
{
    return find_helper(id, 0, m_neurons.size() - 1);
}


Neuron* NeuralNet::find_helper(NeuronID id, int low, int high)
{
    // base case
    if(high < low) return nullptr;

    int middle = low + (high - low) / 2;

    Neuron* n = &m_neurons[middle];
    if(n->ID == id)
    {
        return n;
    }
    else if(id < n->ID)
    {
        return find_helper(id, low, middle - 1);
    }
    else
    {
        return find_helper(id, middle + 1, high);
    }
}


NeuralNet::NeuralNet(const std::vector<NeuronGene>& neuron_genes,
                     const std::vector<LinkGene>& link_genes) : m_neurons()
{
    //first, create all the required neurons
    for(const auto& ng : neuron_genes)
    {
        m_neurons.push_back(Neuron(ng.Type, ng.ID, ng.ActivationResponse, ng.SplitX, ng.SplitY));
    }

    //now create the links.
    for(const auto& link_gene : link_genes)
    {
        if(link_gene.IsEnabled)
        {
            auto from_neuron = find_neuron_by_id(link_gene.FromNeuronID);
            auto to_neuron = find_neuron_by_id(link_gene.ToNeuronID);

            assert(from_neuron && to_neuron);

            Link tmp_link(from_neuron, to_neuron, link_gene.Weight, link_gene.IsRecurrent);

            from_neuron->OutLinks.push_back(tmp_link);
            to_neuron->InLinks.push_back(tmp_link);
        }
    }
}


std::vector<double> NeuralNet::Update(const std::vector<double>& inputs, const UPDATE_TYPE update_type)
{
    std::vector<double> outputs;

    int flush_count = 1;

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
    return std::move(outputs);
}


nlohmann::json NeuralNet::serialize() const
{
    using namespace cpplinq;
    auto neurons = from(m_neurons)
        >> select([](const Neuron& n)
        {
            return n.serialize();
        })
        >> to_vector();
    return nlohmann::json(neurons);
}


std::string to_string(const NeuralNet& nn)
{
    using std::to_string;
    std::string result;
    for(auto& neuron : nn.m_neurons)
    {
        result += to_string(neuron.Type) + to_string(neuron.ID) +  " ";
    }
    return result;
}


// TODO: Check number of neurons == (input_size + output_size + bias) then return 1
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
