#ifndef __PHENOTYPE_H__
#define __PHENOTYPE_H__

#include <vector>
#include <memory>

#include "genes.h"
#include "genome.h"

namespace neat
{

enum class UPDATE_TYPE
{
    SNAPSHOT,
    ACTIVE
};


struct Link;
struct Neuron;
class NeuralNet;
typedef std::shared_ptr<NeuralNet> SNeuralNetPtr;

struct Link
{
    Neuron* In;
    Neuron* Out;

    double Weight;

    bool IsRecurrent;
    Link(Neuron* in, Neuron* out, double weight, bool recurrent)
        : In(in),
          Out(out),
          Weight(weight),
          IsRecurrent(recurrent)
    {}
};



struct Neuron
{
    NeuronType Type;
    NeuronID ID;
    std::vector<Link> InLinks;
    std::vector<Link> OutLinks;
    double ActivationResponse;
    double OutputSignal;

    Neuron(NeuronType type,
           NeuronID id,
           double act_response): Type(type),
                                 ID(id),
                                 InLinks(),
                                 OutLinks(),
                                 ActivationResponse(act_response),
                                 OutputSignal(0)
    {}
};


class NeuralNet
{
private:
    std::vector<Neuron> m_neurons;
    std::size_t m_net_depth;

public:
    NeuralNet(std::vector<Neuron> neurons, std::size_t net_depth)
        : m_neurons(neurons),
          m_net_depth(net_depth)
    {}

    NeuralNet(const Genome& g);
    NeuralNet(const std::vector<NeuronGene>& neuron_genes,
              const std::vector<LinkGene>& link_genes,
              std::size_t depth);

    std::vector<double> Update(const std::vector<double>& inputs, const UPDATE_TYPE update_type = UPDATE_TYPE::ACTIVE);

    friend std::string to_string(const NeuralNet& nn);
};


};
#endif
