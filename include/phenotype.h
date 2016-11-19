#ifndef __PHENOTYPE_H__
#define __PHENOTYPE_H__

#include <vector>

#include "genes.h"

namespace neat
{

enum class UPDATE_TYPE
{
    SNAPSHOT,
    ACTIVE
};

struct Neuron;

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
    std::vector<Neuron*> m_neurons;
    std::size_t m_net_depth;

public:
    NeuralNet(std::vector<Neuron*> neurons, std::size_t net_depth)
        : m_neurons(neurons),
          m_net_depth(net_depth)
    {}

    ~NeuralNet();

    std::vector<double> Update(const std::vector<double>& inputs, const UPDATE_TYPE update_type);

};

};
#endif
