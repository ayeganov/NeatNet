#ifndef __PHENOTYPE_H__
#define __PHENOTYPE_H__

#include <vector>
#include <memory>

#include "json.hpp"

#include "genes.h"
#include "genome.h"
#include "serialize.h"


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


struct Link : public ISerialize
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

    nlohmann::json serialize() const;
};


struct Neuron : public ISerialize
{
    NeuronType Type;
    NeuronID ID;
    std::vector<Link> InLinks;
    std::vector<Link> OutLinks;
    double ActivationResponse;
    double OutputSignal;
    double SplitX;
    double SplitY;

    Neuron(NeuronType type,
           NeuronID id,
           double act_response,
           double splitx,
           double splity): Type(type),
                                 ID(id),
                                 InLinks(),
                                 OutLinks(),
                                 ActivationResponse(act_response),
                                 OutputSignal(0),
                                 SplitX(splitx),
                                 SplitY(splity)
    {}

    nlohmann::json serialize() const;
};


class NeuralNet : public ISerialize
{
private:
    std::vector<Neuron> m_neurons;

    std::size_t GetDepth(const Neuron* n, std::size_t depth) const;

public:
    NeuralNet(std::vector<Neuron>& neurons)
        : m_neurons(neurons)
    {}

    NeuralNet(const Genome& g);
    NeuralNet(const std::vector<NeuronGene>& neuron_genes,
              const std::vector<LinkGene>& link_genes);


    std::vector<double> Update(const std::vector<double>& inputs, const UPDATE_TYPE update_type = UPDATE_TYPE::ACTIVE);

    nlohmann::json serialize() const;

    // Getters and setters
    std::size_t GetDepth() const;
    const std::vector<Neuron>& GetNeurons() const { return m_neurons; }

    // Friends
    friend std::string to_string(const NeuralNet& nn);
};


};
#endif
