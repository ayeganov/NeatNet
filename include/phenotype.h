#ifndef __PHENOTYPE_H__
#define __PHENOTYPE_H__

#include <vector>

namespace neat
{

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
    std::vector<Link> InLinks;
    std::vector<Link> OutLinks;
};

};
#endif
