#include <cmath>
#include <iostream>

#include "cpplinq.hpp"

#include "phenotype.h"

namespace neat
{

double sigmoid(double input, double act_response)
{
    return (1.0 / (1.0 + std::exp(-input / act_response)));
}


NeuralNet::~NeuralNet()
{
    std::cout << "Don't forget to clean up the network." << std::endl;
}


std::vector<double> NeuralNet::Update(const std::vector<double>& inputs, const UPDATE_TYPE update_type)
{
    std::vector<double> outputs;

    int flush_count = update_type == UPDATE_TYPE::SNAPSHOT ? m_net_depth : 1;

    for(int i = 0; i < flush_count; ++i)
    {
        outputs.clear();
        int neuron_idx = 0;

        while(m_neurons[neuron_idx]->Type == NeuronType::INPUT)
        {
            m_neurons[neuron_idx]->OutputSignal = inputs[neuron_idx];
            ++neuron_idx;
        }

        // set the bias neuron output to 1
        m_neurons[neuron_idx++]->OutputSignal = 1;

        // step through the network a neuron at a time
        while (neuron_idx < m_neurons.size())
        {
            //sum this neuron's inputs by iterating through all the links into
            //the neuron
            double sum = cpplinq::from(m_neurons[neuron_idx]->InLinks)
                >> cpplinq::select([](const Link& link)
                    {
                        return link.Weight * link.In->OutputSignal;
                    })
                >> cpplinq::sum();

            //now put the sum through the activation function and assign the
            //value to this neuron's output
            m_neurons[neuron_idx]->OutputSignal =
                sigmoid(sum, m_neurons[neuron_idx]->ActivationResponse);

            if (m_neurons[neuron_idx]->Type == NeuronType::OUTPUT)
            {
                //add to our outputs
                outputs.push_back(m_neurons[neuron_idx]->OutputSignal);
            }

            //next neuron
            ++neuron_idx;
        }
    }

    if(update_type == UPDATE_TYPE::SNAPSHOT)
    {
        for(auto& neuron : m_neurons)
        {
            neuron->OutputSignal = 0.0;
        }
    }
    return outputs;
}

};
