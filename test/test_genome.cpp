#define CATCH_CONFIG_MAIN
#include "../include/catch.hpp"

#include <iostream>

#include "../include/cpplinq.hpp"

#include "../include/genome.h"

SCENARIO("Genome gets initialized with correct number of neurons", "[genome]")
{
    GIVEN("Genome ID, num inputs and outputs")
    {
        int ID = 1;
        int inputs = 3;
        int outputs = 2;

        WHEN("Genome is created")
        {
            Genome g(ID, inputs, outputs);

            THEN("It contains correct number of neurons")
            {
                auto neurons = g.NeuronGenes();
                // add 1 for bias neuron
                REQUIRE(neurons.size() == inputs + outputs + 1);
            }

            THEN("Neuron weights never exceed -1 and 1 range")
            {
                auto links = g.NeuronLinks();
                for(auto link : links)
                {
                    REQUIRE(link.Weight >= -1.0);
                    REQUIRE(link.Weight <= 1.0);
                }
            }
        }
    }
}
