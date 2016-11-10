#define CATCH_CONFIG_MAIN
#include "../include/catch.hpp"

#include <iostream>
#include <vector>

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
            neat::Genome g(ID, inputs, outputs);

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

    GIVEN("BLKAJDF")
    {
        std::vector<int> v{1,2,3};
        WHEN("BLKJAKLSDFJ")
        {
            int result = cpplinq::from(v)
            >> cpplinq::select([](int i)
                {
                std::cout << "i = " << i << std::endl;
                    return i * i;
                })
            >> cpplinq::first_or_default([](int val) { return val > 2; });
            std::cout << "And result is " << result << std::endl;
        }
    }
}
