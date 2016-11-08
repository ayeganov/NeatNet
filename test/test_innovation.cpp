#define CATCH_CONFIG_MAIN
#include "../include/catch.hpp"

#include <iostream>

#include "../include/cpplinq.hpp"

#include "../include/innovation.h"

SCENARIO("InnovationDB AddNewInnovation [innovation]")
{
    GIVEN("InnovationDB instance")
    {
        neat::InnovationDB inno_db;

        WHEN("Genome is created")
        {
            std::vector<neat::Innovation> innovations;
            auto result = cpplinq::from(innovations)
                >> cpplinq::first_or_default();
            std::cout << to_string(result) << std::endl;
        }

//            Genome g(ID, inputs, outputs);
//
//            THEN("It contains correct number of neurons")
//            {
//                auto neurons = g.NeuronGenes();
//                // add 1 for bias neuron
//                REQUIRE(neurons.size() == inputs + outputs + 1);
//            }
//
//            THEN("Neuron weights never exceed -1 and 1 range")
//            {
//                auto links = g.NeuronLinks();
//                for(auto link : links)
//                {
//                    REQUIRE(link.Weight >= -1.0);
//                    REQUIRE(link.Weight <= 1.0);
//                }
//            }
//        }
    }
}
