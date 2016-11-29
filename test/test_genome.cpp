#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <iostream>
#include <vector>

#include "cpplinq.hpp"

#include "genome.h"

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

            THEN("It contains correct number of links")
            {
                std::size_t num_links = g.NumLinks();
                REQUIRE(num_links == ((inputs+1) * outputs));
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

    GIVEN("Genome ID, num inputs and outputs")
    {
        neat::GenomeID ID = 1;
        int inputs = 3;
        int outputs = 2;
        neat::Genome g(ID, inputs, outputs);
        neat::InnovationDB inno_db(g.NumGenes() + 1, g.NumGenes() + g.NumLinks() + 1);

        int num_links = g.NumLinks();
        int num_neurons = g.NumGenes();

        WHEN("Neuron is added")
        {
            THEN("Number of neurons goes up by 1")
            {
                g.AddNeuron(1.0, inno_db, 5);
                REQUIRE(num_neurons + 1 == g.NumGenes());
            }
        }

        WHEN("Link is added")
        {
            THEN("Number of links goes up by 1")
            {
                bool link_added = g.AddLink(1.0, 1.0, inno_db, 5, 5);
                REQUIRE(link_added);
                REQUIRE(g.NumLinks() == num_links + 1);
            }
        }
    }
}

SCENARIO("BLALBLAB", "[genome]")
{
    using namespace cpplinq;
    GIVEN("BLAKJSDF")
    {
        WHEN("KJKJDF")
        {
            THEN("FDKJFDKF")
            {
                typedef int* iptr;
                int v1[] = {1,2,3,4};
                char v2[] = {'a','b','c'};
                iptr v3[1] = {};
                from_array(v1) >> zip_with(from_array(v2)) >> for_each([](const std::pair<int, char>& p)
                {
                    std::cout << "First " << p.first << " and second " << p.second << std::endl;
                });
                int v = 5;
                iptr vptr = &v;
                iptr res = cpplinq::from_array(v2) >> select([&vptr](char c)
                    {
                        if(c == 'd')
                            return vptr;
                        else
                            return static_cast<iptr>(nullptr);
                    })
                    >> first_or_default([](iptr i) { return i != nullptr; });
                if(res)
                    std::cout << "Unexpected " << res << std::endl;
                else
                    std::cout << "Expected!" << std::endl;
            }
        }
    }
}
