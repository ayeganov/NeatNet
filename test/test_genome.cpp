#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <iostream>
#include <vector>

#include "cpplinq.hpp"
#include "genes.h"
#include "genome.h"

SCENARIO("Genome gets initialized with correct number of neurons", "[GenomeInit]")
{
    using namespace cpplinq;
    GIVEN("Genome ID, 3 inputs and 2 outputs")
    {
        int ID = 1;
        int inputs = 3;
        int outputs = 2;

        neat::Genome g(ID, inputs, outputs);

        WHEN("Genome is created")
        {

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

        WHEN("Genome's weights get mutated with 0\% probability and 0\% replacement")
        {
            std::vector<double> orig_weights = from(g.NeuronLinks())
                >> select([](const neat::LinkGene& g) { return g.Weight; } ) >> to_vector();

            g.MutateWeights(0, 0, 0.5);
            THEN("The weights do not change.")
            {
                std::vector<double> new_weights = from(g.NeuronLinks())
                    >> select([](const neat::LinkGene& g) { return g.Weight; } ) >> to_vector();

                from(orig_weights)
                    >> zip_with(from(new_weights))
                    >> for_each([](std::pair<double, double> p)
                    {
                        REQUIRE(p.first == p.second);
                    });
            }
        }

        WHEN("Genome's weights get mutated with 100\% probability and 0\% replacement")
        {
            std::vector<double> orig_weights = from(g.NeuronLinks())
                >> select([](const neat::LinkGene& g) { return g.Weight; } ) >> to_vector();

            g.MutateWeights(1.0, 0, 0.5);
            THEN("The weights DO change.")
            {
                std::vector<double> new_weights = from(g.NeuronLinks())
                    >> select([](const neat::LinkGene& g) { return g.Weight; } ) >> to_vector();

                from(orig_weights)
                    >> zip_with(from(new_weights))
                    >> for_each([](std::pair<double, double> p)
                    {
                        REQUIRE(p.first != p.second);
                    });
            }
        }

        WHEN("Genome's weights get mutated with 100\% probability and 100\% replacement")
        {
            std::vector<double> orig_weights = from(g.NeuronLinks())
                >> select([](const neat::LinkGene& g) { return g.Weight; } ) >> to_vector();

            g.MutateWeights(1.0, 1.0, 0.5);
            THEN("The weights DO change.")
            {
                std::vector<double> new_weights = from(g.NeuronLinks())
                    >> select([](const neat::LinkGene& g) { return g.Weight; } ) >> to_vector();

                from(orig_weights)
                    >> zip_with(from(new_weights))
                    >> for_each([](std::pair<double, double> p)
                    {
                        REQUIRE(p.first != p.second);
                    });
            }
        }
    }
}

SCENARIO("A neuron is added to a genome", "[AddNeuron]")
{
    using namespace cpplinq;
    GIVEN("A single genome with 2 inputs and 1 output")
    {
        neat::GenomeID id = 1;
        std::size_t inputs = 2;
        std::size_t outputs = 1;
        neat::Genome g(id, inputs, outputs);

        REQUIRE(g.NumNeurons() == 4);
        REQUIRE(g.NumLinks() == 3);

        neat::InnovationDB idb(g.NeuronGenes(), g.NeuronLinks());

        WHEN("AddNeuron is called with 0\% probability")
        {
            g.AddNeuron(0, idb, 100);

            THEN("Genome does not gain an extra neuron, and 2 links")
            {
                REQUIRE(g.NumNeurons() == 4);
                REQUIRE(g.NumLinks() == 3);
            }
        }

        WHEN("AddNeuron is called with 100\% probability")
        {
            g.AddNeuron(1, idb, 100);

            THEN("Genome gains an extra neuron, and 2 links")
            {
                REQUIRE(g.NumNeurons() == 5);
                REQUIRE(g.NumLinks() == 5); // 2 links because the old link
                                            // get disabled, and not removed.
            }
        }

        WHEN("AddNeuron is called multiple times with 100\% probability")
        {
            range(0, 5) >> for_each([&g, &idb](int _) { g.AddNeuron(1.0, idb, 100); });

            THEN("No links ever point to the input or bias neurons.")
            {
                auto input_bias_ids = range(0, inputs + 1) >> select([](int id)
                    { return neat::GenomeID(id); }) >> to_vector();

                for(auto& lg : g.NeuronLinks())
                {
                    for(auto id : input_bias_ids)
                    {
                        REQUIRE(lg.ToNeuronID != id);
                    }
                }
            }

            THEN("All neuron ID's follow a natural sequence")
            {
                range(0, g.NumNeurons()) >> zip_with(from(g.NeuronGenes()))
                    >> for_each([](std::pair<int, neat::NeuronGene> p)
                    {
                        REQUIRE(neat::NeuronID(p.first) == p.second.ID);
                    });
            }
        }

        // TODO: Add checks for consistency
        // 2) All NeuronID's must be stored within the innovation db
    }
}

SCENARIO("A link is added to the genome", "[AddLink]")
{
    using namespace cpplinq;
    GIVEN("A single genome with 2 inputs and 1 output and 3 hidden neurons")
    {
        neat::Genome g(1, 2, 1);

        neat::InnovationDB idb(g.NeuronGenes(), g.NeuronLinks());
        range(0, 3) >> for_each([&g, &idb](int _) { g.AddNeuron(1.0, idb, 5); });

        WHEN("AddLink is called with 0\% mutation probability")
        {
            int num_neurons = g.NumNeurons();
            g.AddLink(0.0, 0.0, idb, 5, 5);
            int num_links = g.NumLinks();

            THEN("No links get added.")
            {
                REQUIRE(g.NumLinks() == num_links);
            }
        }

        WHEN("AddLink is called with 100\% mutation and 0\% self recurrent probability")
        {
            int num_links = g.NumLinks();
            bool has_self_recurrent = from(g.NeuronLinks())
                >> any([](const neat::LinkGene& lg)
                {
                    return lg.FromNeuronID == lg.ToNeuronID;
                });

            REQUIRE(!has_self_recurrent);
            REQUIRE(g.AddLink(1.0, 0.0, idb, 100, 100));
            REQUIRE(num_links + 1 == g.NumLinks());

            THEN("No self recurring links get added.")
            {
                has_self_recurrent = from(g.NeuronLinks())
                    >> any([](const neat::LinkGene& lg) { return lg.FromNeuronID == lg.ToNeuronID; });

                REQUIRE(!has_self_recurrent);
            }
        }
    }
}
