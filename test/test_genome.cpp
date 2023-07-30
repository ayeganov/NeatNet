#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <iostream>
#include <vector>
#include <set>

#include "cpplinq.hpp"
#include "genes.h"
#include "genome.h"
#include "params.h"


SCENARIO("Genome gets initialized with correct number of neurons", "[GenomeInit]")
{
    using namespace cpplinq;
    GIVEN("Genome ID, 3 inputs and 2 outputs")
    {
        int ID = 1;
        int inputs = 8;
        int outputs = 4;

        neat::Params p;
        neat::Genome g(ID, inputs, outputs, &p);

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

            THEN("Link innovation IDs are unique")
            {
                std::set<neat::InnovationID> inno_ids;
                for(auto& link : g.NeuronLinks())
                {
                    auto num_ids = inno_ids.count(link.InnovID);
                    REQUIRE(num_ids == 0);
                    inno_ids.insert(link.InnovID);
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
        neat::Params p;
        neat::Genome g(id, inputs, outputs, &p);

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
                REQUIRE(g.NumLinks() == 5); // +2 links because the old link
                                            // get disabled, and not removed.
                int num_disabled = 0;
                for(auto& link : g.NeuronLinks())
                {
                    num_disabled += int(!link.IsEnabled);
                }
                REQUIRE(num_disabled == 1);
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
                        REQUIRE(lg.ToNeuronID != neat::NeuronID(id.raw_value()));
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
        neat::Params p;
        neat::Genome g(1, 2, 1, &p);

        neat::InnovationDB idb(g.NeuronGenes(), g.NeuronLinks());
        range(0, 3) >> for_each([&g, &idb](int _) { g.AddNeuron(1.0, idb, 5); });

        WHEN("AddLink is called with 0\% mutation probability")
        {
            int num_neurons = g.NumNeurons();
            int num_links = g.NumLinks();
            g.AddLink(0.0, 0.0, idb, 5, 5);

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


SCENARIO("Genomes get crossed over", "[Crossover]")
{
    using namespace cpplinq;
    GIVEN("Two identical genomes")
    {
        neat::Params p;
        neat::Genome mom(1, 5, 2, &p);
        neat::Genome dad(2, 5, 2, &p);

        mom.SetFitness(1.0);
        dad.SetFitness(1.0);

        neat::InnovationDB inno_db(mom.NeuronGenes(), mom.NeuronLinks());

        WHEN("They crossover")
        {
            neat::Genome baby = mom.Crossover(dad, inno_db, 3);

            THEN("The baby is identical to parents")
            {
                REQUIRE(baby.NumLinks() == mom.NumLinks());
                REQUIRE(baby.NumNeurons() == mom.NumNeurons());
                REQUIRE(baby.NumHiddenNeurons() == mom.NumHiddenNeurons());

                for(int link_idx = 0; link_idx < baby.NumLinks(); ++link_idx)
                {
                    auto& baby_link = baby.NeuronLinks()[link_idx];
                    auto& mom_link = mom.NeuronLinks()[link_idx];
                    REQUIRE(baby_link.InnovID == mom_link.InnovID);
                    REQUIRE(baby_link.IsEnabled == mom_link.IsEnabled);
                    REQUIRE(baby_link.IsRecurrent == mom_link.IsRecurrent);
                    REQUIRE(baby_link.ToNeuronID == mom_link.ToNeuronID);
                    REQUIRE(baby_link.FromNeuronID == mom_link.FromNeuronID);
                }

                auto neuron_zip = from(baby.NeuronGenes())
                >> zip_with(from(mom.NeuronGenes()))
                >> to_vector();

                for(auto& pair : neuron_zip)
                {
                    REQUIRE(pair.first.Type == pair.second.Type);
                    REQUIRE(pair.first.ID == pair.second.ID);
                    REQUIRE(pair.first.IsRecurrent == pair.second.IsRecurrent);
                    REQUIRE(pair.first.SplitX == pair.second.SplitX);
                    REQUIRE(pair.first.SplitY == pair.second.SplitY);
                }
            }
        }
    }

    GIVEN("Two genomes, where one inherits from the other")
    {
        neat::Params p;
        neat::Genome dad(1, 2, 1, &p);
        neat::InnovationDB inno_db(dad.NeuronGenes(), dad.NeuronLinks());

        WHEN("Genomes are mutated in the way they are defined in the paper")
        {
            neat::LinkGene* one_to_three = dad.FindLinkConnectingNeurons(1, 3);
            REQUIRE(one_to_three != nullptr);

            dad.AddNeuronToLink(*one_to_three, inno_db);

            neat::Genome mom(2, dad.NeuronGenes(), dad.NeuronLinks(), 2, 1, &p);

            neat::LinkGene* four_to_three = mom.FindLinkConnectingNeurons(4, 3);
            REQUIRE(four_to_three != nullptr);

            mom.AddNeuronToLink(*four_to_three, inno_db);

            dad.ConnectNeurons(0, 4, inno_db);

            mom.ConnectNeurons(2, 4, inno_db);
            mom.ConnectNeurons(0, 5, inno_db);

            // Set fitness to prefer mom's genes over dad's
            dad.SetFitness(1.0);
            mom.SetFitness(1.1);
            auto baby = dad.Crossover(mom, inno_db, 3);

            auto neuron_zip = from(baby.NeuronLinks())
            >> zip_with(from(mom.NeuronLinks()))
            >> to_vector();

            for(auto& p : neuron_zip)
            {
                REQUIRE(p.first.InnovID == p.second.InnovID);
                REQUIRE(p.first.IsRecurrent == p.second.IsRecurrent);
            }

            // Set fitness to prefer dad's genes over mom's
            dad.SetFitness(1.1);
            mom.SetFitness(1.0);

            baby = dad.Crossover(mom, inno_db, 3);

            neuron_zip = from(baby.NeuronLinks())
            >> zip_with(from(dad.NeuronLinks()))
            >> to_vector();

            for(auto& p : neuron_zip)
            {
                REQUIRE(p.first.InnovID == p.second.InnovID);
                REQUIRE(p.first.IsRecurrent == p.second.IsRecurrent);
            }
        }
    }
}


// Add more cases to this scenario
SCENARIO("Compatibility is calculated between genomes", "[CalculateDifferenceScore]")
{
    using namespace cpplinq;
    GIVEN("Two identical genomes")
    {
        neat::Params p;
        neat::Genome g1(1, 2, 1, &p);
        neat::Genome g2(2, g1.NeuronGenes(), g1.NeuronLinks(), 2, 1, &p);

        WHEN("Their compatibility is calculated")
        {
            double score = g1.CalculateDifferenceScore(g2);

            THEN("It is equal to 0")
            {
                REQUIRE(score == 0.0);
            }
        }
    }

    GIVEN("Two different genomes")
    {
        neat::Params p;
        neat::Genome g1(1, 2, 1, &p);
        neat::Genome g2(2, 2, 1, &p);

        WHEN("Their compatibility is calculated")
        {
            double score = g1.CalculateDifferenceScore(g2);

            THEN("It is not equal to 0")
            {
                REQUIRE(score > 0.0);
            }
        }
    }
}
