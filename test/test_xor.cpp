#include "serialize.h"
#include <iomanip>
#include <ios>
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "cpplinq.hpp"

#include <vector>

#include "genalg.h"
#include "utils.h"


const int NUM_INPUTS = 2;
const int NUM_OUTPUTS = 1;
const double WIN_FITNESS = 15.95;


double xor_fitness(neat::SNeuralNetPtr brain)
{
    auto z_z = brain->Update( {0, 0} )[0];
    auto z_o = brain->Update( {0, 1} )[0];
    auto o_z = brain->Update( {1, 0} )[0];
    auto o_o = brain->Update( {1, 1} )[0];

    double error = 0.0;
    error += std::fabs(1 - z_o);
    error += std::fabs(1 - o_z);
    error += z_z;
    error += o_o;

    return std::pow(4 - error, 2);
}


SCENARIO("GenAlg gets initialized with population size of 40, 2 inputs and 1 output.", "[genalg]")
{
    std::cout << std::setprecision(15) << std::boolalpha << std::fixed;
//    auto& ran = Utils::DefaultRandom::Instance(5);

//    for(int i = 0; i < 5; ++i)
//    {
//      std::cout << i << " = " << ran.RandomDouble() << '\n';
//    }

    GIVEN("A task of learning a XOR function")
    {
        neat::Params p("xor_params.json");
        neat::GenAlg ga(NUM_INPUTS, NUM_OUTPUTS, p);
        auto brains = ga.CreateNeuralNetworks();
        bool solved = false;

        THEN("It learns it well")
        {
            int generations = 1000;
            for(int gen = 0; gen < generations && !solved; ++gen)
            {
                std::vector<double> fitnesses{};
                for(auto brain : brains)
                {
                    double fitness = xor_fitness(brain);
                    fitnesses.push_back(fitness);
                    solved = fitness > WIN_FITNESS;

                    if(solved)
                    {
                      std::cout << to_string(ga.BestGenome()) << std::endl;
                      neat::serialize_to_file("/tmp/xor_winner.json", *brain);

                        auto stats = ga.SpeciesStats();
                        std::cout << "Species size " << "Mean: " << stats.Mean() << ", Max: " << stats.MaxValue() << ", Min: " << stats.MinValue() << std::endl;
                        std::cout << "Generation: " << ga.Generation() << ", Species size: " << ga.GetSpecies().size() << std::endl;
                        goto done;
                    }
                }
                brains = ga.Epoch(fitnesses);
//                std::cout << "best ever fitness: " << ga.BestEverFitness() << "\n";
                auto stats = ga.SpeciesStats();
//                std::cout << "Species size " << "Mean: " << stats.Mean() << ", Max: " << stats.MaxValue() << ", Min: " << stats.MinValue() << std::endl;

//                auto stats = ga.SpeciesStats();
//                std::cout << "Species size " << "Mean: " << stats.Mean() << ", Max: " << stats.MaxValue() << ", Min: " << stats.MinValue() << std::endl;
//                std::cout << "Generation: " << ga.Generation() << ", Species size: " << ga.GetSpecies().size()
//                          << ", Best seen fitness: " << ga.BestEverFitness() << std::endl;
//                if(gen == 100)
//                {
//                  solved = true;
//                  break;
//                }
            }
done:
            REQUIRE(solved);
        }
    }
}
