#include "phenotype.h"
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "cpplinq.hpp"

#include <vector>

#include "genalg.h"
#include "utils.h"



const int NUM_INPUTS = 2;
const int NUM_OUTPUTS = 1;
const double WIN_FITNESS = 15.99 * 5;


double xor_fitness(neat::SNeuralNetPtr brain)
{
    double score = 0.0;
    for(int i = 0; i < 5; ++i)
    {
      auto z_z = brain->Update( {0, 0}, neat::UPDATE_TYPE::ACTIVE)[0];
      auto z_o = brain->Update( {0, 1}, neat::UPDATE_TYPE::ACTIVE)[0];
      auto o_z = brain->Update( {1, 0}, neat::UPDATE_TYPE::ACTIVE)[0];
      auto o_o = brain->Update( {1, 1}, neat::UPDATE_TYPE::ACTIVE)[0];

      double error = 0.0;
      error += std::fabs(1 - z_o);
      error += std::fabs(1 - o_z);
      error += z_z;
      error += o_o;

      score += std::pow(4 - error, 2);
    }
    return score;
}


SCENARIO("GenAlg gets initialized with population size of 40, 2 inputs and 1 output.", "[genalg]")
{
    GIVEN("A task of learning a XOR function")
    {
        neat::Params p("xor_params.json");
        neat::GenAlg ga(NUM_INPUTS, NUM_OUTPUTS, p);
        auto brains = ga.CreateNeuralNetworks();
        bool solved = false;

        THEN("It learns it well")
        {
            int generations = 1500;
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
                        auto stats = ga.SpeciesStats();
                        std::cout << "Species size " << "Mean: " << stats.Mean() << ", Max: " << stats.MaxValue() << ", Min: " << stats.MinValue() << std::endl;
                        std::cout << "Generation: " << ga.Generation() << ", Species size: " << ga.GetSpecies().size() << std::endl;
                        goto done;
                    }
                }
                brains = ga.Epoch(fitnesses);
                if(gen % 1000 == 0)
                {
                  std::cout << "Fitnesses: " << fitnesses << "\n";
                }
                std::cout << "Best fitness: " << ga.BestEverFitness() << "\n";
            }
done:
            REQUIRE(solved);
        }
    }
}
