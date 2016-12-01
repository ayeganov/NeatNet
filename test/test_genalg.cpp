#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "cpplinq.hpp"

#include <vector>
#include <iostream>
#include <limits>

#include "genalg.h"


double xor_fitness(neat::SNeuralNetPtr brain)
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

    return std::pow(4 - error, 2);
}

template <typename T>
bool almost_equal(T x, T y)
{
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    const double epsilon = 0.01;
    auto diff = std::fabs(x - y);
    return diff < epsilon;
}

SCENARIO("GenAlg gets initialized with population size of 40 and 2 inputs and 1 output.", "[genalg]")
{
    GIVEN("A task of learning a XOR function")
    {
        neat::GenAlg ga(40, 2, 1);
        auto brains = ga.CreateNeuralNetworks();
        bool solved = false;

        int generations = 100;
        Utils::RunningStat rs;
        for(int gen = 0; gen < generations; ++gen)
        {
            if(gen > 0)
            {
                neat::SNeuralNetPtr best_brain = ga.BestNN();

                if(best_brain)
                {
                    neat::Genome best = ga.BestGenome();
                    auto z_z = best_brain->Update( {0, 0}, neat::UPDATE_TYPE::ACTIVE)[0];
                    auto z_o = best_brain->Update( {0, 1}, neat::UPDATE_TYPE::ACTIVE)[0];
                    auto o_z = best_brain->Update( {1, 0}, neat::UPDATE_TYPE::ACTIVE)[0];
                    auto o_o = best_brain->Update( {1, 1}, neat::UPDATE_TYPE::ACTIVE)[0];

                    std::cout << "Best brain fitness: " << best.Fitness() << std::endl;
                    std::cout << "0 0: " << z_z << std::endl;
                    std::cout << "0 1: " << z_o << std::endl;
                    std::cout << "1 0: " << o_z << std::endl;
                    std::cout << "1 1: " << o_o << std::endl;

                    std::cout << "Best brain score: " << xor_fitness(best_brain) << std::endl;;

                    if(almost_equal(z_z, 0.0) && almost_equal(z_o, 1.0) && almost_equal(o_z, 1.0) && almost_equal(o_o, 0.0))
                    {
                        solved = true;
                        break;
                    }
                }
            }
            std::vector<double> fitnesses;

            for(auto brain : brains)
            {
                double fitness = xor_fitness(brain);
                rs.Push(fitness);
                fitnesses.push_back(fitness);
            }

            brains = ga.Epoch(fitnesses);
            const Utils::RunningStat& species_stat = ga.SpeciesStats();
            auto& links_stat = ga.GenomeLinksStats();
            auto& neurons_stat = ga.GenomeNeuronStats();

            std::cout << "Generation: " << gen << " best ever fitness: " << ga.BestEverFitness() << std::endl;
//            std::cout << "Average fitness: " << rs.Mean() << ", STD: " << rs.StandardDeviation() << std::endl;
            std::cout << "Max fitness: " << rs.MaxValue() << ", Min: " << rs.MinValue() << std::endl;
            std::cout << "Mean species: " << species_stat.Mean() << ", STD: " << species_stat.StandardDeviation()
                << ", Max: " << species_stat.MaxValue() << ", Min: " << species_stat.MinValue() << std::endl;
//            std::cout << "Mean links: " << links_stat.Mean() << ", STD: " << links_stat.StandardDeviation()
//                << ", Min: " << links_stat.MinValue() << ", Max: " << links_stat.MaxValue() << std::endl;
//            std::cout << "Mean neurons: " << neurons_stat.Mean() << ", STD: " << neurons_stat.StandardDeviation()
//                << ", Min: " << neurons_stat.MinValue() << ", Max: " << neurons_stat.MaxValue() << std::endl;

            std::cout << std::endl;
        }
        if(solved)
        {
            std::cout << "Found solution within " << ga.Generation() << " generations." << std::endl;
        }
    }
}
