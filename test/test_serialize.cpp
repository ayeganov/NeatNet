#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "cpplinq.hpp"
#include "json.hpp"

#include <vector>
#include <fstream>

#include "genome.h"
#include "phenotype.h"
#include "genalg.h"
#include "serialize.h"


double xnor_fitness(neat::SNeuralNetPtr brain)
{
    auto z_z = brain->Update( {0, 0} )[0];
    auto z_o = brain->Update( {0, 1} )[0];
    auto o_z = brain->Update( {1, 0} )[0];
    auto o_o = brain->Update( {1, 1} )[0];

    double error = 0.0;
    error += std::fabs(1 - z_z);
    error += std::fabs(1 - o_o);
    error += o_z;
    error += z_o;

    return std::pow(4 - error, 2);
}

SCENARIO("A neural network gets serialized into a json object", "[serialize]")
{
    using namespace nlohmann;

    GIVEN("A neural network instance")
    {
        neat::Params p;
        neat::Genome g(1, 3, 2, &p);
        neat::NeuralNet nn(g);

        neat::serialize_to_file("network.json", nn);
    }
}
