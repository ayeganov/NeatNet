#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "cpplinq.hpp"
#include "json.hpp"

#include <vector>

#include "genome.h"
#include "phenotype.h"
#include "genalg.h"
#include "serialize.h"


SCENARIO("A neural network gets serialized into a json file", "[serialize]")
{
    using namespace nlohmann;
    using namespace cpplinq;

    GIVEN("A neural network instance")
    {
        neat::Params p;
        neat::Genome g(1, 3, 2, &p);
        neat::InnovationDB inno_db(g.NeuronGenes(), g.NeuronLinks());

        g.AddNeuron(1.0, inno_db, 100);
        g.AddNeuron(1.0, inno_db, 100);
        g.AddNeuron(1.0, inno_db, 100);
        g.AddNeuron(1.0, inno_db, 100);
        g.AddNeuron(1.0, inno_db, 100);

        g.AddLink(1.0, 0.5, inno_db, 100, 100);
        g.AddLink(1.0, 0.5, inno_db, 100, 100);
        g.AddLink(1.0, 0.5, inno_db, 100, 100);
        g.AddLink(1.0, 0.5, inno_db, 100, 100);
        g.AddLink(1.0, 0.5, inno_db, 100, 100);

        neat::NeuralNet nn(g);

        WHEN("It gets serialized to file")
        {
            neat::serialize_to_file("network.json", nn);

            THEN("It gets deserialized correctly")
            {
                auto object = neat::deserialize_from_file("network.json");
                neat::NeuralNet desernn(object);

                auto nnresult = nn.Update({0.5, 0.5, 0.5});
                auto desernnresult = desernn.Update({0.5, 0.5, 0.5});

                from(nnresult)
                >> zip_with(from(desernnresult))
                >> for_each([](const std::pair<double, double>& p)
                {
                    double diff = std::fabs(p.first - p.second);
                    REQUIRE(diff < 0.0000000001);
                });
            }
        }
    }
}
