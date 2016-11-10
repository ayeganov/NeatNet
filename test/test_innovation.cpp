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

        WHEN("DB is queried")
        {
            auto result = cpplinq::from(inno_db.Innovations())
                >> cpplinq::first_or_default();
            REQUIRE(result.Type == neat::InnovationType::NONE);
        }

        WHEN("Innovation is added")
        {
            int from_id = 1;
            int to_id = 2;
            int innovation_id = 1;
            inno_db.AddNewInnovation(from_id, to_id, neat::InnovationType::NEW_LINK);
            auto result = cpplinq::from(inno_db.Innovations())
                >> cpplinq::first_or_default();

            REQUIRE(result.Type == neat::InnovationType::NEW_LINK);
            REQUIRE(result.InnovationID == innovation_id);
            REQUIRE(result.NeuronFromID == from_id);
            REQUIRE(result.NeuronToID == to_id);
        }
    }
}
