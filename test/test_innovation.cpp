#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include <iostream>

#include "cpplinq.hpp"

#include "innovation.h"

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
            neat::NeuronID from_id = 1;
            neat::NeuronID to_id = 2;
            neat::InnovationID innovation_id = 1;
            inno_db.AddLinkInnovation(from_id, to_id);
            auto result = cpplinq::from(inno_db.Innovations())
                >> cpplinq::first_or_default();

            REQUIRE(result.Type == neat::InnovationType::NEW_LINK);
            REQUIRE(result.ID == innovation_id);
            REQUIRE(result.NeuronFromID == from_id);
            REQUIRE(result.NeuronToID == to_id);
        }
    }
}
