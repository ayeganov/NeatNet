#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "params.h"


SCENARIO("Reading a parameters file", "[params]")
{
    GIVEN("A valid path to a valid parameters file")
    {
        using json = nlohmann::json;
        std::string config_path = "test_params.json";

        WHEN("File is read")
        {
            neat::Params p(config_path);
            THEN("All values are read correctly")
            {
                REQUIRE(p.ActivationMutationChance() == 1);
                REQUIRE(p.AddLinkChance() == 2);
                REQUIRE(p.AddNeuronChance() == 3);
                REQUIRE(p.AddRecurLinkChance() == 4);
                REQUIRE(p.CompatibilityThreshold() == 5);
                REQUIRE(p.CrossoverChance() == 6);
                REQUIRE(p.DisjointScaler() == 7);
                REQUIRE(p.ExcessScaler() == 8);
                REQUIRE(p.MatchScaler() == 9);
                REQUIRE(p.MaxActivationPerturbation() == 10);
                REQUIRE(p.MaxNeurons() == 11);
                REQUIRE(p.MaxPerturbation() == 12);
                REQUIRE(p.MutationChance() == 13);
                REQUIRE(p.NumAddLinkAttempts() == 14);
                REQUIRE(p.NumAddRecurLinkAttempts() == 15);
                REQUIRE(p.NumBestGenomes() == 16);
                REQUIRE(p.NumFindOldLinkAttempts() == 17);
                REQUIRE(p.NumGensAllowedNoImprov() == 18);
                REQUIRE(p.OldPenaltyScaler() == 19);
                REQUIRE(p.OldPenaltyThreshold() == 20);
                REQUIRE(p.PopulationSize() == 21);
                REQUIRE(p.SurvivalRate() == 22);
                REQUIRE(p.NewWeightChance() == 23);
                REQUIRE(p.YoungBonusThreshold() == 24);
                REQUIRE(p.YoungBonusScaler() == 25);
            }
        }

        WHEN("File is read into json object and params is initialized with it.")
        {
            std::ifstream in(config_path);
            json obj = json::parse(in);
            neat::Params p(obj);

            THEN("All values are read correctly")
            {
                REQUIRE(p.ActivationMutationChance() == 1);
                REQUIRE(p.AddLinkChance() == 2);
                REQUIRE(p.AddNeuronChance() == 3);
                REQUIRE(p.AddRecurLinkChance() == 4);
                REQUIRE(p.CompatibilityThreshold() == 5);
                REQUIRE(p.CrossoverChance() == 6);
                REQUIRE(p.DisjointScaler() == 7);
                REQUIRE(p.ExcessScaler() == 8);
                REQUIRE(p.MatchScaler() == 9);
                REQUIRE(p.MaxActivationPerturbation() == 10);
                REQUIRE(p.MaxNeurons() == 11);
                REQUIRE(p.MaxPerturbation() == 12);
                REQUIRE(p.MutationChance() == 13);
                REQUIRE(p.NumAddLinkAttempts() == 14);
                REQUIRE(p.NumAddRecurLinkAttempts() == 15);
                REQUIRE(p.NumBestGenomes() == 16);
                REQUIRE(p.NumFindOldLinkAttempts() == 17);
                REQUIRE(p.NumGensAllowedNoImprov() == 18);
                REQUIRE(p.OldPenaltyScaler() == 19);
                REQUIRE(p.OldPenaltyThreshold() == 20);
                REQUIRE(p.PopulationSize() == 21);
                REQUIRE(p.SurvivalRate() == 22);
                REQUIRE(p.NewWeightChance() == 23);
                REQUIRE(p.YoungBonusThreshold() == 24);
                REQUIRE(p.YoungBonusScaler() == 25);
            }

        }
    }
}


SCENARIO("Reading parameters from string", "[params]")
{
    GIVEN("A valid string with parameters")
    {
        std::string config = "{ \"ChanceAddLink\": 10}";

        WHEN("String is parsed")
        {
            neat::Params p = neat::Params::FromString(config);
            THEN("All values are read correctly")
            {
                REQUIRE(p.AddLinkChance() == 10);
            }
        }
    }
}


SCENARIO("Copying params instance with copy constructor", "[params]")
{
    GIVEN("A valid path to a valid parameters file")
    {
        std::string config = "{ \"ChanceAddLink\": 10}";

        WHEN("Params is initialized with another instance of Params")
        {
            neat::Params p = neat::Params::FromString(config);
            neat::Params pp(p);

            THEN("All values are read and copied correctly")
            {
                REQUIRE(p.AddLinkChance() == 10);
                REQUIRE(pp.AddLinkChance() == 10);
            }
        }
    }
}
