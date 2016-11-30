#define CATCH_CONFIG_MAIN
#include "catch.hpp"


#include "genalg.h"


SCENARIO("GenAlg gets initialized with population size of 40 and 2 inputs and 1 output.", "[genalg]")
{
    GIVEN("A task of learning a XOR function")
    {
        neat::GenAlg ga(40, 2, 1);

    }
}
