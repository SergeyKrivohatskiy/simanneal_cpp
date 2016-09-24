#include <catch.hpp>
#include <Annealer.hpp>

TEST_CASE("Create annealer, copy, assign and move",
          "[constructors][move][operator=]")
{
    simanneal_cpp::Annealer annealer;
    simanneal_cpp::Annealer copyAnnealer(annealer);
    simanneal_cpp::Annealer movedAnnealer(copyAnnealer);

    simanneal_cpp::Annealer annealer2;
    annealer2 = annealer;
    annealer = std::move(annealer2);
}
