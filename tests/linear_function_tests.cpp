#include <Annealer.hpp>
#include <random>
#include <catch.hpp>

class LinearAnnealer : public simanneal_cpp::Annealer<double>
{
public:
    typedef simanneal_cpp::Annealer<double> super;
    LinearAnnealer(state_t const &initialState) :
        super(initialState, getEnergy(initialState)),
        m_gen(),
        m_randomDistribution(0.7, 1.4)
    {
    }

public:
    static energy_t getEnergy(state_t const &state)
    {
        return std::abs(state * 0.5);
    }

private:
    energy_t moveState(state_t &state) const override
    {
        state = state * m_randomDistribution(m_gen);
        return getEnergy(state);
    }

private:
    mutable std::mt19937 m_gen;
    std::uniform_real_distribution<state_t> m_randomDistribution;
};



TEST_CASE("Creating Annealer test", "[Annealer]")
{
    LinearAnnealer annealer(40.0);
    CHECK(annealer.bestState() == 40.0);
    CHECK(annealer.bestEnergy() == LinearAnnealer::getEnergy(40.0));
}



TEST_CASE("RunAnnealing tests", "[Annealer][runAnnealing]")
{
    LinearAnnealer annealer(20.0);
    auto initialState = annealer.bestState();
    auto initialEnergy = annealer.bestEnergy();
    SECTION("zero steps")
    {
        annealer.runAnnealing(2, 1, 0, 0);
        CHECK(initialState == annealer.bestState());
        CHECK(initialEnergy == annealer.bestEnergy());
    }
    SECTION("many steps")
    {
        annealer.runAnnealing(1, 1e-2, 100, 0);
        CHECK(initialState != annealer.bestState());
        // 100 steps with low temperature should be
        // enough to decrease energy
        CHECK(initialEnergy > annealer.bestEnergy());
    }
}



TEST_CASE("computeRunSchedule tests", "[Annealer][computeRunSchedule]")
{
    LinearAnnealer annealer(30.0);

    LinearAnnealer::run_schedule schedule1 = annealer.computeRunSchedule(0.02);
    LinearAnnealer::run_schedule schedule2 = annealer.computeRunSchedule(0.04);

    double stepsRatio = schedule2.steps / static_cast<double>(schedule1.steps);
    CHECK(stepsRatio == Approx(2.0).epsilon(0.3));
    CHECK(schedule1.maxT == Approx(schedule2.maxT).epsilon(0.5));
    CHECK(schedule1.minT == Approx(schedule2.minT).epsilon(0.5));
    CHECK(schedule1.minT < schedule1.maxT);


    std::chrono::system_clock::duration startTime =
        std::chrono::system_clock::now().time_since_epoch();
    annealer.runAnnealing(schedule2, 0);
    double elapsedMinutes = std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::system_clock::now().time_since_epoch() - startTime).count() / 60;

    CHECK(elapsedMinutes == Approx(0.04).epsilon(0.3));
}