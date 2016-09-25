#pragma once
#include <iostream>

namespace simanneal_cpp
{

    // TODO rewrite to template<T>, T is state_t
    class Annealer
    {
    public:
        typedef double state_t;
        typedef double temperature_t;
        typedef double energy_t;
        struct run_schedule
        {
            temperature_t maxT;
            temperature_t minT;
            size_t steps;
        };

    public:
        Annealer(state_t &&initialState,
                 energy_t initialStateEnergy,
                 std::ostream &updatesOut = std::cout);

        void runAnnealing(
            run_schedule const &schedule,
            size_t updates = 100);

        void runAnnealing(
            temperature_t maxT = 25000.0,
            temperature_t minT = 2.5,
            size_t steps = 50000,
            size_t updates = 100);

        run_schedule computeRunSchedule(
            double targetRunTime,
            size_t steps = 2000);

        state_t const &bestState() const;

        energy_t const &bestEnergy() const;

    protected:
        virtual energy_t moveState(state_t &initialState) const = 0;

    private:
        void printUpdate();

    private:
        std::ostream &m_updatesOut;
        state_t m_bestState;
        energy_t m_bestStateEnergy;
	};

} // namespace simanneal_cpp