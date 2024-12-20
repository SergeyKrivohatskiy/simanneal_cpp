#pragma once
#include <iostream>
#include <chrono>
#include <random>

namespace simanneal_cpp
{

    template<class T>
    class Annealer
    {
    public:
        typedef T state_t;
        typedef double temperature_t;
        typedef double energy_t;
        struct run_schedule
        {
            temperature_t maxT;
            temperature_t minT;
            size_t steps;
        };

    public:
        Annealer(state_t const &initialState,
                 energy_t initialStateEnergy,
                 std::ostream &updateMessagesOut = std::cout);

        void runAnnealing(
            run_schedule const &schedule,
            size_t updateMessages = 100);

        void runAnnealing(
            temperature_t maxTemperature = 25000.0,
            temperature_t minTemperature = 2.5,
            size_t steps = 50000,
            size_t updateMessages = 100);

        run_schedule computeRunSchedule(
            double targetRunTimeMinutes,
            size_t steps = 2000,
            bool printProgressMessages = false) const;

        state_t const &bestState() const;

        energy_t const &bestEnergy() const;

    protected:
        virtual energy_t moveState(state_t &initialState) const = 0;

    private:
        typedef std::chrono::system_clock::duration time_t;
        void printUpdate(
            time_t const &startTime,
            size_t step,
            size_t steps,
            temperature_t const &temperature,
            energy_t const &energy,
            double acceptance = 0.0,
            double improvement = 0.0);
        void testTemperatureRun(temperature_t t, size_t steps,
            double &accepts, double &improves, energy_t &E) const;

    private:
        static time_t now();
        void printTimeString(double seconds);

    private:
        mutable std::mt19937 m_randomGenerator;
        mutable std::uniform_real_distribution<double> m_zeroOneUniform;
        std::ostream &m_updatesOut;
        state_t m_bestState;
        energy_t m_bestStateEnergy;
	};

} // namespace simanneal_cpp


#include "Annealer.inl"
