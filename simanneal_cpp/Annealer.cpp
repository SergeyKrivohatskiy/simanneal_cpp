#include "Annealer.hpp"

namespace simanneal_cpp
{

    Annealer::Annealer(state_t && initialState,
        energy_t initialStateEnergy, std::ostream & updatesOut):
        m_updatesOut(updatesOut),
        m_bestState(std::move(initialState)),
        m_bestStateEnergy(initialStateEnergy)
    {
        throw std::runtime_error("Not implemented");
    }

    void Annealer::runAnnealing(
        run_schedule const &schedule, size_t updates)
    {
        return runAnnealing(schedule.maxT, schedule.minT, schedule.steps, updates);
    }

    void Annealer::runAnnealing(
        temperature_t maxT, temperature_t minT, size_t steps, size_t updates)
    {
        throw std::runtime_error("Not implemented");
    }

    Annealer::run_schedule Annealer::computeRunSchedule(
            double targetRunTime, size_t steps)
    {
        throw std::runtime_error("Not implemented");
    }

    Annealer::state_t const &Annealer::bestState() const
    {
        return m_bestState;
    }

    Annealer::energy_t const &Annealer::bestEnergy() const
    {
        return m_bestStateEnergy;
    }

    void Annealer::printUpdate()
    {
        throw std::runtime_error("Not implemented");
    }

} // namespace simanneal_cpp