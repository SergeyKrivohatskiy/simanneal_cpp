#include "Annealer.hpp"
#include <assert.h>
#include <iomanip>

namespace simanneal_cpp
{

    Annealer::Annealer(state_t const &initialState,
        energy_t initialStateEnergy, std::ostream & updatesOut):
        m_randomGenerator(),
        m_zeroOneUniform(0.0, 1.0),
        m_updatesOut(updatesOut),
        m_bestState(initialState),
        m_bestStateEnergy(initialStateEnergy)
    {
    }

    void Annealer::runAnnealing(
        run_schedule const &schedule, size_t updates)
    {
        return runAnnealing(schedule.maxT, schedule.minT, schedule.steps, updates);
    }

    void Annealer::runAnnealing(
        temperature_t const maxT, temperature_t const minT,
        size_t const steps, size_t const updates)
    {
        size_t step = 0;
        std::chrono::system_clock::duration startTime = now();

        assert(minT < maxT);
        assert(0 < minT);
        temperature_t Tfactor = -log(maxT / minT);

        state_t prevS(bestState());
        energy_t prevE = bestEnergy();
        state_t bestS(bestState());
        energy_t bestE = bestEnergy();

        if (updates)
        {
            printUpdate(startTime, step, steps, maxT, bestE);
        }

        size_t trials = 0, accepts = 0, improves = 0;
        while (step < steps)
        {
            step += 1;
            temperature_t T = maxT * exp(Tfactor * step / steps);
            state_t newS(prevS);
            energy_t newE = moveState(newS);
            energy_t dE = newE - prevE;
            trials += 1;
            if (dE < 0.0 || exp(-dE / T) > m_zeroOneUniform(m_randomGenerator))
            {
                accepts += 1;
                if (dE < 0.0)
                {
                    improves += 1;
                }
                prevS = std::move(newS);
                prevE = std::move(newE);
                if (prevE < bestE)
                {
                    bestS = prevS;
                    bestE = prevE;
                }
            }

            if (updates)
            {
                if (step % (steps / updates) == 0)
                {
                    printUpdate(startTime, step, steps, T, newE,
                        accepts / static_cast<double>(trials),
                        improves / static_cast<double>(trials));
                    trials = 0, accepts = 0, improves = 0;
                }
            }
        }

        m_bestState = std::move(bestS);
        m_bestStateEnergy = std::move(bestE);
    }


    void Annealer::testTemperatureRun(temperature_t T, size_t steps,
        double &acceptance, double &improvement, energy_t &E) const
    {
        state_t prevS = bestState();
        energy_t prevE = bestEnergy();
        size_t accepts = 0;
        size_t improves = 0;
        for (size_t step = 0; step < steps; ++step)
        {
            state_t newS(prevS);
            energy_t newE = moveState(newS);
            energy_t dE = newE - prevE;
            if (dE < 0.0 || exp(-dE / T) > m_zeroOneUniform(m_randomGenerator))
            {
                accepts += 1;
                if (dE < 0.0)
                {
                    improves += 1;
                    prevS = std::move(newS);
                    prevE = newE;
                }
            }
        }
        E = prevE;
        acceptance = accepts / static_cast<double>(steps);
        improvement = improves / static_cast<double>(steps);
    }

    Annealer::run_schedule Annealer::computeRunSchedule(
        double const targetRunTime, size_t const steps,
        bool const printProgressMessages) const
    {
        std::chrono::system_clock::duration startTime = now();

        temperature_t T = 0.0;
        energy_t E = bestEnergy();
        state_t state = bestState();
        size_t step = 0;
        while (T == 0.0)
        {
            step += 1;
            T = abs(moveState(state) - E);
        }

        double acceptance, improvement;
        testTemperatureRun(T, steps, acceptance, improvement, E);

        step += steps;
        static temperature_t const MAX_TEMP = 1e25;
        static temperature_t const MIN_TEMP = 1e-10;
        if (printProgressMessages)
        {
            m_updatesOut << "Finding maximum temperature\n";
        }
        while (acceptance > 0.98)
        {
            T = T / 1.5;
            if (T < MIN_TEMP)
            {
                T = MIN_TEMP;
                break;
            }
            testTemperatureRun(T, steps, acceptance, improvement, E);
            step += steps;
        }
        while (acceptance < 0.98)
        {
            T = T * 1.5;
            if (T > MAX_TEMP)
            {
                T = MAX_TEMP;
                break;
            }
            testTemperatureRun(T, steps, acceptance, improvement, E);
            step += steps;
            //self.update(step, T, E, acceptance, improvement);
        }
        temperature_t const Tmax = T;
        if (printProgressMessages)
        {
            m_updatesOut << "\tmaximum temperature is " << Tmax << "\n";
        }


        if (printProgressMessages)
        {
            m_updatesOut << "Finding minimum temperature\n";
        }
        while (improvement > 0.0)
        {
            T = T / 1.5;
            if (T < MIN_TEMP)
            {
                T = MIN_TEMP;
                break;
            }
            testTemperatureRun(T, steps, acceptance, improvement, E);
            step += steps;
            //self.update(step, T, E, acceptance, improvement);
        }
        temperature_t const Tmin = T;
        if (printProgressMessages)
        {
            m_updatesOut << "\tminimum temperature is " << Tmin << "\n";
        }

        double elapsedSeconds = std::chrono::duration_cast
            <std::chrono::duration<double>>(now() - startTime).count();
        size_t const resultSteps = int(60.0 * targetRunTime * step / elapsedSeconds);
        if (printProgressMessages)
        {
            m_updatesOut << "\tTarget steps count is " << resultSteps << "\n";
        }

        return{ Tmax, Tmin, resultSteps };
    }

    Annealer::state_t const &Annealer::bestState() const
    {
        return m_bestState;
    }

    Annealer::energy_t const &Annealer::bestEnergy() const
    {
        return m_bestStateEnergy;
    }

    void Annealer::printUpdate(
        std::chrono::system_clock::duration const &startTime,
        size_t step,
        size_t steps,
        temperature_t const &currentT,
        energy_t const &currentE,
        double const currentAcceptance,
        double const currentImprovement)
    {
        double elapsedSeconds = std::chrono::duration_cast
                <std::chrono::duration<double>>(now() - startTime).count();
        std::ios stateSaver(nullptr);
        stateSaver.copyfmt(m_updatesOut);
        if (step == 0)
        {
            m_updatesOut << " Temperature        Energy    Accept   Improve     Elapsed   Remaining\n";
            m_updatesOut << std::fixed << std::setfill(' ')
                << std::setw(12) << std::setprecision(2) << currentT << "  "
                << std::setw(12) << std::setprecision(2) << currentE << "                      ";
            printTimeString(elapsedSeconds);
            m_updatesOut << "            ";
        } else {
            m_updatesOut << std::fixed << std::setfill(' ')
                << std::setw(12) << std::setprecision(2) << currentT << "  "
                << std::setw(12) << std::setprecision(2) << currentE << "  "
                << std::setw(7) << currentAcceptance * 100 << "%  "
                << std::setw(7) << currentImprovement * 100 << "%  ";
            printTimeString(elapsedSeconds);
            m_updatesOut << "  ";
            double const remainSeconds = (steps - step) * (elapsedSeconds / step);
            printTimeString(remainSeconds);
        }
        m_updatesOut << std::endl;
        m_updatesOut.copyfmt(stateSaver);
    }

    std::chrono::system_clock::duration Annealer::now()
    {
        return std::chrono::system_clock::now().time_since_epoch();
    }

    void Annealer::printTimeString(double secondsFloating)
    {
        size_t seconds = static_cast<size_t>(round(secondsFloating));
        size_t const hours = seconds / 3600;
        seconds = seconds % 3600;
        size_t const minutes = seconds / 60;
        seconds = seconds % 60;
        m_updatesOut << std::fixed
            << std::setw(4) << std::setfill(' ') << hours << ':'
            << std::setw(2) << std::setfill('0') << minutes << ':'
            << std::setw(2) << std::setfill('0') << seconds;
    }

} // namespace simanneal_cpp