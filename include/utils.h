#ifndef __UTILS_H__
#define __UTILS_H__

#include <random>
#include <stdexcept>
#include <type_traits>

namespace Utils
{
template<typename TEngine>
class Random
{
private:
    std::random_device m_rand_dev;
    TEngine m_rand_engine;

    int RandomClamped(int lower, int upper)
    {
        std::uniform_int_distribution<int> unif_int(lower, upper);
        return unif_int(m_rand_engine);
    }

public:
    Random():m_rand_engine(m_rand_dev())
    {
    }

    template<typename TValue>
    TValue RandomClamped(TValue lower_bound, TValue upper_bound)
    {
        if(lower_bound >= upper_bound)
        {
            throw new std::invalid_argument("Lower bound must not be >= upper bound");
        }

        static_assert(std::is_floating_point<TValue>::value || std::is_integral<TValue>::value,
                "Supplied type is not supported for random numbers.");

        typename std::conditional<std::is_floating_point<TValue>::value,
                         std::uniform_real_distribution<TValue>,
                         std::uniform_int_distribution<TValue>>::type type;

        if(std::is_floating_point<TValue>::value)
        {
            std::uniform_real_distribution<TValue> unif_double(lower_bound, upper_bound);
            return unif_double(m_rand_engine);
        }
        else if(std::is_integral<TValue>::value)
        {
            std::uniform_int_distribution<TValue> unif_int(lower_bound, upper_bound);
            return unif_int(m_rand_engine);
        }
    }
};
}
#endif
