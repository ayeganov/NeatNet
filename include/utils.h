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

public:
    Random():m_rand_engine(m_rand_dev())
    {
    }

    template <typename TValue>
    TValue RandomClamped(TValue lower_bound=-1, TValue upper_bound=1)
    {
        if(lower_bound >= upper_bound)
        {
            throw new std::invalid_argument("Lower bound must not be >= upper bound");
        }

        static_assert(std::is_floating_point<TValue>::value ||
                      std::is_integral<TValue>::value, "Must be either integer or double value.");

        typedef typename std::conditional<std::is_floating_point<TValue>::value,
                         std::uniform_real_distribution<double>,
                         std::uniform_int_distribution<int>>::type dist_type;
        dist_type unif_dist(lower_bound, upper_bound);
        return unif_dist(m_rand_engine);
    }

    /**
     * Returns a random double between 0 and 1
     */
    double RandomDouble()
    {
        std::uniform_real_distribution<double> unif_double(0.0, 1.0);
        return unif_double(m_rand_engine);
    }
};


template <typename T, typename Meaning>
class IDType
{
public:
    IDType() {}

    // implicit conversion
    IDType(T value) : m_id(value) {}

    inline operator T () const { return m_id; }
    inline friend bool operator==(IDType<T, Meaning> const& lhs, IDType<T, Meaning> const& rhs)
    {
        return lhs.m_id == rhs.m_id;
    }

    inline IDType<T, Meaning>& operator=(const IDType<T, Meaning>& other)
    {
        if(m_id != other.m_id)
        {
            m_id = other.m_id;
        }
        return *this;
    }

    IDType<T, Meaning>& operator++()
    {
        m_id++;
        return *this;
    }

    IDType<T, Meaning> operator++(int)
    {
        IDType<T, Meaning> tmp = m_id;
        operator++();
        return tmp;
    }

private:
    T m_id;
};

}
#endif
