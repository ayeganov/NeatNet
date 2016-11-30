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

    template <typename OtherT, typename OtherMeaning>
    inline friend bool operator==(IDType<T, Meaning> const& lhs, IDType<OtherT, OtherMeaning> const& rhs)
    {
        constexpr bool same_types = std::is_same<Meaning, OtherMeaning>::value;
        static_assert(same_types, "You can only compare ID's of the same type.");
        return lhs.m_id == rhs.m_id;
    }

    template <typename OtherT, typename OtherMeaning>
    inline IDType<T, Meaning>& operator=(const IDType<OtherT, OtherMeaning>& other)
    {
        constexpr bool same_types = std::is_same<Meaning, OtherMeaning>::value;
        static_assert(same_types, "You can only assign ID's of the same type.");

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


/**
 * This class calculates total, average and standard deviation values of a
 * series of numbers given to it.
 */
class RunningStat
{
public:
    RunningStat(): m_num_values(0),
                   m_old_mean(0),
                   m_new_mean(0),
                   m_old_std(0),
                   m_new_std(0),
                   m_total(0)
    {}

    void Clear()
    {
        m_num_values = 0;
        m_old_mean = m_new_mean = m_old_std = m_new_std = 0.0;
        m_total = 0.0;
    }

    void Push(double value)
    {
        m_total += value;
        ++m_num_values;

        if(m_num_values == 1)
        {
            m_old_mean = m_new_mean = value;
            m_old_std = 0.0;
        }
        else
        {
            m_new_mean = m_old_mean + (value - m_old_mean) / m_num_values;
            m_new_std = m_old_std + (value - m_old_mean) * (value - m_new_mean);

            m_old_mean = m_new_mean;
            m_old_std = m_new_std;
        }
    }

    int NumValues()
    {
        return m_num_values;
    }

    double Mean()
    {
        return (m_num_values > 0) ? m_new_mean : 0.0;
    }

    double Variance()
    {
        return (m_num_values > 1 ? m_new_std / (m_num_values - 1) : 0.0);
    }

    double StandardDeviation()
    {
        return std::sqrt(Variance());
    }

    double Total()
    {
        return m_total;
    }

private:
    int m_num_values;
    double m_old_mean, m_new_mean, m_old_std, m_new_std;
    double m_total;
};


}
#endif
