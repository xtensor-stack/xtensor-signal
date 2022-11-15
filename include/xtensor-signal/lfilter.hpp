
#include <cstddef>

#include <xtensor/xtensor.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xaxis_slice_iterator.hpp>
#include <xtensor/xarray.hpp>

namespace xt
{
    namespace signal
    {
        namespace detail
        {
            template<typename E1, typename E2, typename E3>
            auto lfilter(E1&& b, E2&& a, E3&& x)
            {
                using value_type =  typename std::decay_t<E3>::value_type;
                xt::xtensor<value_type, 1> out = xt::zeros<value_type>({x.shape(0)});

                for(int i=0; i < x.shape(0); i++)
                {
                    value_type tmp = 0;
                    int j=0;
                    for(j=0; j < b.shape(0); j++)
                    {
                        if(!(i - j < 0))
                        {
                            tmp += b(j) * x(i-j);
                        }
                    }

                    for(j=1; j < a.shape(0); j++)
                    {
                        if(!(i - j < 0))
                        {
                            tmp -= a(j)*out(i-j);
                        }
                    }
                    
                    tmp /= a(0);
                    out(i) = tmp;
                }
                return out;
            }
        }
        /*
        * @brief performs a 1D dilter operation along the specified axis.
        * @param b the numerator of the filter expression
        * @param a the denominator of the filter expression
        * @param x input dataset
        * @param axis the axis along which to perform the filter operation
        * @return filtered version of x
        */
        template<typename E1, typename E2, typename E3>
        auto lfilter(E1&& b, E2&& a, E3&& x, std::ptrdiff_t axis = -1)
        {
            using value_type =  typename std::decay_t<E3>::value_type;
            xt::xarray<value_type> out = xt::zeros<value_type>(x.shape());
            auto saxis = xt::normalize_axis(x.dimension(), axis);
            auto begin = xt::axis_slice_begin(x, saxis);
            auto end = xt::axis_slice_end(x, saxis);
            auto iter_out = xt::axis_slice_begin(out, saxis);

            for (auto iter = begin; iter != end; iter++)
            {
                (*iter_out++) = detail::lfilter(b, a, *iter);
            }

            return out;
        }
    }
}
