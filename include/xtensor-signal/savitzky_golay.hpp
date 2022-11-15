// XTensor includes
#include <xtensor/xnoalias.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xaxis_slice_iterator.hpp>

//blas
#include <xtensor-blas/xlinalg.hpp>

namespace xt
{
    namespace signal
    {
        namespace detail
        {
            xt::xarray<double> CalculateConvolutionCoefficients(
                size_t polyorder,
                xt::xarray<double> b,
                xt::xarray<double> A)
            {
                //Define the output shape
                const auto rows = b.shape(0);
                const auto cols = polyorder + 1;

                //generate the system to solve the least squares problem
                xt::xarray<double> dot_1 = xt::linalg::dot(xt::transpose(A), A);
                xt::xarray<double> c = xt::zeros<double>({ cols });
                xt::xarray<double> dot_2 = xt::zeros<double>({ cols });
                xt::xarray<double> res_coeff = xt::zeros<double>({ rows });
                xt::noalias(res_coeff) = xt::zeros<double>({ rows });
                xt::noalias(dot_2) = xt::linalg::dot(xt::transpose(A), xt::transpose(b));

                //generate the coefficients from solving the system
                xt::noalias(c) = xt::linalg::dot(
                    xt::linalg::inv(dot_1),
                    dot_2);

                //weight the coefficients based on location in the matrix
                //we can weight the left and right handed coefficients the same
                auto i = xt::linspace<size_t>(0, b.shape(0) - 1, b.shape(0));
                auto j = xt::linspace<size_t>(0, polyorder, polyorder + 1);
                auto ii_jj = xt::meshgrid(i, j);
                xt::xarray<double> powers = xt::transpose(xt::pow(std::get<0>(ii_jj), std::get<1>(ii_jj)));
                res_coeff = xt::linalg::dot(c, powers);
                return res_coeff;
            }

            xt::xarray<double> GenerateKernels(size_t width, size_t polyorder)
            {
                const auto rows = width;
                const auto cols = polyorder + 1;

                //generate the Vandermonde Matrix
                const auto row_axis = xt::linspace<size_t>(0, rows - 1, rows);
                const auto col_axis = xt::linspace<size_t>(0, cols - 1, cols);
                const auto yv_xv = xt::meshgrid(row_axis, col_axis);
                xt::xarray<double> A_template = xt::eval(xt::pow(std::get<0>(yv_xv), std::get<1>(yv_xv)));

                xt::xarray<double> coeffs = xt::zeros<double>({ width,  rows });

                //do not thread! This causes a race condition I have yet to find
                for (size_t i = 0; i < width; ++i)
                {
                    xt::xarray<double> b1 = xt::zeros<double>({ rows });
                    b1(i) = 1.0;
                    xt::view(coeffs, i, xt::all()) = CalculateConvolutionCoefficients(polyorder, b1, A_template);
                }

                return coeffs;
            }

            xt::xarray<double> GenerateToeplitzMatrix(size_t data_length, size_t width, size_t polyorder)
            {
                //generate the kernels offline
                auto kernels = GenerateKernels(width, polyorder);

                //some cool math here
                xt::xarray<double> toeplitzMatrix = xt::zeros<double>({ data_length, data_length });
                const size_t window = width;

                width = (width - 1) / 2;

                //generate the toeplitz matrix for the discrete convolution in the hot path
                for (size_t shift = 0; shift <= data_length; shift++)
                {
                    //generate the corners
                    if (shift < width)
                    {
                        auto view = xt::view(toeplitzMatrix, shift, xt::range(0, window));
                        view = xt::view(kernels, shift, xt::all());
                        auto view2 = xt::view(
                            toeplitzMatrix, toeplitzMatrix.shape(0) - width + shift,
                            xt::range(-window, xt::placeholders::_));
                        view2 = xt::flip(xt::view(
                            kernels,
                            width - shift - 1,
                            xt::all()));
                    }
                    //generate the symetric coeffs
                    if (shift >= width && shift < (data_length - width))
                    {
                        auto view = xt::view(
                            toeplitzMatrix, shift,
                            xt::range(
                                shift - width,
                                shift - width + window));
                        view = xt::view(kernels, width, xt::all());
                    }
                }

                return toeplitzMatrix;
            }
        }

        class SavGolTransform
        {
        public:
            SavGolTransform(size_t _data_length,
                size_t _width,
                size_t _polyorder) :
                width(_width),
                polyorder(_polyorder),
                data_length(_data_length)
            {
                if (data_length < width) {
                    throw std::runtime_error(
                        "SavGolFilter: Data length less than configured width");
                }

                if (width == 0) {
                    throw std::runtime_error(
                        "SavGolFilter: Width zero not supported");
                }

                if (polyorder == 0) {
                    throw std::runtime_error(
                        "SavGolFilter: Polynomial order of zero not supported");
                }

                if (width % 2 == 0)
                {
                    throw std::runtime_error(
                        "SavGolFilter: Window length must be odd");
                }

                // Define the toeplitz values needed for the convolution kernel
                _toeplitzMatrix = detail::GenerateToeplitzMatrix(data_length, width, polyorder);
            }

            xt::xarray<double> Transform(xt::xarray<double> data)
            {
                return xt::linalg::dot(_toeplitzMatrix, data);
            }


        private:
            size_t width;
            size_t polyorder;
            size_t data_length;
            xt::xarray<double> _toeplitzMatrix;
        };


        xt::xarray<double> savgol_filter(xt::xarray<double> x, size_t window_length, size_t polyorder, std::ptrdiff_t axis = -1)
        {
            xt::xarray<double> out = xt::zeros<double>(x.shape());
            auto saxis = xt::normalize_axis(x.dimension(), axis);
            auto begin = xt::axis_slice_begin(x, saxis);
            auto end = xt::axis_slice_end(x, saxis);
            auto iter_out = xt::axis_slice_begin(out, saxis);

            SavGolTransform filter(x.shape(saxis), window_length, polyorder);

            for (auto iter = begin; iter != end; iter++)
            {
                (*iter_out++) = filter.Transform(*iter);
            }

            return out;
        }

        xt::xarray<double> savgol_coeffs(size_t window_length, size_t polyorder)
        {
            return xt::view(detail::GenerateKernels(window_length, polyorder), (window_length - 1) / 2);
        }
    }
}


