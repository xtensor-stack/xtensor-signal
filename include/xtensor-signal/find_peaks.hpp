#ifndef XTENSOR_SIGNAL_FIND_PEAKS_HPP
#define XTENSOR_SIGNAL_FIND_PEAKS_HPP

#include <algorithm>
#include <optional>
#include <type_traits>
#include <variant>

#include <xtensor/xbuilder.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xtensor.hpp>

namespace xt {
namespace signal {
/**
 * @brief finds all peaks and applies filters to the peaks based on the
 * parameters provided.
 * @details see
 * https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
 * for more details
 */
template <typename T> class find_peaks {
public:
  using height_t =
      std::variant<float, std::pair<float, float>, xt::xtensor<float, 1>,
                   std::pair<xt::xtensor<float, 1>, xt::xtensor<float, 1>>>;

  using threshold_t =
      std::variant<float, std::pair<float, float>, xt::xtensor<float, 1>,
                   std::pair<xt::xtensor<float, 1>, xt::xtensor<float, 1>>>;

  using prominence_t =
      std::variant<float, std::pair<float, float>, xt::xtensor<float, 1>,
                   std::pair<xt::xtensor<float, 1>, xt::xtensor<float, 1>>>;

  using width_t =
      std::variant<float, std::pair<float, float>, xt::xtensor<float, 1>,
                   std::pair<xt::xtensor<float, 1>, xt::xtensor<float, 1>>>;

  using plateau_t =
      std::variant<size_t, std::pair<size_t, size_t>, xt::xtensor<size_t, 1>,
                   std::pair<xt::xtensor<size_t, 1>, xt::xtensor<size_t, 1>>>;

  find_peaks();
  /**
   * @param minimum height thresholds to be filtered. Only minimum heights are
   * implemented.
   */
  find_peaks &set_height(height_t height);
  /**
   * @param threshold defines the threshold required defining the required
   * vertical distance between peaks
   */
  find_peaks &set_threshold(threshold_t threshold);
  /**
   * @param distance defines the horizontal distance between peaks proritized by
   * height
   */
  find_peaks &set_distance(size_t distance);
  /**
   * @param prominence deines the prominance required by peaks
   */
  find_peaks &set_prominence(prominence_t prominence);
  /**
   * @param width defines the minimum width for eligable points.
   */
  find_peaks &set_width(width_t width);
  /**
   * @param wlen defined wlen for calculating prominence. Minimizing wLen can
   * improve performance
   */
  find_peaks &set_wlen(size_t wlen);
  /**
   * @param rel_height used in defining the width of peaks based on the
   * location. Defaults to .5
   */
  find_peaks &set_rel_height(float rel_height);
  /**
   * @param plateau_size defines the plateau required for peaks
   */
  find_peaks &set_plateau_size(plateau_t plateau_size);

  /**
   * @brief finds the peaks on a 1D dataset based on the parameters provided
   * previously
   * @param x 1D Dataset with peaks to find
   * @return xarray of peak locations found in Data after applying the
   * pre-defined filters
   */
  xtensor<T, 1> operator()(xtensor<T, 1> x);

private:
  xtensor<size_t, 1> check_height(const xtensor<size_t, 1> &peaks,
                                  const xtensor<T, 1> &x);
  xtensor<size_t, 1> check_threshold(const xtensor<size_t, 1> &peaks,
                                     const xtensor<T, 1> &x);
  xtensor<size_t, 1> check_distance(const xtensor<size_t, 1> &peaks,
                                    const xtensor<T, 1> &x);
  xtensor<size_t, 1> check_prominence(const xtensor<size_t, 1> &peaks,
                                      const xtensor<T, 1> &x);
  xtensor<size_t, 1> check_width(const xtensor<size_t, 1> &peaks,
                                 const xtensor<T, 1> &x);
  xtensor<size_t, 1> check_plateau_size(const xtensor<size_t, 1> &peaks,
                                        const xtensor<size_t, 1> &left,
                                        const xtensor<size_t, 1> &right);
  std::optional<height_t> m_height;
  std::optional<threshold_t> m_threshold;
  std::optional<size_t> m_distance;
  std::optional<prominence_t> m_prominence;
  std::optional<width_t> m_width;
  std::optional<size_t> m_wlen;
  std::optional<plateau_t> m_plateau_size;
  float m_rel_height;
};

namespace detail {
template <typename... Ts> struct Overload : Ts... {
  using Ts::operator()...;
};
template <class... Ts> Overload(Ts...) -> Overload<Ts...>;

template <class E1, class E2, class E3, class E4 = decltype(xt::xnone())>
auto select_by_peak_threshold(E1 &&x, E2 &&peaks, E3 &&tmin,
                              E4 &&tmax = xt::xnone()) {
  // Stack thresholds on both sides to make min / max operations easier :
  // tmin is compared with the smaller, and tmax with the greater thresold to
  // each peak's side
  using value_type = typename std::decay<E1>::type::value_type;
  auto peak_left = xt::eval(peaks - 1);
  auto peak_right = xt::eval(peaks + 1);
  xt::xarray<value_type> left =
      xt::view(x, xt::keep(peaks)) - xt::view(x, xt::keep(peak_left));
  xt::xarray<value_type> right =
      xt::view(x, xt::keep(peaks)) - xt::view(x, xt::keep(peak_right));
  auto stacked_thresholds = xt::vstack(xt::xtuple(left, right));
  xt::xarray<bool> keep = xt::ones<bool>({peaks.size()});
  if constexpr (std::is_same<typename std::decay<E3>::type,
                             decltype(xt::xnone())>::value == false) {
    auto min_thresholds = xt::amin(stacked_thresholds, {0});
    keep = keep && (tmin <= min_thresholds);
  }
  if constexpr (std::is_same<typename std::decay<E4>::type,
                             decltype(xt::xnone())>::value == false) {
    auto max_thresholds = xt::amax(stacked_thresholds, {0});
    keep = keep && (max_thresholds <= tmax);
  }
  return std::make_tuple(keep, stacked_thresholds(0), stacked_thresholds(1));
}

template <class E1, class E2, class E3>
auto select_by_peak_distance(E1 &&peaks, E2 &&priority, E3 &&distance) {
  int64_t j, k;
  size_t peaks_size = peaks.shape(0);
  // Round up because actual peak distance can only be natural number
  auto distance_ = std::ceil(distance);
  auto keep =
      xt::eval(xt::ones<uint8_t>({peaks_size})); // Prepare array of flags

  // Create map from `i` (index for `peaks` sorted by `priority`) to `j` (index
  // for `peaks` sorted by position).This allows to iterate `peaks`and `keep`
  // with `j` by order of `priority` while still maintaining the ability to
  // step to neighbouring peaks with(`j` + 1) or (`j` - 1).
  auto priority_to_position = xt::argsort(priority);
  auto index = xt::eval(xt::arange<int>(peaks_size - 1, -1, -1));
  // Highest priority first->iterate in reverse order(decreasing)
  for (auto i : index) {
    // "Translate" `i` to `j` which points to current peak whose
    // neighbours are to be evaluated
    j = priority_to_position(i);
    if (keep(j) == 0) {
      // Skip evaluation for peak already marked as "don't keep"
      continue;
    }

    k = j - 1;
    // Flag "earlier" peaks for removal until minimal distance is exceeded
    while (0 <= k && peaks(j) - peaks(k) < distance_) {
      keep(k) = 0;
      k -= 1;
    }

    k = j + 1;
    // Flag "later" peaks for removal until minimal distance is exceeded
    while (k < peaks_size && peaks(k) - peaks(j) < distance_) {
      keep(k) = 0;
      k += 1;
    }
  }
  return keep;
}

/**
 * @brief Evaluate where the generic property of peaks confirms to an interval.
 * @param peak_properties
 *     An array with properties for each peak.
 * @param pmin
 *     Lower interval boundary for `peak_properties`. ``None`` is interpreted as
 *     an open border.
 * @param pmax
 *     Upper interval boundary for `peak_properties`. ``None`` is interpreted as
 *     an open border.
 * @returns
 *     A boolean mask evaluating to true where `peak_properties` confirms to the
 *     interval.
 * @notes Derived from:
 * https://github.com/scipy/scipy/blob/main/scipy/signal/_peak_finding.py
 */
template <class E1, class E2, class E3 = decltype(xt::xnone())>
auto select_by_property(E1 &&peak_properties, E2 &&pmin,
                        E3 &&pmax = xt::xnone()) {
  xt::xarray<bool> keep = xt::ones<bool>({peak_properties.shape(0)});

  // if pmin is available
  if constexpr (std::is_same<typename std::decay<E2>::type,
                             decltype(xt::xnone())>::value == false) {
    keep = keep && (pmin <= peak_properties);
  }

  // if pmax is available
  if constexpr (std::is_same<typename std::decay<E3>::type,
                             decltype(xt::xnone())>::value == false) {
    keep = keep && (peak_properties <= pmax);
  }

  return keep;
}

/**
 * @brief Calculate the prominence of each peak in a signal.
 * @details The prominence of a peak measures how much a peak stands out from
 * the surrounding baseline of the signal and is defined as the vertical
 * distance between the peak and its lowest contour line.
 * @param x sequence A signal with peaks.
 * @param peaks sequence Indices of peaks in `x`.
 * @param wlen int A window length in samples that optionally limits the
 * evaluated area for each peak to a subset of `x`. The peak is always placed in
 * the middle of the window therefore the given length is rounded up to the next
 * odd integer. This parameter can speed up the calculation See notes.
 * @returns
 * prominences The calculated prominences for each peak in `peaks`.
 * left_bases, right_bases
 *     The peaks' bases as indices in `x` to the left and right of each peak.
 *     The higher base of each pair is a peak's lowest contour line.
 * @note Derived from:
 * https://github.com/scipy/scipy/blob/main/scipy/signal/_peak_finding_utils.pyx
 */
template <class E1, class E2, class E3>
auto peak_prominences(E1 &&x, E2 &&peaks, E3 &&wlen) {
  auto prominences = xt::empty<double>({peaks.shape(0)});
  auto left_bases = xt::empty<int64_t>({peaks.shape(0)});
  auto right_bases = xt::empty<int64_t>({peaks.shape(0)});

  for (size_t peak_nr = 0; peak_nr < peaks.shape(0); peak_nr++) {
    auto peak = peaks(peak_nr);
    auto i_min = 0;
    auto i_max = x.shape(0) - 1;
    XTENSOR_ASSERT(!(i_min >= peak || peak >= i_max));
    if (2 <= wlen) {
      i_min = std::max(static_cast<int64_t>(peak - wlen / 2),
                       static_cast<int64_t>(i_min));
      i_max = std::min(static_cast<int64_t>(peak + wlen / 2),
                       static_cast<int64_t>(i_max));
    }

    // find the left bases in interval [i_min, peak]
    left_bases(peak_nr) = peak;
    int64_t i = left_bases(peak_nr);
    auto left_min = x(peak);
    while (i_min <= i && x(i) <= x(peak)) {
      if (x(i) < left_min) {
        left_min = x(i);
        left_bases(peak_nr) = i;
      }
      i--;
    }

    right_bases(peak_nr) = peak;
    i = right_bases(peak_nr);
    auto right_min = x(peak);
    while (i <= i_max && x(i) <= x(peak)) {
      if (x(i) <= right_min) {
        right_min = x(i);
        right_bases(peak_nr) = i;
      }
      i++;
    }

    prominences(peak_nr) = x(peak) - std::max(left_min, right_min);
  }

  return std::make_tuple(prominences, left_bases, right_bases);
}

/**
 * @brief Ensure argument `wlen` is of type `np.intp` and larger than 1.
 * @returns The original `value` rounded up to an integer or -1 if `value` was
 * None.
 * @note Derived from
 * https://github.com/scipy/scipy/blob/main/scipy/signal/_peak_finding.py
 */
int arg_wlen_as_expected(std::optional<size_t> value) {
  // if the value is a none type
  if (!value.has_value()) {
    return -1;
  } else {
    // otherwise we have a number
    // could probably add a check for arithmatic type here
    if (1 < value.value()) {
      return std::ceil(value.value());
    }
  }

  XTENSOR_ASSERT(value.value() > 1);

  return 0;
}

/**
 * @brief Calculate the width of each peak in a signal.
 * @details This function calculates the width of a peak in samples at a
 * relative distance to the peak's height and prominence.
 * @param x A signal with peaks.
 * @param peaks Indices of peaks in `x`.
 * @param rel_height
 *     Chooses the relative height at which the peak width is measured as a
 *     percentage of its prominence. 1.0 calculates the width of the peak at
 *     its lowest contour line while 0.5 evaluates at half the prominence
 *     height. Must be at least 0. See notes for further explanation.
 * @param prominence_data
 *     A tuple of three arrays matching the output of `peak_prominences` when
 *     called with the same arguments `x` and `peaks`. This data are calculated
 *     internally if not provided.
 * @param wlen
 *     A window length in samples passed to `peak_prominences` as an optional
 *     argument for internal calculation of `prominence_data`. This argument
 *     is ignored if `prominence_data` is given.
 * @param left_bases length must equal the length of peaks and defines the
 * location of the left side index of the given peak
 * @param right_bases length must equal the length of peaks and defines the
 * location of the right side index of the given peak
 * @returns returns a tuple of 4 xarray or xtensor objects with the following
 * order: peak widths, height of the peak above the threshold (height -
 * prom*rel_height), left side of the width, right side of the width
 * @note Derived from
 * https://github.com/scipy/scipy/blob/main/scipy/signal/_peak_finding_utils.pyx
 */
template <class E1, class E2, class E3, class E4, class E5, class E6>
auto peak_widths(E1 &&x, E2 &&peaks, E3 &&rel_height, E4 &&prominences,
                 E5 &&left_bases, E6 &&right_bases) {
  XTENSOR_ASSERT(rel_height > 0);

  XTENSOR_ASSERT(peaks.shape(0) == prominences.shape(0) &&
                 left_bases.shape(0) == right_bases.shape(0) &&
                 prominences.shape(0) == right_bases.shape(0));

  auto widths = xt::empty<double>({peaks.shape(0)});
  auto width_heights = xt::empty<double>({peaks.shape(0)});
  auto left_ips = xt::empty<double>({peaks.shape(0)});
  auto right_ips = xt::empty<double>({peaks.shape(0)});

  for (size_t p = 0; p < peaks.shape(0); p++) {
    size_t i_min = left_bases(p);
    size_t i_max = right_bases(p);
    auto peak = peaks(p);

    // validate the bounds and order
    XTENSOR_ASSERT(
        !(i_min < 0 || peak < i_min || i_max < peak || i_max >= x.shape(0)));

    auto height = x(peak) - prominences(p) * rel_height;
    width_heights(p) = height;
    auto i = peak;
    // find intersecption point on left side
    while (i_min < i && height < x(i)) {
      i--;
    }

    double left_ip = static_cast<double>(i);
    if (x(i) < height) {
      left_ip += (height - x(i)) / (x(i + 1) - x(i));
    }

    i = peak;
    while (i < i_max && height < x(i)) {
      i++;
    }

    double right_ip = static_cast<double>(i);
    if (x(i) < height) {
      right_ip -= (height - x(i)) / (x(i - 1) - x(i));
    }

    widths(p) = right_ip - left_ip;
    left_ips(p) = left_ip;
    right_ips(p) = right_ip;
  }
  return std::make_tuple(widths, width_heights, left_ips, right_ips);
}

/**
 * @brief Find local maxima in a 1D array.
 * @details This function finds all local maxima in a 1D array and returns the
 * indices for their edges and midpoints (rounded down for even plateau sizes).
 * https://github.com/scipy/scipy/blob/main/scipy/signal/_peak_finding_utils.pyx
 * @returns tuple of length 3 packed in the following order: peak location, left
 * edge, right edge
 */
template <typename T>
std::tuple<xt::xarray<size_t>, xt::xarray<size_t>, xt::xarray<size_t>>
local_maxima_1d(T &&x) {
  // Preallocate, there can't be more maxima than half the size of `x`
  std::vector<size_t> _midpoints;
  std::vector<size_t> _left_edges;
  std::vector<size_t> _right_edges;

  size_t m = 0;
  size_t i = 1; // Pointer to current sample, first one can't be maxima
  size_t i_max = x.shape(0) - 1; // Last sample can't be maxima

  while (i < i_max) {
    if (x(i - 1) < x(i)) {
      auto i_ahead = i + 1;
      // Find next sample that is unequal to x[i]
      while (i_ahead < i_max && x(i_ahead) == x(i)) {
        i_ahead++;
      }
      // Maxima is found if next unequal sample is smaller than x[i]
      if (x(i_ahead) < x(i)) {
        _left_edges.push_back(i);
        _right_edges.push_back(i_ahead - 1);
        _midpoints.push_back((_left_edges.back() + _right_edges.back()) / 2);
        m++;
        i = i_ahead;
      }
    }
    i++;
  }

  // this is weird because you cannot easily append to a xtensor
  xt::xarray<size_t> midpoints = xt::adapt(_midpoints, {_midpoints.size()});
  xt::xarray<size_t> left_edges = xt::adapt(_left_edges, {_left_edges.size()});
  xt::xarray<size_t> right_edges =
      xt::adapt(_right_edges, {_right_edges.size()});
  return std::make_tuple(midpoints, left_edges, right_edges);
}

} // namespace detail

/**
 * @brief implements the peak widths interface.
 * @param x 1D data vector which matches the data vector used to generate peaks.
 * @param peaks locations of maximums in x. Generally generated from find peaks.
 * @param rel_height defines the height of the peak that is used to generate the
 * width.
 * @param wlen defines the window length used to calculate prominance.
 * @returns tuple of arrays of widths, width heights, left bound and right
 * bounds.
 */
template <typename E1, typename E2, typename E3 = double>
auto peak_widths(E1 &&x, E2 &&peaks, E3 &&rel_height,
                 std::optional<size_t> wlen) {
  // check that we have only one dimention
  XTENSOR_ASSERT(x.shape().size() == 1);
  XTENSOR_ASSERT(peaks.shape().size() == 1);

  // declare an internal variable for prominance data
  xt::xarray<double> _prominence_data;
  // check if prominance data is being supplied

  // Calculate prominence if not supplied and use wlen if supplied.
  // check if wlen is an acceptable parameter
  auto wlen_safe = detail::arg_wlen_as_expected(wlen);
  auto [prominences, left_bases, right_bases] =
      detail::peak_prominences(x, peaks, wlen_safe);
  return detail::peak_widths(x, peaks, rel_height, prominences, left_bases,
                             right_bases);
}
template <typename T> find_peaks<T>::find_peaks() : m_rel_height(.5) {}

template <typename T>
find_peaks<T> &find_peaks<T>::set_height(height_t height) {
  m_height = std::make_optional(std::move(height));
  return *this;
}
template <typename T>
find_peaks<T> &find_peaks<T>::set_threshold(threshold_t threshold) {
  m_threshold = std::make_optional(std::move(threshold));
  return *this;
}
template <typename T>
find_peaks<T> &find_peaks<T>::set_distance(size_t distance) {
  m_distance = std::make_optional(std::move(distance));
  return *this;
}
template <typename T>
find_peaks<T> &find_peaks<T>::set_prominence(prominence_t prominence) {
  m_prominence = std::make_optional(std::move(prominence));
  return *this;
}
template <typename T> find_peaks<T> &find_peaks<T>::set_width(width_t width) {
  m_width = std::make_optional(std::move(width));
  return *this;
}
template <typename T> find_peaks<T> &find_peaks<T>::set_wlen(size_t wlen) {
  m_wlen = std::make_optional(std::move(wlen));
  return *this;
}
template <typename T>
find_peaks<T> &find_peaks<T>::set_rel_height(float rel_height) {
  // did not move trivial type move = copy overhead
  m_rel_height = rel_height;
  return *this;
}
template <typename T>
find_peaks<T> &find_peaks<T>::set_plateau_size(plateau_t plateau_size) {
  m_plateau_size = std::make_optional(std::move(plateau_size));
  return *this;
}
template <typename T>
xtensor<size_t, 1> find_peaks<T>::check_height(const xtensor<size_t, 1> &peaks,
                                               const xtensor<T, 1> &x) {
  auto keep = std::visit(
      detail::Overload{
          [&](float arg) { return detail::select_by_property(x, arg); },
          [&](std::pair<float, float> arg) {
            return detail::select_by_property(x, arg.first, arg.second);
          },
          [&](xt::xtensor<float, 1> arg) {
            return detail::select_by_property(x, arg);
          },
          [&](std::pair<xt::xtensor<float, 1>, xt::xtensor<float, 1>> arg) {
            return detail::select_by_property(x, arg.first, arg.second);
          }},
      m_height.value());
  return xt::filter(peaks, keep);
}
template <typename T>
xtensor<size_t, 1>
find_peaks<T>::check_threshold(const xtensor<size_t, 1> &peaks,
                               const xtensor<T, 1> &x) {
  auto keep = std::visit(
      detail::Overload{
          [&](float arg) {
            auto [keep, left_thresholds, right_thresholds] =
                detail::select_by_peak_threshold(x, peaks, arg);
            return keep;
          },
          [&](std::pair<float, float> arg) {
            auto [keep, left_thresholds, right_thresholds] =
                detail::select_by_peak_threshold(x, peaks, arg.first,
                                                 arg.second);
            return keep;
          },
          [&](xt::xtensor<float, 1> arg) {
            auto [keep, left_thresholds, right_thresholds] =
                detail::select_by_peak_threshold(x, peaks, arg);
            return keep;
          },
          [&](std::pair<xt::xtensor<float, 1>, xt::xtensor<float, 1>> arg) {
            auto [keep, left_thresholds, right_thresholds] =
                detail::select_by_peak_threshold(x, peaks, arg.first,
                                                 arg.second);
            return keep;
          }},
      m_threshold.value());

  return xt::filter(peaks, keep);
}
template <typename T>
xtensor<size_t, 1>
find_peaks<T>::check_distance(const xtensor<size_t, 1> &peaks,
                              const xtensor<T, 1> &x) {
  auto keep = detail::select_by_peak_distance(
      peaks, xt::eval(xt::view(x, xt::keep(peaks))), m_distance.value());
  return xt::filter(peaks, keep);
}
template <typename T>
xtensor<size_t, 1>
find_peaks<T>::check_prominence(const xtensor<size_t, 1> &peaks,
                                const xtensor<T, 1> &x) {
  auto wlen_safe = detail::arg_wlen_as_expected(m_wlen);
  auto res = detail::peak_prominences(x, peaks, wlen_safe);
  auto keep = std::visit(
      detail::Overload{
          [&](float arg) {
            return detail::select_by_property(std::get<0>(res), arg);
          },
          [&](std::pair<float, float> arg) {
            return detail::select_by_property(std::get<0>(res), arg.first,
                                              arg.second);
          },
          [&](xt::xtensor<float, 1> arg) {
            return detail::select_by_property(std::get<0>(res), arg);
          },
          [&](std::pair<xt::xtensor<float, 1>, xt::xtensor<float, 1>> arg) {
            return detail::select_by_property(std::get<0>(res), arg.first,
                                              arg.second);
          }},
      m_prominence.value());
  return xt::filter(peaks, keep);
}
template <typename T>
xtensor<size_t, 1> find_peaks<T>::check_width(const xtensor<size_t, 1> &peaks,
                                              const xtensor<T, 1> &x) {
  auto widths = peak_widths(x, peaks, m_rel_height, m_wlen);
  auto keep = std::visit(
      detail::Overload{
          [&](float arg) {
            return detail::select_by_property(std::get<0>(widths), arg);
          },
          [&](std::pair<float, float> arg) {
            return detail::select_by_property(std::get<0>(widths), arg.first,
                                              arg.second);
          },
          [&](xt::xtensor<float, 1> arg) {
            return detail::select_by_property(std::get<0>(widths), arg);
          },
          [&](std::pair<xt::xtensor<float, 1>, xt::xtensor<float, 1>> arg) {
            return detail::select_by_property(std::get<0>(widths), arg.first,
                                              arg.second);
          }},
      m_width.value());
  return xt::filter(peaks, keep);
}
template <typename T>
xtensor<size_t, 1>
find_peaks<T>::check_plateau_size(const xtensor<size_t, 1> &peaks,
                                  const xtensor<size_t, 1> &left,
                                  const xtensor<size_t, 1> &right) {
  auto plateau_sizes = left - right + 1;
  auto keep = std::visit(
      detail::Overload{
          [&](float arg) {
            return detail::select_by_property(plateau_sizes, arg);
          },
          [&](std::pair<float, float> arg) {
            return detail::select_by_property(plateau_sizes, arg.first,
                                              arg.second);
          },
          [&](xt::xtensor<float, 1> arg) {
            return detail::select_by_property(plateau_sizes, arg);
          },
          [&](std::pair<xt::xtensor<float, 1>, xt::xtensor<float, 1>> arg) {
            return detail::select_by_property(plateau_sizes, arg.first,
                                              arg.second);
          }},
      m_plateau_size.value());
  return xt::filter(peaks, keep);
}

template <typename T> xtensor<T, 1> find_peaks<T>::operator()(xtensor<T, 1> x) {
  // check that we have only one dimention
  XTENSOR_ASSERT(x.shape().size() == 1);
  auto all_peaks = detail::local_maxima_1d(x);
  auto peaks = std::get<0>(all_peaks);
  if (m_plateau_size.has_value()) {
    peaks = check_plateau_size(peaks, std::get<1>(all_peaks),
                               std::get<2>(all_peaks));
  }
  if (m_height.has_value()) {
    peaks = check_height(peaks, xt::view(x, xt::keep(std::get<0>(all_peaks))));
  }
  if (m_threshold.has_value()) {
    peaks = check_threshold(peaks, x);
  }
  if (m_distance.has_value()) {
    peaks = check_distance(peaks, x);
  }
  if (m_prominence.has_value()) {
    peaks = check_prominence(peaks, x);
  }
  if (m_width.has_value()) {
    peaks = check_width(peaks, x);
  }
  return peaks;
}
} // namespace signal
} // namespace xt

#endif
