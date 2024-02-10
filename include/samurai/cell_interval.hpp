// Copyright 2024 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <cassert>

#include <xtensor/xfixed.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include "cell.hpp"
#include "kspace/utils.hpp"

namespace samurai
{
    /** @class CellInterval
     *  @brief Define an interval of mesh cells in multi dimensions.
     *
     *  A cell interval is defined by its level, the interval of integer coordinates
     *  for the first dimension, its integer coordinates for the remaining dimensions
     *  and its index in the data array.
     *
     *  @tparam dim_ The dimension of the cell.
     *  @tparam TInterval The type of the interval.
     *  @tparam Topology    Cell topology as an integer (by default a fully open cell of full dimension)
     */
    template <std::size_t dim_, class TInterval, std::size_t Topology = (1ul << dim_) - 1>
    struct CellInterval
    {
        static constexpr std::size_t dim      = dim_;
        static constexpr std::size_t topology = Topology;

        using interval_t    = TInterval;
        using value_t       = typename interval_t::value_t;
        using index_t       = typename interval_t::index_t;
        using indices_t     = xt::xtensor_fixed<value_t, xt::xshape<dim - 1>>;
        using all_indices_t = xt::xtensor<value_t, 2>;
        using coords_t      = xt::xtensor<double, 2>;
        using cell_t        = Cell<dim, interval_t, topology>;

        CellInterval()                    = default;
        CellInterval(CellInterval const&) = default;
        CellInterval(CellInterval&&)      = default;
        ~CellInterval()                   = default;

        CellInterval(std::size_t level, interval_t const& interval, indices_t const& indices);

        /// Number of cells in the interval
        std::size_t size() const;

        /// Access each cell of the interval
        cell_t operator[](std::size_t i) const;

        /// All indices spawn by the cell interval
        all_indices_t all_indices() const;

        /// The center of the cell
        coords_t center() const;
        xt::xtensor<double, 1> center(std::size_t i) const;

        /// The corner of the cell (position of the point of same index)
        coords_t corner() const;
        xt::xtensor<double, 1> corner(std::size_t i) const;

        void to_stream(std::ostream& os) const;

        /// The level of the cell interval.
        std::size_t level = 0;

        /// The integer interval of the coordinates along the first dimension
        interval_t interval;

        /// The integer coordinates along the remaining dimensions
        indices_t indices;

        /// The length of a cell of full dimension.
        double length = 0;
    };

    template <std::size_t dim_, class TInterval, std::size_t Topology>
    inline CellInterval<dim_, TInterval, Topology>::CellInterval(std::size_t level, interval_t const& interval, indices_t const& indices)
        : level(level)
        , interval(interval)
        , indices(indices)
        , length(cell_length(level))
    {
    }

    template <std::size_t dim_, class TInterval, std::size_t Topology>
    inline std::size_t CellInterval<dim_, TInterval, Topology>::size() const
    {
        return interval.size();
    }

    template <std::size_t dim_, class TInterval, std::size_t Topology>
    inline auto CellInterval<dim_, TInterval, Topology>::operator[](std::size_t pos) const -> cell_t
    {
        assert((pos < size()) && "Out of bound position");
        const auto i = interval.start + static_cast<value_t>(pos) * interval.step;
        return cell_t{level, i, indices, interval.index + i};
    }

    template <std::size_t dim_, class TInterval, std::size_t Topology>
    inline auto CellInterval<dim_, TInterval, Topology>::all_indices() const -> all_indices_t
    {
        typename all_indices_t::shape_type shape = {dim, size()};
        all_indices_t all_indices_(shape);
        xt::view(all_indices_, 0, xt::all())      = xt::arange<value_t>(interval.start, interval.end, interval.step);
        xt::view(all_indices_, xt::range(1, dim)) = xt::view(indices, xt::all(), xt::newaxis());
        return all_indices_;
    }

    template <std::size_t dim_, class TInterval, std::size_t Topology>
    inline auto CellInterval<dim_, TInterval, Topology>::center() const -> coords_t
    {
        return length * (all_indices() + 0.5 * xt::view(topology_as_xtensor<dim_>(Topology), xt::all(), xt::newaxis()));
    }

    template <std::size_t dim_, class TInterval, std::size_t Topology>
    inline xt::xtensor<double, 1> CellInterval<dim_, TInterval, Topology>::center(std::size_t i) const
    {
        return length * (xt::view(all_indices(), i, xt::all()) + 0.5 * topology_as_xtensor<dim_>(Topology)[i]);
    }

    template <std::size_t dim_, class TInterval, std::size_t Topology>
    inline auto CellInterval<dim_, TInterval, Topology>::corner() const -> coords_t
    {
        return length * all_indices();
    }

    template <std::size_t dim_, class TInterval, std::size_t Topology>
    inline xt::xtensor<double, 1> CellInterval<dim_, TInterval, Topology>::corner(std::size_t i) const
    {
        return length * xt::view(all_indices(), i, xt::all());
    }

    template <std::size_t dim_, class TInterval, std::size_t Topology>
    inline void CellInterval<dim_, TInterval, Topology>::to_stream(std::ostream& os) const
    {
        os << "Cell -> level: " << level << " interval: " << interval << " indices: " << indices << " index: " << index
           << " topology: " << topology_as_string<dim_>(topology);
    }

    template <std::size_t dim, class TInterval, std::size_t Topology>
    inline std::ostream& operator<<(std::ostream& out, const CellInterval<dim, TInterval, Topology>& ci)
    {
        ci.to_stream(out);
        return out;
    }

    template <std::size_t dim_, class TInterval, std::size_t Topology>
    inline bool operator==(const CellInterval<dim_, TInterval, Topology>& lhs, const CellInterval<dim_, TInterval, Topology>& rhs)
    {
        return lhs.level == rhs.level && lhs.interval == rhs.interval && lhs.indices == rhs.indices;
    }

} // namespace samurai
