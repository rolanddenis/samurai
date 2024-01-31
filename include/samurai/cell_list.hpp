// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <array>

#include <fmt/color.h>

#include "level_cell_list.hpp"
#include "samurai_config.hpp"

namespace samurai
{

    /////////////////////////
    // CellList definition //
    /////////////////////////

    template <std::size_t dim_,
              class TInterval       = default_config::interval_t,
              std::size_t max_size_ = default_config::max_level,
              std::size_t Topology  = (1ul << dim_) - 1>
    class CellList
    {
      public:

        static constexpr auto dim      = dim_;
        static constexpr auto max_size = max_size_;
        static constexpr auto topology = Topology;

        using lcl_type = LevelCellList<dim, TInterval, Topology>;

        CellList();

        const lcl_type& operator[](std::size_t i) const;
        lcl_type& operator[](std::size_t i);

        void to_stream(std::ostream& os) const;

      private:

        std::array<lcl_type, max_size + 1> m_cells;
    };

    /////////////////////////////
    // CellList implementation //
    /////////////////////////////

    /**
     * Default contructor which sets the level for each LevelCellArray.
     */
    template <std::size_t dim_, class TInterval, std::size_t max_size_, std::size_t Topology>
    inline CellList<dim_, TInterval, max_size_, Topology>::CellList()
    {
        for (std::size_t level = 0; level <= max_size; ++level)
        {
            m_cells[level] = {level};
        }
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_, std::size_t Topology>
    inline auto CellList<dim_, TInterval, max_size_, Topology>::operator[](std::size_t i) const -> const lcl_type&
    {
        return m_cells[i];
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_, std::size_t Topology>
    inline auto CellList<dim_, TInterval, max_size_, Topology>::operator[](std::size_t i) -> lcl_type&
    {
        return m_cells[i];
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_, std::size_t Topology>
    inline void CellList<dim_, TInterval, max_size_, Topology>::to_stream(std::ostream& os) const
    {
        for (std::size_t level = 0; level <= max_size; ++level)
        {
            os << fmt::format(fg(fmt::color::crimson) | fmt::emphasis::bold, "Level {}\n", level);
            m_cells[level].to_stream(os);
            os << "\n";
        }
    }

    template <std::size_t dim_, class TInterval, std::size_t max_size_, std::size_t Topology>
    inline std::ostream& operator<<(std::ostream& out, const CellList<dim_, TInterval, max_size_, Topology>& cell_list)
    {
        cell_list.to_stream(out);
        return out;
    }
} // namespace samurai
