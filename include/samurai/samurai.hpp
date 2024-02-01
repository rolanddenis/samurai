// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#pragma once

#include <cstddef>

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
namespace mpi = boost::mpi;
#endif

namespace samurai
{

    inline void initialize([[maybe_unused]] int& argc, [[maybe_unused]] char**& argv)
    {
#ifdef SAMURAI_WITH_MPI
        MPI_Init(&argc, &argv);
#endif
    }

    inline void initialize()
    {
#ifdef SAMURAI_WITH_MPI
        MPI_Init(nullptr, nullptr);
#endif
    }

    inline void finalize()
    {
#ifdef SAMURAI_WITH_MPI
        MPI_Finalize();
#endif
    }

    template <class TValue, class TIndex>
    struct Interval;

    template <std::size_t dim_, class TInterval, std::size_t max_size_, std::size_t Topology = ((1ul << dim_) - 1)>
    class CellArray;

    template <class D, class Config, std::size_t Topology = ((1ul << Config::dim) - 1)>
    class Mesh_base;

    template <class F, class... CT>
    class subset_operator;

    template <std::size_t Dim, class TInterval, std::size_t Topology = (1ul << Dim) - 1>
    class LevelCellList;

    template <std::size_t Dim, class TInterval, std::size_t Topology = (1ul << Dim) - 1>
    class LevelCellArray;

    template <class Config>
    class UniformMesh;

    template <class mesh_t, class value_t, std::size_t size, bool SOA>
    class Field;
}
