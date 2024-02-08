#include <gtest/gtest.h>

#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/cell.hpp>
#include <samurai/cell_interval.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/mesh.hpp>

namespace samurai
{
    TEST(cell_interval, bracket_operator)
    {
        constexpr std::size_t dim      = 3;
        constexpr std::size_t topology = 5;
        using interval_t               = typename CellArray<dim>::interval_t;
        using cell_interval_t          = CellInterval<dim, interval_t, topology>;
        using cell_t                   = typename cell_interval_t::cell_t;

        interval_t interval{2, 10, -2};
        CellInterval<dim, interval_t, topology> ci(1, interval, {2, 3});

        EXPECT_EQ(ci.size(), interval.size());
        EXPECT_EQ(ci[3], cell_t(1, interval.start + 3, {2, 3}, 3));
    }

    TEST(cell_interval, all_indices)
    {
        constexpr std::size_t dim      = 3;
        constexpr std::size_t topology = 5;
        using interval_t               = typename CellArray<dim>::interval_t;
        using cell_interval_t          = CellInterval<dim, interval_t, topology>;
        using all_indices_t            = typename cell_interval_t::all_indices_t;

        interval_t interval{2, 10, -2};
        CellInterval<dim, interval_t, topology> ci(1, interval, {2, 3});
        all_indices_t target = {
            {2, 3, 4, 5, 6, 7, 8, 9},
            {2, 2, 2, 2, 2, 2, 2, 2},
            {3, 3, 3, 3, 3, 3, 3, 3}
        };
        EXPECT_EQ(ci.all_indices(), target);
    }

    TEST(cell_interval, corner)
    {
        constexpr std::size_t dim      = 3;
        constexpr std::size_t topology = 5;
        using interval_t               = typename CellArray<dim>::interval_t;
        using cell_interval_t          = CellInterval<dim, interval_t, topology>;
        using coords_t                 = typename cell_interval_t::coords_t;

        interval_t interval{2, 10, -2};
        CellInterval<dim, interval_t, topology> ci(1, interval, {2, 3});
        coords_t target = {
            {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5},
            {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
            {1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5}
        };

        EXPECT_EQ(ci.corner(), target);
        EXPECT_EQ(ci.corner(1), xt::view(target, 1, xt::all()));
    }

    TEST(cell_interval, center)
    {
        constexpr std::size_t dim      = 3;
        constexpr std::size_t topology = 5;
        using interval_t               = typename CellArray<dim>::interval_t;
        using cell_interval_t          = CellInterval<dim, interval_t, topology>;
        using coords_t                 = typename cell_interval_t::coords_t;

        interval_t interval{2, 10, -2};
        CellInterval<dim, interval_t, topology> ci(1, interval, {2, 3});
        coords_t target = {
            {1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75},
            {1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00},
            {1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75}
        };

        EXPECT_EQ(ci.center(), target);
        EXPECT_EQ(ci.center(1), xt::view(target, 1, xt::all()));
    }

    TEST(cell_interval, algorithm)
    {
        constexpr size_t dim = 2;
        using config_t       = samurai::MRConfig<dim>;
        using mesh_t         = MRMesh<config_t>;
        using cell_list_t    = typename mesh_t::cl_type;

        auto f = [](auto const& coords)
        {
            return xt::cos(xt::view(coords, 0, xt::all())) * xt::sin(xt::view(coords, 1, xt::all()));
        };

        cell_list_t cell_list;
        cell_list[1][{1}].add_interval({2, 5});
        cell_list[2][{5}].add_interval({-2, 8});
        cell_list[2][{5}].add_interval({9, 10});
        cell_list[2][{6}].add_interval({10, 12});

        mesh_t mesh(cell_list, 0, 3);

        auto ref_field = samurai::make_field<double, 1>("ref", mesh);
        samurai::for_each_cell(mesh,
                               [&](auto const& cell)
                               {
                                   ref_field[cell] = f(cell.center())(0);
                               });

        auto test1_field = samurai::make_field<double, 1>("test", mesh);
        samurai::for_each_cell_interval(mesh,
                                        [&](auto const& ci)
                                        {
                                            test1_field(ci) = f(ci.center());
                                        });
        EXPECT_EQ(ref_field, test1_field);

        auto test2_field = samurai::make_field<double, 1>("test", mesh);
        samurai::for_each_cell_interval(mesh,
                                        [&](auto const& ci)
                                        {
                                            samurai::for_each_cell(ci,
                                                                   [&](auto const& cell)
                                                                   {
                                                                       test2_field[cell] = f(cell.center())(0);
                                                                   });
                                        });
        EXPECT_EQ(ref_field, test2_field);
    }
}
