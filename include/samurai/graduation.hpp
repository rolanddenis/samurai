#pragma once

#include "algorithm.hpp"
#include "field.hpp"
#include "stencil.hpp"
#include "subset/subset_op.hpp"

namespace samurai
{
    template <class Mesh>
    bool is_graduated(const Mesh& mesh)
    {
        constexpr auto stencil = star_stencil_kspace<Mesh::dim>();

        bool cond = true;

        std::size_t min_level = mesh.min_level();
        std::size_t max_level = mesh.max_level();

        for (std::size_t level = min_level + 2; level <= max_level; ++level)
        {
            for (std::size_t level_below = min_level; level_below < level - 1; ++level_below)
            {
                stencil.foreach(
                    [&] (auto c)
                    {
                        auto s = c.apply([] (auto... i) { return xt::xtensor_fixed<int, xt::xshape<Mesh::dim>>{static_cast<int>(i.indexShift())...}; });
                        auto set = intersection(translate(mesh[level], s), mesh[level_below]).on(level_below);
                        set(
                            [&cond](const auto&, const auto&)
                            {
                                cond = false;
                            });
                    }
                );
                if (!cond)
                {
                    return false;
                }             }
        }
        return true;
    }


    template <class Mesh, class TStencil>
    bool is_graduated(const Mesh& mesh, const TStencil stencil)
    {
        bool cond = true;

        std::size_t min_level = mesh.min_level();
        std::size_t max_level = mesh.max_level();

        for (std::size_t level = min_level + 2; level <= max_level; ++level)
        {
            for (std::size_t level_below = min_level; level_below < level - 1; ++level_below)
            {
                for (std::size_t is = 0; is < stencil.shape()[0]; ++is)
                {
                    auto s   = xt::view(stencil, is);
                    auto set = intersection(translate(mesh[level], s), mesh[level_below]).on(level_below);
                    set(
                        [&cond](const auto&, const auto&)
                        {
                            cond = false;
                        });
                    if (!cond)
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    template <class Mesh>
    void make_graduation(Mesh& mesh)
    {
        constexpr auto stencil = star_stencil_kspace<Mesh::dim>();

        static constexpr std::size_t dim = Mesh::dim;
        using cl_type                    = typename Mesh::cl_type;

        std::size_t min_level = mesh.min_level();
        std::size_t max_level = mesh.max_level();

        auto tag = make_field<bool, 1>("tag", mesh);

        while (true)
        {
            tag.resize();
            tag.fill(false);

            for (std::size_t level = min_level + 2; level <= max_level; ++level)
            {
                for (std::size_t level_below = min_level; level_below < level - 1; ++level_below)
                {
                    stencil.foreach(
                        [&] (auto c)
                        {
                            auto s = c.apply([] (auto... i) { return xt::xtensor_fixed<int, xt::xshape<dim>>{static_cast<int>(i.indexShift())...}; });
                            auto set = intersection(translate(mesh[level], s), mesh[level_below]).on(level_below);
                            set(
                                [&](const auto& i, const auto& index)
                                {
                                    tag(level_below, i, index) = true;
                                });
                        }
                    );
                }
            }

            cl_type cl;
            for_each_interval(mesh,
                              [&](std::size_t level, const auto& interval, const auto& index_yz)
                              {
                                  auto itag = interval.start + interval.index;
                                  for (auto i = interval.start; i < interval.end; ++i, ++itag)
                                  {
                                      if (tag[itag])
                                      {
                                          static_nested_loop<dim - 1, 0, 2>(
                                              [&](auto s)
                                              {
                                                  auto index = 2 * index_yz + s;
                                                  cl[level + 1][index].add_interval({2 * i, 2 * i + 2});
                                              });
                                      }
                                      else
                                      {
                                          cl[level][index_yz].add_point(i);
                                      }
                                  }
                              });
            Mesh new_ca = {cl, true};

            if (new_ca == mesh)
            {
                break;
            }

            std::swap(mesh, new_ca);
        }
    }

    template <class Mesh, class TStencil>
    void make_graduation(Mesh& mesh, const TStencil stencil)
    {
        static constexpr std::size_t dim = Mesh::dim;
        using cl_type                    = typename Mesh::cl_type;

        std::size_t min_level = mesh.min_level();
        std::size_t max_level = mesh.max_level();

        auto tag = make_field<bool, 1>("tag", mesh);

        while (true)
        {
            tag.resize();
            tag.fill(false);

            for (std::size_t level = min_level + 2; level <= max_level; ++level)
            {
                for (std::size_t level_below = min_level; level_below < level - 1; ++level_below)
                {
                    for (std::size_t is = 0; is < stencil.shape()[0]; ++is)
                    {
                        auto s   = xt::view(stencil, is);
                        auto set = intersection(translate(mesh[level], s), mesh[level_below]).on(level_below);
                        set(
                            [&](const auto& i, const auto& index)
                            {
                                tag(level_below, i, index) = true;
                            });
                    }
                }
            }

            cl_type cl;
            for_each_interval(mesh,
                              [&](std::size_t level, const auto& interval, const auto& index_yz)
                              {
                                  auto itag = interval.start + interval.index;
                                  for (auto i = interval.start; i < interval.end; ++i, ++itag)
                                  {
                                      if (tag[itag])
                                      {
                                          static_nested_loop<dim - 1, 0, 2>(
                                              [&](auto s)
                                              {
                                                  auto index = 2 * index_yz + s;
                                                  cl[level + 1][index].add_interval({2 * i, 2 * i + 2});
                                              });
                                      }
                                      else
                                      {
                                          cl[level][index_yz].add_point(i);
                                      }
                                  }
                              });
            Mesh new_ca = {cl, true};

            if (new_ca == mesh)
            {
                break;
            }

            std::swap(mesh, new_ca);
        }
    }
}
