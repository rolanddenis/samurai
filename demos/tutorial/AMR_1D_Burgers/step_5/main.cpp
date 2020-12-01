
#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>

#include "../step_3/init_sol.hpp"
#include "../step_3/mesh.hpp"

#include "../step_4/AMR_criterion.hpp"
#include "../step_4/update_mesh.hpp"

#include "update_ghost.hpp"
#include "make_graduation.hpp"

/**
 * What will we learn ?
 * ====================
 *
 * - update the ghost cells
 * - make the graduation of the mesh
 *
 */

int main()
{
    constexpr std::size_t dim = 1;
    std::size_t start_level = 8;
    std::size_t min_level = 2;
    std::size_t max_level = 8;

    samurai::Box<double, dim> box({-3}, {3});
    Mesh<MeshConfig<dim>> mesh(box, start_level, min_level, max_level);

    auto phi = init_sol(mesh);

    std::size_t i_adapt = 0;
    while(i_adapt < (max_level - min_level + 1))
    {
        auto tag = samurai::make_field<std::size_t, 1>("tag", mesh);

        update_ghost(phi);                                             // <--------------------------------
        AMR_criterion(phi, tag);
        make_graduation(tag);                                          // <--------------------------------

        samurai::save(fmt::format("step_5_criterion-{}", i_adapt++), mesh, phi, tag);

        if (update_mesh(phi, tag))
        {
            break;
        };
    }

    auto level = samurai::make_field<int, 1>("level", mesh);
    samurai::for_each_interval(mesh[MeshID::cells], [&](std::size_t l, const auto& i, auto)
    {
        level(l, i) = l;
    });
    samurai::save("step_5", mesh, phi, level);

    return 0;
}

