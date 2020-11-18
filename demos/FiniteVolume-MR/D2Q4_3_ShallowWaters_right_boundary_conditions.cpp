#include <math.h>
#include <vector>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

#include <samurai/samurai.hpp>
#include "coarsening.hpp"
#include "refinement.hpp"
#include "criteria.hpp"
#include "prediction_map_2d.hpp"


const double lambda = 2.8;

const double gravity = 1.;
const double height = 0.5;

const double sigma_h_x  = 5.e-2;
const double sigma_h_xy = 0.5;
const double sigma_q_x  = 7.e-2;
const double sigma_q_xy = 0.5;

const double s_h_x  = 1./(0.5 + sigma_h_x);
const double s_h_xy = 1./(0.5 + sigma_h_xy);
const double s_q_x  = 1./(0.5 + sigma_q_x);
const double s_q_xy = 1./(0.5 + sigma_q_xy);

/// Timer used in tic & toc
auto tic_timer = std::chrono::high_resolution_clock::now();

/// Launching the timer
void tic()
{
    tic_timer = std::chrono::high_resolution_clock::now();
}


/// Stopping the timer and returning the duration in seconds
double toc()
{
    const auto toc_timer = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> time_span = toc_timer - tic_timer;
    return time_span.count();
}





template<class coord_index_t>
auto compute_prediction(std::size_t min_level, std::size_t max_level)
{
    coord_index_t i = 0, j = 0;
    std::vector<std::vector<prediction_map<coord_index_t>>> data(max_level-min_level+1);

    for(std::size_t k=0; k<max_level-min_level+1; ++k)
    {
        int size = (1<<k);
        data[k].resize(8);
        for (int l = 0; l < size; ++l)
        {
            // Be careful, there is no sign on this fluxes

            // Along x (vertical edge)
            data[k][0] += prediction(k, i*size - 1, j*size + l); // In W
            data[k][1] += prediction(k, (i+1)*size - 1, j*size + l); // Out E
            data[k][2] += prediction(k, i*size + l, j*size - 1); // In S
            data[k][3] += prediction(k, i*size + l, (j+1)*size - 1); // Out N

            // Along y (horizontal edge)
            data[k][4] += prediction(k, (i+1)*size, j*size + l); // In E
            data[k][5] += prediction(k, i*size, j*size + l); // Out W
            data[k][6] += prediction(k, i*size + l, (j+1)*size); // In N
            data[k][7] += prediction(k, i*size + l, j*size); // Out S
        }
    }
    return data;
}


template<class Config>
auto init_f(samurai::Mesh<Config> &mesh, double t)
{
    constexpr std::size_t nvel = 12; // 4 * 3
    samurai::BC<2> bc{ {{ {samurai::BCType::neumann, 0.0},
                       {samurai::BCType::neumann, 0.0},
                       {samurai::BCType::neumann, 0.0},
                       {samurai::BCType::neumann, 0.0}
                    }} };

    samurai::Field<Config, double, nvel> f("f", mesh, bc);
    f.array().fill(0);

    mesh.for_each_cell([&](auto &cell) {
        auto center = cell.center();
        auto x = center[0];
        auto y = center[1];

        double x0 = 0.4;
        double y0 = 0.4;
        double radius = 0.2;

        double x02 = -0.6;
        double y02 = -0.6;
        double radius2 = 0.1;

        double h = 1. + ((pow(x - x0, 2.) + pow(y - y0, 2.) < pow(radius, 2.)) ? height : 0); //+
                        // 2./3.*((pow(x - x02, 2.) + pow(y - y02, 2.) < pow(radius2, 2.)) ? height : 0);


        // h = 1. + height * exp(-(pow(x-0.0, 2.)+pow(y-0.0, 2.))/(2 * pow(2.3548*0.03/2, 2.)));


        double qx = 0.0; // x-momentum
        double qy = 0.0; // y-momentum


        // Conserved momenti
        double m0_0 = h;
        double m1_0 = qx;
        double m2_0 = qy;

        // Non conserved at equilibrium
        double m0_1 = m1_0;
        double m0_2 = m2_0;
        double m0_3 = 0.0;

        double m1_1 = m1_0*m1_0/m0_0 + 0.5*gravity*m0_0*m0_0;
        double m1_2 = m1_0*m2_0/m0_0;
        double m1_3 = 0.0;

        double m2_1 = m1_0*m2_0/m0_0;
        double m2_2 = m2_0*m2_0/m0_0 + 0.5*gravity*m0_0*m0_0;
        double m2_3 = 0.0;

        // We come back to the distributions
        f[cell][0] = .25 * m0_0 + .5/lambda * (m0_1)                    + .25/(lambda*lambda) * m0_3;
        f[cell][1] = .25 * m0_0                    + .5/lambda * (m0_2) - .25/(lambda*lambda) * m0_3;
        f[cell][2] = .25 * m0_0 - .5/lambda * (m0_1)                    + .25/(lambda*lambda) * m0_3;
        f[cell][3] = .25 * m0_0                    - .5/lambda * (m0_2) - .25/(lambda*lambda) * m0_3;

        f[cell][4] = .25 * m1_0 + .5/lambda * (m1_1)                    + .25/(lambda*lambda) * m1_3;
        f[cell][5] = .25 * m1_0                    + .5/lambda * (m1_2) - .25/(lambda*lambda) * m1_3;
        f[cell][6] = .25 * m1_0 - .5/lambda * (m1_1)                    + .25/(lambda*lambda) * m1_3;
        f[cell][7] = .25 * m1_0                    - .5/lambda * (m1_2) - .25/(lambda*lambda) * m1_3;

        f[cell][8]  = .25 * m2_0 + .5/lambda * (m2_1)                    + .25/(lambda*lambda) * m2_3;
        f[cell][9]  = .25 * m2_0                    + .5/lambda * (m2_2) - .25/(lambda*lambda) * m2_3;
        f[cell][10] = .25 * m2_0 - .5/lambda * (m2_1)                    + .25/(lambda*lambda) * m2_3;
        f[cell][11] = .25 * m2_0                    - .5/lambda * (m2_2) - .25/(lambda*lambda) * m2_3;

    });

    return f;
}



// I do many separate functions because the return type
// is not necessarely the same between directions and I want to avoid
// using a template, which indeed comes back to the same than this.
template<class Mesh>
auto get_adjacent_boundary_east(Mesh & mesh, std::size_t level, samurai::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};

    std::size_t coeff = 1 << (mesh.max_level() - level); // When we are not at the finest level, we must translate more

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -coeff * xp)),
                                              difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -coeff * yp))), // Removing NE
                                   difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), coeff * yp))), // Removing SE
                        mesh[type][level]);//.on(level);
}
template<class Mesh>
auto get_adjacent_boundary_north(Mesh & mesh, std::size_t level, samurai::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};

    std::size_t coeff = 1 << (mesh.max_level() - level);

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -coeff * yp)),
                                              difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -coeff * xp))), // Removing NE
                                   difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), coeff * xp))), // Removing NW
                        mesh[type][level]);//.on(level);
}
template<class Mesh>
auto get_adjacent_boundary_west(Mesh & mesh, std::size_t level, samurai::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};

    std::size_t coeff = 1 << (mesh.max_level() - level);

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), coeff * xp)),
                                              difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -coeff * yp))), // Removing NW
                                   difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), coeff * yp))), // Removing SW
                        mesh[type][level]);//.on(level);
}
template<class Mesh>
auto get_adjacent_boundary_south(Mesh & mesh, std::size_t level, samurai::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};

    std::size_t coeff = 1 << (mesh.max_level() - level);

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), coeff * yp)),
                                              difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -coeff * xp))), // Removing SE
                                   difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), coeff * xp))), // Removing SW
                        mesh[type][level]);//.on(level);
}
template<class Mesh>
auto get_adjacent_boundary_northeast(Mesh & mesh, std::size_t level, samurai::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};
    const xt::xtensor_fixed<int, xt::xshape<2>> d11{1, 1};

    std::size_t coeff = 1 << (mesh.max_level() - level);

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -coeff * d11)),
                                              translate(mesh.initial_mesh(), - coeff * yp)), // Removing vertical strip
                                   translate(mesh.initial_mesh(), - coeff * xp)), // Removing horizontal strip
                        mesh[type][level]);//.on(level);
}
template<class Mesh>
auto get_adjacent_boundary_northwest(Mesh & mesh, std::size_t level, samurai::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};
    const xt::xtensor_fixed<int, xt::xshape<2>> d1m1{1, -1};

    std::size_t coeff = 1 << (mesh.max_level() - level);

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), coeff * d1m1)),
                                              translate(mesh.initial_mesh(), - coeff * yp)), // Removing vertical strip
                                   translate(mesh.initial_mesh(), coeff * xp)), // Removing horizontal strip
                        mesh[type][level]);//.on(level);
}
template<class Mesh>
auto get_adjacent_boundary_southwest(Mesh & mesh, std::size_t level, samurai::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};
    const xt::xtensor_fixed<int, xt::xshape<2>> d11{1, 1};

    std::size_t coeff = 1 << (mesh.max_level() - level);

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), coeff * d11)),
                                              translate(mesh.initial_mesh(), coeff * yp)), // Removing vertical strip
                                   translate(mesh.initial_mesh(), coeff * xp)), // Removing horizontal strip
                        mesh[type][level]);//.on(level);
}
template<class Mesh>
auto get_adjacent_boundary_southeast(Mesh & mesh, std::size_t level, samurai::MeshType type)
{
    const xt::xtensor_fixed<int, xt::xshape<2>> xp{1, 0};
    const xt::xtensor_fixed<int, xt::xshape<2>> yp{0, 1};
    const xt::xtensor_fixed<int, xt::xshape<2>> d1m1{1, -1};

    std::size_t coeff = 1 << (mesh.max_level() - level);

    return intersection(difference(difference(difference(mesh.initial_mesh(), translate(mesh.initial_mesh(), -coeff * d1m1)),
                                              translate(mesh.initial_mesh(), coeff * yp)), // Removing vertical strip
                                   translate(mesh.initial_mesh(), -coeff * xp)), // Removing horizontal strip
                        mesh[type][level]);//.on(level);
}





// We have to average only the fluxes
template<class Field, class pred>
void one_time_step_overleaves_corrected(Field &f, const pred& pred_coeff, std::size_t iter)
{
    constexpr std::size_t nvel = Field::size;
    using coord_index_t = typename Field::coord_index_t;

    auto mesh = f.mesh();
    auto max_level = mesh.max_level();
    auto min_level = mesh.min_level();

    std::cout<<std::endl<<"[+] Projecting"<<std::flush;
    samurai::mr_projection(f);
    std::cout<<std::endl<<"[+] Updating BC"<<std::flush;
    f.update_bc(); // It is important to do so
    std::cout<<std::endl<<"[+] Predicting"<<std::flush;
    samurai::mr_prediction(f);
    if (min_level != max_level) {
        std::cout<<std::endl<<"[+] Predicting overleaves"<<std::flush;
        samurai::mr_prediction_overleaves(f);
    }


    Field new_f{"new_f", mesh};
    new_f.array().fill(0.);

    // This stored the fluxes computed at the level
    // of the overleaves
    Field fluxes{"fluxes", mesh};
    fluxes.array().fill(0.);


    Field advected{"advected", mesh};
    advected.array().fill(0.);

    for (std::size_t level = 0; level <= max_level; ++level)
    {

        if (level == max_level) {

            std::cout<<std::endl<<"[+] Advecting at finest"<<std::flush;

            std::cout<<std::endl<<"[=] East"<<std::flush;
            auto leaves_east = get_adjacent_boundary_east(mesh, max_level, samurai::MeshType::cells);
            leaves_east.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                // Mass - bounce back
                advected(2, level, k, h) =  f(0, level, k    , h    );

                // Momentum - anti bounce back
                for (int scheme_n = 1; scheme_n < 3; ++scheme_n)    {
                    advected(2 + 4 * scheme_n, level, k, h) =  -1. * f(0 + 4 * scheme_n, level, k    , h    );
                }

                // Standard
                for (int scheme_n = 0; scheme_n < 3; ++scheme_n)    {
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k - 1, h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h - 1);
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h + 1);
                }
            });

            std::cout<<std::endl<<"[=] North"<<std::flush;
            auto leaves_north = get_adjacent_boundary_north(mesh, max_level, samurai::MeshType::cells);
            leaves_north.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                advected(3, level, k, h) =  f(1, level, k,     h    );

                for (int scheme_n = 1; scheme_n < 3; ++scheme_n)    {
                    advected(3 + 4 * scheme_n, level, k, h) =  -1. * f(1 + 4 * scheme_n, level, k,     h    );
                }

                for (int scheme_n = 0; scheme_n < 3; ++scheme_n)    {
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k - 1, h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h - 1);
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k + 1 ,h    );
                }
            });

            std::cout<<std::endl<<"[=] NorthEast"<<std::flush;
            auto leaves_northeast = get_adjacent_boundary_northeast(mesh, max_level, samurai::MeshType::cells);
            leaves_northeast.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                advected(2, level, k, h) =  f(0, level, k    , h    );
                advected(3, level, k, h) =  f(1, level, k,     h    );

                for (int scheme_n = 1; scheme_n < 3; ++scheme_n)    {
                    advected(2 + 4 * scheme_n, level, k, h) =  -1. * f(0 + 4 * scheme_n, level, k    , h    );
                    advected(3 + 4 * scheme_n, level, k, h) =  -1. * f(1 + 4 * scheme_n, level, k,     h    );
                }

                for (int scheme_n = 0; scheme_n < 3; ++scheme_n)    {
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k - 1, h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h - 1);
                }
            });

            std::cout<<std::endl<<"[=] West"<<std::flush;
            auto leaves_west = get_adjacent_boundary_west(mesh, max_level, samurai::MeshType::cells);
            leaves_west.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                advected(0, level, k, h) =  f(2, level, k    , h    );

                for (int scheme_n = 1; scheme_n < 3; ++scheme_n)    {
                    advected(0 + 4 * scheme_n, level, k, h) = -1. * f(2 + 4 * scheme_n, level, k    , h    );
                }

                for (int scheme_n = 0; scheme_n < 3; ++scheme_n)    {
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h - 1);
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k + 1, h    );
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h + 1);
                }
            });

            std::cout<<std::endl<<"[=] NorthWest"<<std::flush;
            auto leaves_northwest = get_adjacent_boundary_northwest(mesh, max_level, samurai::MeshType::cells);
            leaves_northwest.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                advected(0, level, k, h) =  f(2, level, k    , h    );
                advected(3, level, k, h) =  f(1, level, k,     h    );

                for (int scheme_n = 1; scheme_n < 3; ++scheme_n)    {
                    advected(0 + 4 * scheme_n, level, k, h) =  -1. * f(2 + 4 * scheme_n, level, k    , h    );
                    advected(3 + 4 * scheme_n, level, k, h) =  -1. * f(1 + 4 * scheme_n, level, k,     h    );
                }

                for (int scheme_n = 0; scheme_n < 3; ++scheme_n)    {
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h - 1);
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k + 1, h    );
                }
            });

            std::cout<<std::endl<<"[=] South"<<std::flush;
            auto leaves_south = get_adjacent_boundary_south(mesh, max_level, samurai::MeshType::cells);
            leaves_south.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                advected(1, level, k, h) =  f(3, level, k,     h    );

                for (int scheme_n = 1; scheme_n < 3; ++scheme_n)    {
                    advected(1 + 4 * scheme_n, level, k, h) =  -1. * f(3 + 4 * scheme_n, level, k,     h    );
                }


                for (int scheme_n = 0; scheme_n < 3; ++scheme_n)    {
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k - 1, h    );
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k + 1, h    );
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h + 1);
                }
            });

            std::cout<<std::endl<<"[=] SouthWest"<<std::flush;
            auto leaves_southwest = get_adjacent_boundary_southwest(mesh, max_level, samurai::MeshType::cells);
            leaves_southwest.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y


                advected(0, level, k, h) =  f(2, level, k    , h    );
                advected(1, level, k, h) =  f(3, level, k,     h    );

                for (int scheme_n = 1; scheme_n < 3; ++scheme_n)    {
                    advected(0 + 4 * scheme_n, level, k, h) =  -1. * f(2 + 4 * scheme_n, level, k    , h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  -1. * f(3 + 4 * scheme_n, level, k,     h    );
                }

                for (int scheme_n = 0; scheme_n < 3; ++scheme_n)    {

                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k + 1, h    );
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h + 1);
                }
            });

            std::cout<<std::endl<<"[=] SouthEast"<<std::flush;
            auto leaves_southeast = get_adjacent_boundary_southeast(mesh, max_level, samurai::MeshType::cells);
            leaves_southeast.on(max_level)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                advected(1, level, k, h) =  f(3, level, k,     h    );
                advected(2, level, k, h) =  f(0, level, k    , h    );

                for (int scheme_n = 1; scheme_n < 3; ++scheme_n)    {
                    advected(1 + 4 * scheme_n, level, k, h) =  -1. * f(3 + 4 * scheme_n, level, k,     h    );
                    advected(2 + 4 * scheme_n, level, k, h) =  -1. * f(0 + 4 * scheme_n, level, k    , h    );
                }

                for (int scheme_n = 0; scheme_n < 3; ++scheme_n)    {
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k - 1, h    );

                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h + 1);
                }
            });


            // Advection far from the boundary
            auto tmp1 = union_(union_(union_(leaves_east, leaves_north), leaves_west), leaves_south);
            auto tmp2 = union_(union_(union_(leaves_northeast, leaves_northwest), leaves_southwest), leaves_southeast);
            auto all_leaves_boundary = union_(tmp1, tmp2);
            auto internal_leaves = samurai::difference(mesh[samurai::MeshType::cells][max_level],
                                      all_leaves_boundary).on(max_level); // It is very important to project at this point

            std::cout<<std::endl<<"[=] Far from the boundary"<<std::flush;
            internal_leaves([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                // We enforce a bounce-back
                for (int scheme_n = 0; scheme_n < 3; ++scheme_n)    { // We have 4 schemes
                    advected(0 + 4 * scheme_n, level, k, h) =  f(0 + 4 * scheme_n, level, k - 1, h    );
                    advected(1 + 4 * scheme_n, level, k, h) =  f(1 + 4 * scheme_n, level, k,     h - 1);
                    advected(2 + 4 * scheme_n, level, k, h) =  f(2 + 4 * scheme_n, level, k + 1, h    );
                    advected(3 + 4 * scheme_n, level, k, h) =  f(3 + 4 * scheme_n, level, k,     h + 1);
                }
            });


            // Its time for collision which is local
            auto leaves = intersection(mesh[samurai::MeshType::cells][max_level],
                                       mesh[samurai::MeshType::cells][max_level]);

            std::cout<<std::endl<<"[+] Colliding at finest"<<std::flush;
            leaves([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                auto f0 = xt::eval(advected(0, level, k, h));
                auto f1 = xt::eval(advected(1, level, k, h));
                auto f2 = xt::eval(advected(2, level, k, h));
                auto f3 = xt::eval(advected(3, level, k, h));

                auto f4 = xt::eval(advected(4, level, k, h));
                auto f5 = xt::eval(advected(5, level, k, h));
                auto f6 = xt::eval(advected(6, level, k, h));
                auto f7 = xt::eval(advected(7, level, k, h));

                auto f8  = xt::eval(advected(8,  level, k, h));
                auto f9  = xt::eval(advected(9,  level, k, h));
                auto f10 = xt::eval(advected(10, level, k, h));
                auto f11 = xt::eval(advected(11, level, k, h));

                // We compute the advected momenti
                auto m0_0 = xt::eval(                 f0 + f1 + f2 + f3) ;
                auto m0_1 = xt::eval(lambda        * (f0      - f2      ));
                auto m0_2 = xt::eval(lambda        * (     f1      - f3));
                auto m0_3 = xt::eval(lambda*lambda * (f0 - f1 + f2 - f3));

                auto m1_0 = xt::eval(                 f4 + f5 + f6 + f7) ;
                auto m1_1 = xt::eval(lambda        * (f4      - f6      ));
                auto m1_2 = xt::eval(lambda        * (     f5      - f7));
                auto m1_3 = xt::eval(lambda*lambda * (f4 - f5 + f6 - f7));

                auto m2_0 = xt::eval(                 f8 + f9 + f10 + f11) ;
                auto m2_1 = xt::eval(lambda        * (f8      - f10      ));
                auto m2_2 = xt::eval(lambda        * (     f9       - f11));
                auto m2_3 = xt::eval(lambda*lambda * (f8 - f9 + f10 - f11));


                m0_1 = (1 - s_h_x)  *  m0_1 + s_h_x * (m1_0);
                m0_2 = (1 - s_h_x)  *  m0_2 + s_h_x * (m2_0);
                m0_3 = (1 - s_h_xy) * m0_3;


                m1_1 = (1 - s_q_x)  * m1_1 + s_q_x * (m1_0*m1_0/m0_0 + 0.5*gravity*m0_0*m0_0);
                m1_2 = (1 - s_q_x)  * m1_2 + s_q_x * (m1_0*m2_0/m0_0);
                m1_3 = (1 - s_q_xy) * m1_3;

                m2_1 = (1 - s_q_x)  * m2_1 + s_q_x * (m1_0*m2_0/m0_0);
                m2_2 = (1 - s_q_x)  * m2_2 + s_q_x * (m2_0*m2_0/m0_0 + 0.5*gravity*m0_0*m0_0);
                m2_3 = (1 - s_q_xy) * m2_3;

                new_f(0, level, k, h) =  .25 * m0_0 + .5/lambda * (m0_1)                    + .25/(lambda*lambda) * m0_3;
                new_f(1, level, k, h) =  .25 * m0_0                    + .5/lambda * (m0_2) - .25/(lambda*lambda) * m0_3;
                new_f(2, level, k, h) =  .25 * m0_0 - .5/lambda * (m0_1)                    + .25/(lambda*lambda) * m0_3;
                new_f(3, level, k, h) =  .25 * m0_0                    - .5/lambda * (m0_2) - .25/(lambda*lambda) * m0_3;

                new_f(4, level, k, h) =  .25 * m1_0 + .5/lambda * (m1_1)                    + .25/(lambda*lambda) * m1_3;
                new_f(5, level, k, h) =  .25 * m1_0                    + .5/lambda * (m1_2) - .25/(lambda*lambda) * m1_3;
                new_f(6, level, k, h) =  .25 * m1_0 - .5/lambda * (m1_1)                    + .25/(lambda*lambda) * m1_3;
                new_f(7, level, k, h) =  .25 * m1_0                    - .5/lambda * (m1_2) - .25/(lambda*lambda) * m1_3;

                new_f(8, level, k, h)  =  .25 * m2_0 + .5/lambda * (m2_1)                    + .25/(lambda*lambda) * m2_3;
                new_f(9, level, k, h)  =  .25 * m2_0                    + .5/lambda * (m2_2) - .25/(lambda*lambda) * m2_3;
                new_f(10, level, k, h) =  .25 * m2_0 - .5/lambda * (m2_1)                    + .25/(lambda*lambda) * m2_3;
                new_f(11, level, k, h) =  .25 * m2_0                    - .5/lambda * (m2_2) - .25/(lambda*lambda) * m2_3;


            });

        }

        else
        {

            std::size_t j = max_level - (level + 1);
            double coeff = 1. / (1 << (2*j)); // ATTENTION A LA DIMENSION 2 !!!!

            // This is necessary because the only overleaves we have to advect
            // on are the ones superposed with the leaves to which we come back
            // eventually in the process
            auto overleaves_east = intersection(get_adjacent_boundary_east(mesh, level + 1, samurai::MeshType::overleaves),
                                                mesh[samurai::MeshType::cells][level]);

            auto overleaves_northeast = intersection(get_adjacent_boundary_northeast(mesh, level + 1, samurai::MeshType::overleaves),
                                                mesh[samurai::MeshType::cells][level]);

            auto overleaves_southeast = intersection(get_adjacent_boundary_southeast(mesh, level + 1, samurai::MeshType::overleaves),
                                                mesh[samurai::MeshType::cells][level]);

            auto touching_east = union_(union_(overleaves_east, overleaves_northeast),
                                        overleaves_southeast);

            // General
            touching_east.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                for (int scheme_n = 0; scheme_n < 3; ++scheme_n)    {

                    for(auto &c: pred_coeff[j][0].coeff) // In W
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(0 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(0 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                    for(auto &c: pred_coeff[j][1].coeff) // Out E
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(0 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(0 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                    for(auto &c: pred_coeff[j][3].coeff) // Out N
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(1 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(1 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                    for(auto &c: pred_coeff[j][5].coeff) // Out W
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(2 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(2 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }


                    for(auto &c: pred_coeff[j][7].coeff) // Out S
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(3 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(3 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }
                }
            });
            // Corrections
            std::cout<<std::endl<<"[=] East"<<std::flush;
            overleaves_east.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                for (int scheme_n = 0; scheme_n < 3; ++scheme_n)    {
                    for(auto &c: pred_coeff[j][2].coeff) // In S
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(1 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(1 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                    for(auto &c: pred_coeff[j][6].coeff) // In N
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(3 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(3 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                }

                fluxes(2, level + 1, k, h) += (1<<j) * coeff * f(0, level + 1, k, h);

                for (int scheme_n = 1; scheme_n < 3; ++scheme_n)    {
                    fluxes(2 + 4 * scheme_n, level + 1, k, h) += (1<<j) * coeff * (-1.) * f(0 + 4 * scheme_n, level + 1, k, h);
                }
            });

            std::cout<<std::endl<<"[=] NorthEast"<<std::flush;
            overleaves_northeast.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                for (int scheme_n = 0; scheme_n < 3; ++scheme_n)    {
                    for(auto &c: pred_coeff[j][2].coeff) // In S
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(1 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(1 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }
                }

                fluxes(3, level + 1, k, h) += (1<<j) * coeff * f(1, level + 1, k, h);
                fluxes(2, level + 1, k, h) += (1<<j) * coeff * f(0, level + 1, k, h);
                for (int scheme_n = 1; scheme_n < 3; ++scheme_n)    {
                    fluxes(3 + 4 * scheme_n, level + 1, k, h) += (1<<j) * coeff * (-1) * f(1 + 4 * scheme_n, level + 1, k, h);
                    fluxes(2 + 4 * scheme_n, level + 1, k, h) += (1<<j) * coeff * (-1) * f(0 + 4 * scheme_n, level + 1, k, h);
                }

            });

            std::cout<<std::endl<<"[=] SouthEast"<<std::flush;
            overleaves_southeast.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                for (int scheme_n = 0; scheme_n < 3; ++scheme_n)    {


                    for(auto &c: pred_coeff[j][6].coeff) // In N
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(3 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(3 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }
                }

                fluxes(1, level + 1, k, h) += (1<<j) * coeff * f(3, level + 1, k, h);
                fluxes(2, level + 1, k, h) += (1<<j) * coeff * f(0, level + 1, k, h);

                for (int scheme_n = 1; scheme_n < 3; ++scheme_n)    {
                    fluxes(1 + 4 * scheme_n, level + 1, k, h) += (1<<j) * coeff * (-1.) * f(3 + 4 * scheme_n, level + 1, k, h);
                    fluxes(2 + 4 * scheme_n, level + 1, k, h) += (1<<j) * coeff * (-1.) * f(0 + 4 * scheme_n, level + 1, k, h);
                }

            });


            auto overleaves_west = intersection(get_adjacent_boundary_west(mesh, level + 1, samurai::MeshType::overleaves),
                                                mesh[samurai::MeshType::cells][level]);

            auto overleaves_northwest = intersection(get_adjacent_boundary_northwest(mesh, level + 1, samurai::MeshType::overleaves),
                                                mesh[samurai::MeshType::cells][level]);

            auto overleaves_southwest = intersection(get_adjacent_boundary_southwest(mesh, level + 1, samurai::MeshType::overleaves),
                                                mesh[samurai::MeshType::cells][level]);

            auto touching_west = union_(union_(overleaves_west, overleaves_northwest),
                                        overleaves_southwest);

            std::cout<<std::endl<<"[=] West/NorthWest/SouthWest"<<std::flush;
            touching_west.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                for (int scheme_n = 0; scheme_n < 3; ++scheme_n)    {

                    for(auto &c: pred_coeff[j][1].coeff) // Out E
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(0 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(0 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                    for(auto &c: pred_coeff[j][3].coeff) // Out N
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(1 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(1 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }


                    for(auto &c: pred_coeff[j][4].coeff) // In E
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(2 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(2 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                    for(auto &c: pred_coeff[j][5].coeff) // Out W
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(2 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(2 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                    for(auto &c: pred_coeff[j][7].coeff) // Out S
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(3 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(3 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }
                }
            });

            std::cout<<std::endl<<"[=] West"<<std::flush;
            overleaves_west.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                for (int scheme_n = 0; scheme_n < 3; ++scheme_n)    {

                    for(auto &c: pred_coeff[j][2].coeff) // In S
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(1 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(1 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                    for(auto &c: pred_coeff[j][6].coeff) // In N
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(3 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(3 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }
                }

                fluxes(0, level + 1, k, h) += (1<<j) * coeff * f(2, level + 1, k, h);

                for (int scheme_n = 1; scheme_n < 3; ++scheme_n)    {
                    fluxes(0 + 4 * scheme_n, level + 1, k, h) += (1<<j) * coeff * (-1.) * f(2 + 4 * scheme_n, level + 1, k, h);
                }

            });

            std::cout<<std::endl<<"[=] NorthWest"<<std::flush;
            overleaves_northwest.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                for (int scheme_n = 0; scheme_n < 3; ++scheme_n)    {

                    for(auto &c: pred_coeff[j][2].coeff) // In S
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(1 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(1 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }
                }
                fluxes(0, level + 1, k, h) += (1<<j) * coeff * f(2, level + 1, k, h);
                fluxes(3, level + 1, k, h) += (1<<j) * coeff * f(1, level + 1, k, h);

                for (int scheme_n = 1; scheme_n < 3; ++scheme_n)    {
                    fluxes(0 + 4 * scheme_n, level + 1, k, h) += (1<<j) * coeff * (-1.) * f(2 + 4 * scheme_n, level + 1, k, h);
                    fluxes(3 + 4 * scheme_n, level + 1, k, h) += (1<<j) * coeff * (-1.) * f(1 + 4 * scheme_n, level + 1, k, h);
                }
            });

            std::cout<<std::endl<<"[=] SouthWest"<<std::flush;
            overleaves_southwest.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                for (int scheme_n = 0; scheme_n < 3; ++scheme_n)    {

                    for(auto &c: pred_coeff[j][6].coeff) // In N
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(3 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(3 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }
                }

                fluxes(0, level + 1, k, h) += (1<<j) * coeff * f(2, level + 1, k, h);
                fluxes(1, level + 1, k, h) += (1<<j) * coeff * f(3, level + 1, k, h);

                for (int scheme_n = 1; scheme_n < 3; ++scheme_n)    {
                    fluxes(0 + 4 * scheme_n, level + 1, k, h) += (1<<j) * coeff * (-1.) * f(2 + 4 * scheme_n, level + 1, k, h);
                    fluxes(1 + 4 * scheme_n, level + 1, k, h) += (1<<j) * coeff * (-1.) * f(3 + 4 * scheme_n, level + 1, k, h);
                }
            });


            auto overleaves_south = intersection(get_adjacent_boundary_south(mesh, level + 1, samurai::MeshType::overleaves),
                                                mesh[samurai::MeshType::cells][level]);

            auto overleaves_north = intersection(get_adjacent_boundary_north(mesh, level + 1, samurai::MeshType::overleaves),
                                                mesh[samurai::MeshType::cells][level]);


            auto north_and_south = union_(overleaves_south, overleaves_north);

            std::cout<<std::endl<<"[=] North/South"<<std::flush;
            north_and_south.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                for (int scheme_n = 0; scheme_n < 3; ++scheme_n)    {

                    for(auto &c: pred_coeff[j][0].coeff) // In W
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(0 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(0 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                    for(auto &c: pred_coeff[j][1].coeff) // Out E
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(0 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(0 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                    for(auto &c: pred_coeff[j][3].coeff) // Out N
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(1 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(1 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }


                    for(auto &c: pred_coeff[j][4].coeff) // In E
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(2 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(2 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                    for(auto &c: pred_coeff[j][5].coeff) // Out W
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(2 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(2 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                    for(auto &c: pred_coeff[j][7].coeff) // Out S
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(3 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(3 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }
                }

            });


            std::cout<<std::endl<<"[=] South"<<std::flush;
            overleaves_south.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                for (int scheme_n = 0; scheme_n < 3; ++scheme_n)    {

                    for(auto &c: pred_coeff[j][6].coeff) // In N
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(3 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(3 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }
                }

                fluxes(1, level + 1, k, h) += (1<<j) * coeff * f(3, level + 1, k, h);

                for (int scheme_n = 1; scheme_n < 3; ++scheme_n)    {
                    fluxes(1 + 4 * scheme_n, level + 1, k, h) += (1<<j) * coeff * (-1.) * f(3 + 4 * scheme_n, level + 1, k, h);
                }


            });

            std::cout<<std::endl<<"[=] North"<<std::flush;
            overleaves_north.on(level + 1)([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                for (int scheme_n = 0; scheme_n < 3; ++scheme_n)    {
                    for(auto &c: pred_coeff[j][2].coeff) // In S
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(1 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(1 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }
                }
                fluxes(3, level + 1, k, h) += (1<<j) * coeff * f(1, level + 1, k, h);

                for (int scheme_n = 1; scheme_n < 3; ++scheme_n)    {
                    fluxes(3 + 4 * scheme_n, level + 1, k, h) += (1<<j) * coeff * (-1) * f(1 + 4 * scheme_n, level + 1, k, h);
                }

            });



            // // To update
            std::cout<<std::endl<<"[=] Far from the boundary"<<std::flush;
            auto overleaves_far_boundary = difference(mesh[samurai::MeshType::cells][level],
                                                      union_(union_(touching_east, touching_west),
                                                             north_and_south)).on(level + 1);  // Again, it is very important to project before using

            overleaves_far_boundary([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                for (int scheme_n = 0; scheme_n < 3; ++scheme_n)    {

                    for(auto &c: pred_coeff[j][0].coeff) // In W
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(0 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(0 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                    for(auto &c: pred_coeff[j][1].coeff) // Out E
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(0 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(0 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                    for(auto &c: pred_coeff[j][2].coeff) // In S
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(1 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(1 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                    for(auto &c: pred_coeff[j][3].coeff) // Out N
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(1 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(1 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }


                    for(auto &c: pred_coeff[j][4].coeff) // In E
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(2 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(2 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                    for(auto &c: pred_coeff[j][5].coeff) // Out W
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(2 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(2 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                    for(auto &c: pred_coeff[j][6].coeff) // In N
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(3 + 4 * scheme_n, level + 1, k, h) += coeff * c.second * f(3 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }

                    for(auto &c: pred_coeff[j][7].coeff) // Out S
                    {
                        coord_index_t stencil_x, stencil_y;
                        std::tie(stencil_x, stencil_y) = c.first;

                        fluxes(3 + 4 * scheme_n, level + 1, k, h) -= coeff * c.second * f(3 + 4 * scheme_n, level + 1, k + stencil_x, h + stencil_y);
                    }
                }

            });

// Now that projection has been done, we have to come back on the leaves below the overleaves
            auto leaves = samurai::intersection(mesh[samurai::MeshType::cells][level],
                                             mesh[samurai::MeshType::cells][level]);

            std::cout<<std::endl<<"[+] Projection of the overleaves on their leaves and collision"<<std::flush;
            leaves([&](auto& index, auto &interval, auto) {
                auto k = interval[0]; // Logical index in x
                auto h = index[0];    // Logical index in y

                auto f0 = xt::eval(f(0, level, k, h)) + 0.25 * (fluxes(0, level + 1, 2*k,     2*h)
                                                              + fluxes(0, level + 1, 2*k + 1, 2*h)
                                                              + fluxes(0, level + 1, 2*k,     2*h + 1)
                                                              + fluxes(0, level + 1, 2*k + 1, 2*h + 1));

                auto f1 = xt::eval(f(1, level, k, h)) + 0.25 * (fluxes(1, level + 1, 2*k,     2*h)
                                                              + fluxes(1, level + 1, 2*k + 1, 2*h)
                                                              + fluxes(1, level + 1, 2*k,     2*h + 1)
                                                              + fluxes(1, level + 1, 2*k + 1, 2*h + 1));

                auto f2 = xt::eval(f(2, level, k, h)) + 0.25 * (fluxes(2, level + 1, 2*k,     2*h)
                                                              + fluxes(2, level + 1, 2*k + 1, 2*h)
                                                              + fluxes(2, level + 1, 2*k,     2*h + 1)
                                                              + fluxes(2, level + 1, 2*k + 1, 2*h + 1));

                auto f3 = xt::eval(f(3, level, k, h)) + 0.25 * (fluxes(3, level + 1, 2*k,     2*h)
                                                              + fluxes(3, level + 1, 2*k + 1, 2*h)
                                                              + fluxes(3, level + 1, 2*k,     2*h + 1)
                                                              + fluxes(3, level + 1, 2*k + 1, 2*h + 1));


                auto f4 = xt::eval(f(4, level, k, h)) + 0.25 * (fluxes(4, level + 1, 2*k,     2*h)
                                                              + fluxes(4, level + 1, 2*k + 1, 2*h)
                                                              + fluxes(4, level + 1, 2*k,     2*h + 1)
                                                              + fluxes(4, level + 1, 2*k + 1, 2*h + 1));

                auto f5 = xt::eval(f(5, level, k, h)) + 0.25 * (fluxes(5, level + 1, 2*k,     2*h)
                                                              + fluxes(5, level + 1, 2*k + 1, 2*h)
                                                              + fluxes(5, level + 1, 2*k,     2*h + 1)
                                                              + fluxes(5, level + 1, 2*k + 1, 2*h + 1));

                auto f6 = xt::eval(f(6, level, k, h)) + 0.25 * (fluxes(6, level + 1, 2*k,     2*h)
                                                              + fluxes(6, level + 1, 2*k + 1, 2*h)
                                                              + fluxes(6, level + 1, 2*k,     2*h + 1)
                                                              + fluxes(6, level + 1, 2*k + 1, 2*h + 1));

                auto f7 = xt::eval(f(7, level, k, h)) + 0.25 * (fluxes(7, level + 1, 2*k,     2*h)
                                                              + fluxes(7, level + 1, 2*k + 1, 2*h)
                                                              + fluxes(7, level + 1, 2*k,     2*h + 1)
                                                              + fluxes(7, level + 1, 2*k + 1, 2*h + 1));


                auto f8 = xt::eval(f(8, level, k, h)) + 0.25 * (fluxes(8, level + 1, 2*k,     2*h)
                                                              + fluxes(8, level + 1, 2*k + 1, 2*h)
                                                              + fluxes(8, level + 1, 2*k,     2*h + 1)
                                                              + fluxes(8, level + 1, 2*k + 1, 2*h + 1));

                auto f9 = xt::eval(f(9, level, k, h)) + 0.25 * (fluxes(9, level + 1, 2*k,     2*h)
                                                              + fluxes(9, level + 1, 2*k + 1, 2*h)
                                                              + fluxes(9, level + 1, 2*k,     2*h + 1)
                                                              + fluxes(9, level + 1, 2*k + 1, 2*h + 1));

                auto f10 = xt::eval(f(10, level, k, h)) + 0.25 * (fluxes(10, level + 1, 2*k,     2*h)
                                                                + fluxes(10, level + 1, 2*k + 1, 2*h)
                                                                + fluxes(10, level + 1, 2*k,     2*h + 1)
                                                                + fluxes(10, level + 1, 2*k + 1, 2*h + 1));

                auto f11 = xt::eval(f(11, level, k, h)) + 0.25 * (fluxes(11, level + 1, 2*k,     2*h)
                                                                + fluxes(11, level + 1, 2*k + 1, 2*h)
                                                                + fluxes(11, level + 1, 2*k,     2*h + 1)
                                                                + fluxes(11, level + 1, 2*k + 1, 2*h + 1));



                // We compute the advected momenti
                auto m0_0 = xt::eval(                 f0 + f1 + f2 + f3) ;
                auto m0_1 = xt::eval(lambda        * (f0      - f2      ));
                auto m0_2 = xt::eval(lambda        * (     f1      - f3));
                auto m0_3 = xt::eval(lambda*lambda * (f0 - f1 + f2 - f3));

                auto m1_0 = xt::eval(                 f4 + f5 + f6 + f7) ;
                auto m1_1 = xt::eval(lambda        * (f4      - f6      ));
                auto m1_2 = xt::eval(lambda        * (     f5      - f7));
                auto m1_3 = xt::eval(lambda*lambda * (f4 - f5 + f6 - f7));

                auto m2_0 = xt::eval(                 f8 + f9 + f10 + f11) ;
                auto m2_1 = xt::eval(lambda        * (f8      - f10      ));
                auto m2_2 = xt::eval(lambda        * (     f9       - f11));
                auto m2_3 = xt::eval(lambda*lambda * (f8 - f9 + f10 - f11));

                m0_1 = (1 - s_h_x)  *  m0_1 + s_h_x * (m1_0);
                m0_2 = (1 - s_h_x)  *  m0_2 + s_h_x * (m2_0);
                m0_3 = (1 - s_h_xy) * m0_3;


                m1_1 = (1 - s_q_x)  * m1_1 + s_q_x * (m1_0*m1_0/m0_0 + 0.5*gravity*m0_0*m0_0);
                m1_2 = (1 - s_q_x)  * m1_2 + s_q_x * (m1_0*m2_0/m0_0);
                m1_3 = (1 - s_q_xy) * m1_3;

                m2_1 = (1 - s_q_x)  * m2_1 + s_q_x * ( m1_0*m2_0/m0_0);
                m2_2 = (1 - s_q_x)  * m2_2 + s_q_x * (m2_0*m2_0/m0_0 + 0.5*gravity*m0_0*m0_0);
                m2_3 = (1 - s_q_xy) * m2_3;

                new_f(0, level, k, h) =  .25 * m0_0 + .5/lambda * (m0_1)                    + .25/(lambda*lambda) * m0_3;
                new_f(1, level, k, h) =  .25 * m0_0                    + .5/lambda * (m0_2) - .25/(lambda*lambda) * m0_3;
                new_f(2, level, k, h) =  .25 * m0_0 - .5/lambda * (m0_1)                    + .25/(lambda*lambda) * m0_3;
                new_f(3, level, k, h) =  .25 * m0_0                    - .5/lambda * (m0_2) - .25/(lambda*lambda) * m0_3;

                new_f(4, level, k, h) =  .25 * m1_0 + .5/lambda * (m1_1)                    + .25/(lambda*lambda) * m1_3;
                new_f(5, level, k, h) =  .25 * m1_0                    + .5/lambda * (m1_2) - .25/(lambda*lambda) * m1_3;
                new_f(6, level, k, h) =  .25 * m1_0 - .5/lambda * (m1_1)                    + .25/(lambda*lambda) * m1_3;
                new_f(7, level, k, h) =  .25 * m1_0                    - .5/lambda * (m1_2) - .25/(lambda*lambda) * m1_3;

                new_f(8, level, k, h)  =  .25 * m2_0 + .5/lambda * (m2_1)                    + .25/(lambda*lambda) * m2_3;
                new_f(9, level, k, h)  =  .25 * m2_0                    + .5/lambda * (m2_2) - .25/(lambda*lambda) * m2_3;
                new_f(10, level, k, h) =  .25 * m2_0 - .5/lambda * (m2_1)                    + .25/(lambda*lambda) * m2_3;
                new_f(11, level, k, h) =  .25 * m2_0                    - .5/lambda * (m2_2) - .25/(lambda*lambda) * m2_3;

            });




        }

    }

    std::swap(f.array(), new_f.array());
}



template<class Field>
void save_solution(Field &f, double eps, std::size_t ite, std::string ext="")
{
    using Config = typename Field::Config;
    auto mesh = f.mesh();
    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    std::stringstream str;
    str << "LBM_D2Q4_3_ShallowWaters_" << ext << "_lmin_" << min_level << "_lmax-" << max_level << "_eps-"
        << eps << "_ite-" << ite;

    auto h5file = samurai::Hdf5(str.str().data());
    h5file.add_mesh(mesh);
    samurai::Field<Config> level_{"level", mesh};
    samurai::Field<Config> h{"h", mesh};
    samurai::Field<Config> qx{"qx", mesh};
    samurai::Field<Config> qy{"qy", mesh};

    mesh.for_each_cell([&](auto &cell) {
        level_[cell] = static_cast<double>(cell.level);
        h[cell] = f[cell][0] + f[cell][1] + f[cell][2] + f[cell][3];
        qx[cell]  = f[cell][4] + f[cell][5] + f[cell][6] + f[cell][7];
        qy[cell]  = f[cell][8] + f[cell][9] + f[cell][10] + f[cell][11];

    });
    h5file.add_field(h);
    h5file.add_field(qx);
    h5file.add_field(qy);

    h5file.add_field(f);
    h5file.add_field(level_);
}

int main(int argc, char *argv[])
{
    cxxopts::Options options("lbm_d2q4_3_Shallow Water");

    options.add_options()
                       ("min_level", "minimum level", cxxopts::value<std::size_t>()->default_value("2"))
                       ("max_level", "maximum level", cxxopts::value<std::size_t>()->default_value("9"))
                       ("epsilon", "maximum level", cxxopts::value<double>()->default_value("0.005"))
                       ("log", "log level", cxxopts::value<std::string>()->default_value("warning"))
                       ("h, help", "Help");

    try
    {
        auto result = options.parse(argc, argv);

        if (result.count("help"))
            std::cout << options.help() << "\n";
        else
        {

            //auto save_string = std::string("bruteforce");
            auto save_string = std::string("overleaves");


            std::map<std::string, spdlog::level::level_enum> log_level{{"debug", spdlog::level::debug},
                                                               {"warning", spdlog::level::warn}};
            constexpr size_t dim = 2;
            using Config = samurai::MRConfig<dim, 2>;

            spdlog::set_level(log_level[result["log"].as<std::string>()]);
            std::size_t min_level = result["min_level"].as<std::size_t>();
            std::size_t max_level = result["max_level"].as<std::size_t>();
            double eps = result["epsilon"].as<double>();

            samurai::Box<double, dim> box({-1, -1}, {1, 1});
            samurai::Mesh<Config> mesh{box, min_level, max_level};

            using coord_index_t = typename Config::coord_index_t;
            auto pred_coeff = compute_prediction<coord_index_t>(min_level, max_level);

            // Initialization
            auto f = init_f(mesh, 0);

            double T = 10.;
            double dx = 1.0 / (1 << max_level);
            double dt = dx / lambda;

            std::size_t N = static_cast<std::size_t>(T / dt);

            std::string prefix ("D2Q4_3_ShallowWaters_");
            std::ofstream s_times;
            std::ofstream s_leaves;
            std::ofstream s_totalcells;
            std::ofstream s_comptime;
            s_times.open(prefix + "times.dat");
            s_leaves.open(prefix + "leaves.dat");
            s_totalcells.open(prefix + "totalcells.dat");
            s_comptime.open(prefix + "comptime.dat");

            for (std::size_t nb_ite = 0; nb_ite < N; ++nb_ite)
            {
                std::cout<<std::endl<<"Iteration number = "<<nb_ite<<std::endl;

                tic();

                std::cout<<std::endl<<"[*] Coarsening"<<std::flush;
                for (std::size_t i=0; i<max_level-min_level; ++i)
                {
                    std::cout<<std::endl<<"Step "<<i<<std::flush;
                    if (coarsening(f, eps, i))
                        break;
                }

                std::cout<<std::endl<<"[*] Refinement"<<std::flush;
                for (std::size_t i=0; i<max_level-min_level; ++i)
                {
                    std::cout<<std::endl<<"Step "<<i<<std::flush;
                    if (refinement(f, eps, 0.0, i))
                        break;
                }


                // std::cout<<std::endl<<"[*] Saving solution"<<std::flush;
                // if (nb_ite % 16 == 0)
                //     save_solution(f, eps, nb_ite/16, save_string+std::string("_before")); // Before applying the scheme


                std::cout<<std::endl<<"[*] Entering time stepping"<<std::flush;
                one_time_step_overleaves_corrected(f, pred_coeff, nb_ite);

                auto mesh_process_plus_iteration = toc();
                auto nb_of_leaves = mesh.nb_cells(samurai::MeshType::cells);
                auto nb_all_cells = mesh.nb_cells();

                std::cout<<std::endl<<"{{{{ ++++++ }}}} Number of cells = "<<nb_of_leaves<<" Total number of cells = "<<nb_all_cells<<" Time elapsed = "<<mesh_process_plus_iteration<<std::flush;
                s_times<<(nb_ite * dt)<<std::endl;
                s_leaves<<nb_of_leaves<<std::endl;
                s_totalcells<<nb_all_cells<<std::endl;
                s_comptime<<mesh_process_plus_iteration<<std::endl;


            }

            s_times.close();
            s_leaves.close();
            s_totalcells.close();
            s_comptime.close();
        }
    }
    catch (const cxxopts::OptionException &e)
    {
        std::cout << options.help() << "\n";
    }
    return 0;
}
