// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include <CLI/CLI.hpp>
#include <cmath>
#include <xtensor/xfixed.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/algorithm/update.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/samurai.hpp>
#include <samurai/stencil_field.hpp>
#include <samurai/uniform_mesh.hpp>

#include <filesystem>
namespace fs = std::filesystem;

template <typename T>
auto exact_rho(T const& x, double t, double u)
{
    const double radius = .2;
    const double center = 0. + t * u;

    using std::abs;
    return 1. * (abs(x - center) <= radius);
}

template <class Mesh>
auto init(Mesh& mesh, double u)
{
    auto rho = samurai::make_field<double, 1>("rho", mesh);
    rho.fill(0.);

    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               rho[cell] = exact_rho(cell.center()[0], 0., u);
                           });

    return rho;
}

template <class Field>
double rho_error(Field const& rho, double u, double t, double ord = 2)
{
    double error = 0.;

    samurai::for_each_cell(rho.mesh(),
                           [&](auto& cell)
                           {
                               error += std::pow(std::abs(rho[cell] - exact_rho(cell.center()[0], t, u)), ord) * cell.length;
                           });

    return std::pow(error, 1. / ord);
}

template <class Field>
void save(const fs::path& path, const std::string& filename, const Field& rho, const std::string& suffix = "")
{
    auto mesh = rho.mesh();

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, rho);
}

int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);

    constexpr std::size_t dim      = 1; // cppcheck-suppress unreadVariable
    using Config                   = samurai::UniformConfig<dim, 2>;
    constexpr std::size_t topology = 0;

    // Simulation parameters
    double left_box   = -2;
    double right_box  = 2;
    double u          = 1.;
    double Tf         = 1.;
    double cfl        = 0.95;
    std::size_t level = 4;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "FD_transport_1d_uniform";
    std::size_t nfiles   = 1;

    CLI::App app{"Finite difference example for the transport equation in 1d on a uniform mesh"};
    app.add_option("--left", left_box, "The left border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--right", right_box, "The right border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--velocity", u, "The velocity of the advection equation")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--level", level, "Level of resolution")->capture_default_str()->group("Simulation parameters");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Ouput");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Ouput");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Ouput");
    CLI11_PARSE(app, argc, argv);

    const samurai::Box<double, dim> box({left_box}, {right_box});
    samurai::UniformMesh<Config, topology> mesh(box, level);

    double dx = samurai::cell_length(level);
    std::cout << dx << std::endl;
    double dt            = cfl * dx / u;
    const double dt_save = Tf / static_cast<double>(nfiles);
    double t             = 0.;

    auto rho = init(mesh, u);
    samurai::make_bc<samurai::Dirichlet>(rho, 0.);

    auto rho_next = samurai::make_field<double, 1>("rho_next", mesh);

    save(path, filename, rho, "_init");

    std::size_t nsave = 1;
    std::size_t nt    = 0;

    while (t != Tf)
    {
        t += dt;
        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }

        rho_next.resize();
        rho_next.fill(0);
        rho_next = rho - dt * samurai::upwind(u, rho);
        std::swap(rho.array(), rho_next.array());

        double error = rho_error(rho, u, t);
        std::cout << fmt::format("iteration {}: t = {}, dt = {}, error = {}", nt++, t, dt, error) << std::endl;

        if (t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
        {
            const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            save(path, filename, rho, suffix);
        }
    }
    samurai::finalize();
    return 0;
}
