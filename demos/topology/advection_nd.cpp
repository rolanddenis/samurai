#include <CLI/CLI.hpp>

#include <xtensor/xarray.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xtensor.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/kspace/kcellnd.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/stencil_field.hpp>

#include <array>
#include <cmath>
#include <filesystem>
#include <string>
#include <tuple>
#include <utility>
namespace fs = std::filesystem;

/// Velocity field
template <typename Tensor>
auto speed(Tensor const& x, double t, double speed_factor)
{
    const auto shape = x.shape();
    auto s           = xt::empty_like(x);

    if (shape[0] == 1)
    {
        // In 1d, homogeneous in space, but time oscillating velocity field.
        const double pi = std::acos(-1.);
        xt::view(s, 0)  = std::cos(2 * pi * t);
    }
    else
    {
        // In 2d and more, rotating (divergence-free) velocity field.
        xt::view(s, 0)                      = -xt::view(x, 1);
        xt::view(s, 1)                      = xt::view(x, 0);
        xt::view(s, xt::range(2, shape[0])) = xt::view(x, xt::range(2, shape[0]));
    }

    s *= speed_factor;
    return s;
}

/// Returns maximal velocity whatever the time or position
double max_speed(double min_corner, double max_corner, double speed_factor, std::size_t dim)
{
    if (dim == 1)
    {
        return speed_factor;
    }
    else
    {
        double radius = std::sqrt(dim) * std::max(-min_corner, max_corner);
        return speed_factor * radius;
    }
}

/// Exact solution depending on position x, time t and speed c
template <typename Tensor>
auto exact_rho(Tensor const& x, double t, double speed_factor)
{
    const double radius = .2; // Radius of the initial shape
    auto shape          = x.shape();
    auto rho            = xt::empty_like(xt::view(x, 0));

    if (shape[0] == 1)
    {
        const double pi     = std::acos(-1.);
        const double center = speed_factor / (2 * pi) * std::sin(2 * pi * t);
        rho                 = xt::abs(xt::flatten(x) - center) <= radius;
    }
    else
    {
        for (std::size_t i = 1; i < shape.size(); ++i)
        {
            shape[i] = 1;
        }

        const double theta  = speed_factor * t;
        auto center         = xt::eval(xt::zeros<double>(shape));
        xt::view(center, 0) = std::cos(theta);
        xt::view(center, 1) = std::sin(theta);
        rho                 = xt::norm_sq(x - center, {0}) <= radius * radius;
    }

    return rho;
}

/// Computing the ord-norm error for a given field at time t (and for speed c)
template <class Field>
double rho_error(Field const& rho, double t, double speed_factor, double ord = 2)
{
    double error = 0.;

    samurai::for_each_cell_interval(rho.mesh(),
                                    [&](auto const& cells)
                                    {
                                        error += xt::sum(xt::pow(xt::abs(rho[cells] - exact_rho(cells.center(), t, speed_factor)), ord)
                                                         * std::pow(cells.length, rho.dim))();
                                    });

    return std::pow(error, 1. / ord);
}

/// Saving field in hdf5 format
template <class Field>
void save(const fs::path& path, const std::string& filename, const Field& rho, const std::string& suffix = "")
{
    std::cout << "Saving... " << std::flush;
    auto mesh   = rho.mesh();
    auto level_ = samurai::make_field<std::size_t, 1>("level", mesh);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               level_[cell] = cell.level;
                           });

    samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, rho, level_);
    std::cout << "Done." << std::endl;
}

/// Initializing a field at initial time t = 0 depending on the given speed factor
template <class Mesh>
auto init_rho(Mesh& mesh, double speed_factor)
{
    auto rho = samurai::make_field<double, 1>("rho", mesh);

    samurai::for_each_cell_interval(mesh,
                                    [&](auto const& cells)
                                    {
                                        rho[cells] = exact_rho(cells.center(), 0., speed_factor);
                                    });

    return rho;
}

/// Initialize empty mesh for each possible topology
template <typename Config, std::size_t... I>
auto init_topology_meshes(std::index_sequence<I...>)
{
    return std::tuple<samurai::MRMesh<Config, I>...>{};
}

/// Initialize empty field for each possible topology
template <typename Meshes, std::size_t... I>
auto init_fields(Meshes const&, std::index_sequence<I...>)
{
    return std::tuple<samurai::Field<std::tuple_element_t<I, Meshes>>...>{};
}

/// Initialize a tuple of CellList for each given mesh
template <typename Meshes, std::size_t... I>
auto init_cell_lists(Meshes const&, std::index_sequence<I...>)
{
    return std::tuple<typename std::tuple_element_t<I, Meshes>::cl_type...>{};
}

/// Run the simulation for the given dimension
template <std::size_t dim, typename Options>
void run_simulation(Options const& options)
{
    std::cout << "Dimension " << dim << std::endl;

    using Config = samurai::MRConfig<dim>;

    // Topology of cells and faces
    constexpr auto kcell       = make_KCellND<dim>();   // A cell
    constexpr auto kfaces      = kcell.lowerIncident(); // All it's faces
    constexpr auto kfaces_curr = kcell.dimension_concatenate(
        [](auto, auto cell)
        {
            return cell.template incident<-1>();
        });                                                    // Faces with same index only (one kcell per topology)
    constexpr std::size_t topology_cnt = kcell.topology() + 1; // Number of possible (full-dimension cell has maximal topology)

    // Domain boundaries
    const samurai::Box<double, dim> box({options.left_box * xt::ones<double>({dim}), options.right_box * xt::ones<double>({dim})});

    ///////////////////////////////////////////////////////////////////////////
    // Creating one mesh per topology
    std::cout << "Creating meshes... " << std::flush;

    std::array<bool, dim> is_periodic;
    is_periodic.fill(options.is_periodic);

    auto meshes     = init_topology_meshes<Config>(std::make_index_sequence<topology_cnt>{});
    using Meshes    = decltype(meshes);
    auto& mesh_cell = std::get<kcell.topology()>(meshes);
    (kcell + kfaces_curr)
        .foreach (
            [&](auto c)
            {
                constexpr std::size_t topology = decltype(c)::topology();
                std::get<topology>(
                    meshes) = std::tuple_element_t<topology, decltype(meshes)>(box, options.min_level, options.max_level, is_periodic);
            });
    std::cout << "Done." << std::endl;

    ///////////////////////////////////////////////////////////////////////////
    // Creating one field per topology
    std::cout << "Initializing fields... " << std::flush;
    auto fields  = init_fields(meshes, std::make_index_sequence<topology_cnt>{});
    using Fields = decltype(fields);
    auto& rho    = std::get<kcell.topology()>(fields);
    rho          = init_rho(std::get<kcell.topology()>(meshes), options.u);
    kfaces_curr.foreach (
        [&](auto c)
        {
            constexpr std::size_t topology = decltype(c)::topology();
            std::get<topology>(fields)     = samurai::make_field<double, 1>("flux_" + std::to_string(topology), std::get<topology>(meshes));
        });
    std::cout << "Done." << std::endl;

    ///////////////////////////////////////////////////////////////////////////
    // Some simulation parameters
    double min_dx        = samurai::cell_length(options.max_level);
    double dt            = options.cfl * min_dx / max_speed(options.left_box, options.right_box, options.u, dim);
    const double dt_save = options.Tf / static_cast<double>(options.nfiles);
    double t             = 0.;

    ///////////////////////////////////////////////////////////////////////////
    // Boundary conditions
    if (!options.is_periodic)
    {
        samurai::make_bc<samurai::Dirichlet>(rho, 0.);
    }

    ///////////////////////////////////////////////////////////////////////////
    // Multi-resolution adaptation
    std::cout << "Initializing MR... " << std::flush;
    auto MRadaptation = samurai::make_MRAdapt(rho);
    MRadaptation(options.mr_epsilon, options.mr_regularity);
    std::cout << "Done." << std::endl;

    ///////////////////////////////////////////////////////////////////////////
    // Time loop
    save(options.path, options.filename, rho, "_init");
    std::size_t nsave = 1;
    std::size_t nt    = 0;

    while (t < options.Tf)
    {
        dt = std::min(dt, options.Tf - t);
        std::cout << fmt::format("it {:5d}: t = {:.4e}, dt = {:.4e}, steps:", nt++, t + dt, dt) << std::flush;

        ///////////////////////////////////////////////////////////////////////
        // Multi-resolution adaptation
        std::cout << " MR" << std::flush;
        MRadaptation(options.mr_epsilon, options.mr_regularity);

        ///////////////////////////////////////////////////////////////////////
        // Updating ghosts
        std::cout << " ghost" << std::flush;
        samurai::update_ghost_mr(rho);

        ///////////////////////////////////////////////////////////////////////
        // Updating face's meshes
        std::cout << " face" << std::flush;

        // A new CellList for each topology
        auto all_cl = init_cell_lists(meshes, std::make_index_sequence<topology_cnt>{});

        // Add appropriate intervals for each faces depending on stored cells
        for_each_cell_interval( // For each interval of the cell mesh
            mesh_cell,
            [&](auto const& cell_interval)
            {
                kfaces.foreach ( // For each face
                    [&](auto kc)
                    {
                        auto shifted_ci = kc.shift(cell_interval);
                        std::get<decltype(kc)::topology()>(all_cl)[shifted_ci.level][shifted_ci.indices].add_interval(shifted_ci.interval);
                    });
            });

        // Update meshes and associated fields
        kfaces_curr.foreach (
            [&](auto c)
            {
                constexpr std::size_t topology = decltype(c)::topology();

                auto& mesh = std::get<topology>(meshes);
                std::tuple_element_t<topology, Meshes> new_mesh(std::get<topology>(all_cl), mesh);
                mesh.swap(new_mesh);

                std::tuple_element_t<topology, Fields> new_field("new_field", mesh);
                using std::swap;
                swap(new_field.array(), std::get<topology>(fields).array());
            });

        ///////////////////////////////////////////////////////////////////////
        // Updating flux fields
        std::cout << " flux" << std::flush;
        kfaces_curr.foreach ( // For each topology of the faces
            [&](auto kc)
            {
                constexpr std::size_t topology = decltype(kc)::topology();
                auto& flux                     = std::get<topology>(fields);

                for_each_cell_interval(std::get<topology>(meshes),
                                       [&](auto const& cell_interval)
                                       {
                                           auto&& [rho_left, rho_right] = kc.upperIncident().shift(rho, cell_interval);
                                           auto u = xt::view(speed(cell_interval.center(), t, options.u), kc.template ortho_direction<0>());
                                           kc.shift(flux, cell_interval) = 0.5 * u * (rho_left + rho_right)
                                                                         + 0.5 * xt::abs(u) * (rho_left - rho_right);
                                       });
            });

        ///////////////////////////////////////////////////////////////////////
        // Updating rho
        std::cout << " rho" << std::flush;
        for_each_cell_interval(mesh_cell,
                               [&](auto const& cell_interval)
                               {
                                   const double factor = dt / samurai::cell_length(cell_interval.level);
                                   kfaces.enumerate(
                                       [&](auto idx, auto kc)
                                       {
                                           constexpr std::size_t topology = decltype(kc)::topology();
                                           auto& flux                     = std::get<topology>(fields);
                                           // FIXME: currently based on lowerIncident order.
                                           rho[cell_interval] -= ((idx % 2 == 0) ? -1 : 1) * factor * kc.shift(flux, cell_interval);
                                       });
                               });

        ///////////////////////////////////////////////////////////////////////
        // Updating time, error, saving
        t += dt;

        std::cout << ", error = " << std::flush;
        double error = rho_error(rho, t, options.u);

        std::cout << fmt::format("{:.4e}, cells =", error);
        for (std::size_t l = mesh_cell.get_union().min_level(); l <= mesh_cell.get_union().max_level(); ++l)
        {
            std::cout << " " << mesh_cell.get_union().nb_cells(l);
        }
        std::cout << std::endl;

        if (t >= static_cast<double>(nsave + 1) * dt_save || t == options.Tf)
        {
            const std::string suffix = fmt::format("_{}d", dim) + ((options.nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "");
            save(options.path, options.filename, rho, suffix);
        }
    }
}

int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);

    struct
    {
        // Simulation parameters
        std::size_t dim  = 2;
        double left_box  = -2;
        double right_box = 2;
        bool is_periodic = false;
        double u         = 1.;
        double Tf        = 1.;
        double cfl       = 0.95;

        // Multiresolution parameters
        std::size_t min_level = 4;
        std::size_t max_level = 8;
        double mr_epsilon     = 2.e-4; // Threshold used by multiresolution
        double mr_regularity  = 1.;    // Regularity guess for multiresolution

        // Output parameters
        fs::path path        = fs::current_path();
        std::string filename = "FV_advection";
        std::size_t nfiles   = 1;
    } options;

    CLI::App app{"Finite volume example for the advection equation in nD using multiresolution and flux pre-computation in dedicated fields"};
    app.add_option("--dim", options.dim, "Dimension of the space (1, 2 or 3)")->capture_default_str()->group("Simulation parameters");
    app.add_option("--left", options.left_box, "The left border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--right", options.right_box, "The right border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_flag("--periodic", options.is_periodic, "Set the domain periodic")->capture_default_str()->group("Simulation parameters");
    app.add_option("--velocity", options.u, "The velocity of the advection equation")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", options.cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", options.Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--min-level", options.min_level, "Minimum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--max-level", options.max_level, "Maximum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--mr-eps", options.mr_epsilon, "The epsilon used by the multiresolution to adapt the mesh")
        ->capture_default_str()
        ->group("Multiresolution");
    app.add_option("--mr-reg",
                   options.mr_regularity,
                   "The regularity criteria used by the multiresolution to "
                   "adapt the mesh")
        ->capture_default_str()
        ->group("Multiresolution");
    app.add_option("--path", options.path, "Output path")->capture_default_str()->group("Ouput");
    app.add_option("--filename", options.filename, "File name prefix")->capture_default_str()->group("Ouput");
    app.add_option("--nfiles", options.nfiles, "Number of output files")->capture_default_str()->group("Ouput");
    CLI11_PARSE(app, argc, argv);

    switch (options.dim)
    {
        case 1:
            run_simulation<1>(options);
            break;
        case 2:
            run_simulation<2>(options);
            break;
        case 3:
            run_simulation<3>(options);
            break;
        default:
            std::cerr << "Invalid dimension " << options.dim << std::endl;
    }

    return 0;
}
