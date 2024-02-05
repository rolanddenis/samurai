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
        s               = speed_factor * std::cos(2 * pi * t);
    }
    else
    {
        // In 2d and more, rotating (divergence-free) velocity field.
        s(0)                      = -x(1);
        s(1)                      = x(0);
        s(xt::range(2, shape[0])) = x(xt::range(2, shape[0]));
    }

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
    const auto shape    = x.shape();
    auto rho            = xt::empty_like(xt::view(x, 0));

    if (shape[0] == 1)
    {
        const double pi     = std::acos(-1.);
        const double center = speed_factor / (2 * pi) * std::sin(2 * pi * t);
        rho                 = xt::abs(x - center) <= radius;
    }
    else
    {
        const double theta            = speed_factor * t;
        xt::xtensor<double, 1> center = xt::zeros<double>({shape[0]});
        center(0)                     = std::cos(theta);
        center(1)                     = std::sin(theta);
        rho                           = xt::norm_sq(x - center, {0}) <= radius * radius;
    }

    return rho;
}

/// Computing the ord-norm error for a given field at time t (and for speed c)
template <class Field>
double rho_error(Field const& rho, double t, double speed_factor, double ord = 2)
{
    double error = 0.;

    samurai::for_each_cell(rho.mesh(),
                           [&](auto& cell)
                           {
                               error += xt::sum(xt::pow(xt::abs(rho[cell] - exact_rho(cell.center(), t, speed_factor)), ord)
                                                * std::pow(cell.length, rho.dim))();
                           });

    return std::pow(error, 1. / ord);
}

/// Initializing a field at initial time t = 0 depending on the given speed factor
template <class Mesh>
auto init_rho(Mesh& mesh, double speed_factor)
{
    auto rho = samurai::make_field<double, 1>("rho", mesh);

    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               rho[cell] = exact_rho(cell.center(), 0., speed_factor)(0);
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

/// Run the simulation for the given dimension
template <std::size_t dim, typename Options>
void run_simulation(Options const& options)
{
    using Config = samurai::MRConfig<dim>;

    // Topology of cells and faces
    constexpr auto kcell               = make_KCellND<dim>();
    constexpr auto kfaces              = kcell.lowerIncident();
    constexpr std::size_t max_topology = kcell.topology();

    // Domain boundaries
    const samurai::Box<double, dim> box({options.left_box * xt::ones<double>({dim}), options.right_box * xt::ones<double>({dim})});

    // Some simulation parameters
    double min_dx        = samurai::cell_length(options.max_level);
    double dt            = options.cfl * min_dx / max_speed(options.left_box, options.right_box, options.u, dim);
    const double dt_save = options.Tf / static_cast<double>(options.nfiles);
    double t             = 0.;

    // Creating one mesh per topology
    std::cout << "Creating meshes... " << std::flush;

    std::array<bool, dim> is_periodic;
    is_periodic.fill(options.is_periodic);

    auto meshes = init_topology_meshes<Config>(std::make_index_sequence<max_topology + 1>{});
    (kcell + kfaces)
        .foreach (
            [&](auto c)
            {
                constexpr std::size_t topology = decltype(c)::topology();
                std::get<topology>(
                    meshes) = std::tuple_element_t<topology, decltype(meshes)>(box, options.min_level, options.max_level, is_periodic);
            });
    std::cout << "Done." << std::endl;

    // Creating one field per topology
    std::cout << "Initializing fields... " << std::flush;
    auto fields                        = init_fields(meshes, std::make_index_sequence<max_topology + 1>{});
    std::get<kcell.topology()>(fields) = init_rho(std::get<kcell.topology()>(meshes), options.u);
    kfaces.foreach (
        [&](auto c)
        {
            constexpr std::size_t topology = decltype(c)::topology();
            std::get<topology>(fields)     = samurai::make_field<double, 1>("flux_" + std::to_string(topology), std::get<topology>(meshes));
        });
    std::cout << "Done." << std::endl;

    // Testing error computation
    std::cout << "error: " << rho_error(std::get<kcell.topology()>(fields), 0., options.u) << std::endl;
}

int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);

    struct
    {
        // Simulation parameters
        std::size_t dim  = 1;
        double left_box  = -2;
        double right_box = 2;
        bool is_periodic = false;
        double u         = 1.;
        double Tf        = 1.;
        double cfl       = 0.95;

        // Multiresolution parameters
        std::size_t min_level = 4;
        std::size_t max_level = 10;
        double mr_epsilon     = 2.e-4; // Threshold used by multiresolution
        double mr_regularity  = 1.;    // Regularity guess for multiresolution

        // Output parameters
        fs::path path        = fs::current_path();
        std::string filename = "FV_advection_1d";
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
