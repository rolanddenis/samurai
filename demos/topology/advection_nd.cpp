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

/// Saving field in hdf5 format
template <class Field>
void save(const fs::path& path, const std::string& filename, const Field& rho, const std::string& suffix = "")
{
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

/// Initialize a tuple of CellList for each given mesh
template <typename Meshes, std::size_t... I>
auto init_cell_lists(Meshes const&, std::index_sequence<I...>)
{
    return std::tuple<typename std::tuple_element_t<I, Meshes>::cl_type...>{};
}

/// Version of samurai::for_each_interval that unfold the trailing indexes array
template <typename Mesh, typename Function, std::size_t... I>
void for_each_interval_unfold(Mesh&& mesh, Function&& fn, std::index_sequence<I...>)
{
    samurai::for_each_interval(std::forward<Mesh>(mesh),
                               [&](std::size_t level, const auto& interval, const auto& index)
                               {
                                   fn(level, interval, index(I)...);
                               });
}

/// Update face mesh so that each cell (cells id) of the cell mesh has its faces in the face mesh.
template <typename Meshes, typename Fields>
void update_face_meshes(Meshes& meshes, Fields& fields)
{
    constexpr std::size_t dim  = std::tuple_element_t<0, Meshes>::dim;
    constexpr auto kcell       = make_KCellND<dim>();   // A cell
    constexpr auto kfaces      = kcell.lowerIncident(); // All it's faces
    constexpr auto kfaces_curr = kcell.dimension_concatenate(
        [](auto, auto cell)
        {
            return cell.template incident<-1>();
        });                                                    // Faces with same index only
    constexpr std::size_t topology_cnt = kcell.topology() + 1; // Number of possible (full-dimension cell has maximal topology)

    // A new CellList for each topology
    auto all_cl = init_cell_lists(meshes, std::make_index_sequence<topology_cnt>{});

    // Add appropriate intervals for each faces depending on stored cells
    for_each_interval_unfold( // For each interval of the cell mesh
        std::get<kcell.topology()>(meshes),
        [&](std::size_t level, const auto& interval, auto... indexes)
        {
            // std::cout << level << " " << interval << " "; ((std::cout << indexes), ...); std::cout << std::endl;
            kfaces.foreach ( // For each face
                [&](auto kc)
                {
                    auto& cl = std::get<decltype(kc)::topology()>(all_cl);
                    kc.shift( // Shift interval and indexes to add proper interval
                        [&](auto shifted_level, const auto& shifted_interval, auto... shifted_indexes)
                        {
                            cl[shifted_level][{shifted_indexes...}].add_interval(shifted_interval);
                        },
                        level,
                        interval,
                        indexes...);
                });
        },
        std::make_index_sequence<dim - 1>{});

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
}

/// Compute the flux for each face (cells id) depending on the neighboring cells
template <typename Meshes, typename Fields, typename Options>
void update_fluxes(Meshes const& meshes, Fields& fields, Options const& options)
{
    constexpr std::size_t dim  = std::tuple_element_t<0, Meshes>::dim;
    constexpr auto kcell       = make_KCellND<dim>();   // A cell
    constexpr auto kfaces      = kcell.lowerIncident(); // All it's faces
    constexpr auto kfaces_curr = kcell.dimension_concatenate(
        [](auto, auto cell)
        {
            return cell.template incident<-1>();
        });                                                    // Faces with same index only
    constexpr std::size_t topology_cnt = kcell.topology() + 1; // Number of possible (full-dimension cell has maximal topology)

    auto const& rho = std::get<kcell.topology()>(fields);

    kfaces_curr.foreach ( // For each topology of the faces
        [&](auto kc)
        {
            constexpr std::size_t topology = decltype(kc)::topology();
            auto& flux                     = std::get<topology>(fields);

            for_each_interval_unfold( // For each interval of the corresponding mesh
                std::get<topology>(meshes),
                [&](std::size_t level, const auto& interval, auto... indexes)
                {
                    // std::cout << level << " " << interval << " "; ((std::cout << indexes), ...); std::cout << std::endl;
                    [[maybe_unused]] auto dummy = [&](auto l, auto i, auto... j)
                    {
                        return rho(l, i, j...);
                    };
                    auto&& [rho_left, rho_right] = kc.upperIncident().shift(dummy, level, interval, indexes...);
                    // auto && [rho_left, rho_right] = kc.upperIncident().shift(rho, level, interval, indexes...); // FIXME: segfault ?!!

                    /*
                    std::cout << "IWH" << std::endl;
                    std::cout << kc.shift(flux, level, interval, indexes...) << std::endl;
                    std::cout << rho(level, interval, indexes...) << std::endl;
                    std::cout << rho_left << std::endl;
                    std::cout << rho_right << std::endl;
                    */

                    // TODO: using speed function instead of a constant velocity!
                    kc.shift(flux, level, interval, indexes...) = 0.5 * options.u * (rho_left + rho_right)
                                                                + 0.5 * std::abs(options.u) * (rho_left - rho_right);
                },
                std::make_index_sequence<dim - 1>{});
        });
}

template <typename Meshes, typename Fields>
void update_rho(Meshes const& meshes, Fields& fields, double dt)
{
    constexpr std::size_t dim  = std::tuple_element_t<0, Meshes>::dim;
    constexpr auto kcell       = make_KCellND<dim>();   // A cell
    constexpr auto kfaces      = kcell.lowerIncident(); // All it's faces
    constexpr auto kfaces_curr = kcell.dimension_concatenate(
        [](auto, auto cell)
        {
            return cell.template incident<-1>();
        });                                                    // Faces with same index only
    constexpr std::size_t topology_cnt = kcell.topology() + 1; // Number of possible (full-dimension cell has maximal topology)

    auto& rho = std::get<kcell.topology()>(fields);

    for_each_interval_unfold(
        rho.mesh(),
        [&](std::size_t level, const auto& interval, auto... indexes)
        {
            const double factor = dt / samurai::cell_length(level);
            kfaces.enumerate(
                [&](auto idx, auto kc)
                {
                    constexpr std::size_t topology = decltype(kc)::topology();
                    auto& flux                     = std::get<topology>(fields);
                    // FIXME: currently based on lowerIncident order.
                    rho(level, interval, indexes...) -= ((idx % 2 == 0) ? -1 : 1) * factor * kc.shift(flux, level, interval, indexes...);
                });
        },
        std::make_index_sequence<dim - 1>{});
}

/// Run the simulation for the given dimension
template <std::size_t dim, typename Options>
void run_simulation(Options const& options)
{
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
    auto fields = init_fields(meshes, std::make_index_sequence<topology_cnt>{});
    auto& rho   = std::get<kcell.topology()>(fields);
    rho         = init_rho(std::get<kcell.topology()>(meshes), options.u);
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
    auto MRadaptation = samurai::make_MRAdapt(rho);
    MRadaptation(options.mr_epsilon, options.mr_regularity);
    save(options.path, options.filename, rho, "_init");

    ///////////////////////////////////////////////////////////////////////////
    // Time loop
    std::size_t nsave = 1;
    std::size_t nt    = 0;

    while (t < options.Tf)
    {
        dt = std::min(dt, options.Tf - t);
        t += dt;
        std::cout << fmt::format("it {:5d}: t = {:.4e}, dt = {:.4e}, steps:", nt++, t, dt) << std::flush;

        std::cout << " MR" << std::flush;
        MRadaptation(options.mr_epsilon, options.mr_regularity);

        std::cout << " ghost" << std::flush;
        samurai::update_ghost_mr(rho);

        std::cout << " face" << std::flush;
        update_face_meshes(meshes, fields);

        std::cout << " flux" << std::flush;
        update_fluxes(meshes, fields, options);

        std::cout << " rho" << std::flush;
        update_rho(meshes, fields, dt);

        std::cout << ", error = " << std::flush;
        double error = rho_error(rho, t, options.u);

        std::cout << fmt::format("{:.4e}, cells =", error);
        for (std::size_t l = mesh_cell.get_union().min_level(); l <= mesh_cell.get_union().max_level(); ++l)
        {
            std::cout << " " << mesh_cell.get_union().nb_cells(l);
        }
        std::cout << std::endl;
    }

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
