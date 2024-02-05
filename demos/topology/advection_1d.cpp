/** Advection equation in 1D using finite volume scheme and different cell topology
 *
 * The idea is to precompute and store the flux in a dedicated field whose mesh is associated to faces instead of cells.
 * There is not much point in that case since it doesn't improve the performance or can be done in another way
 * in Samurai but it illustrates (and test) the simultaneous usage of multiple meshes associated to different topology.
 *
 * This 1D case is basic and works without using any topological dedicated features (see the nD versions for a more generic implementation).
 */

#include <CLI/CLI.hpp>

#include <xtensor/xfixed.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/stencil_field.hpp>
#include <samurai/subset/subset_op.hpp>

#include <filesystem>
namespace fs = std::filesystem;

/// Exact solution depending on position x, time t and speed c
template <typename T>
auto exact_u(T const& x, double t, double c)
{
    const double radius = .2;
    const double center = 0. + t * c;

    using std::abs;
    return 1. * (abs(x - center) <= radius);
}

/// Initializing a mesh at initial time t = 0 depending on the speed c
template <class Mesh>
auto init(Mesh& mesh, double c)
{
    auto u = samurai::make_field<double, 1>("u", mesh);
    u.fill(0.);

    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               u[cell] = exact_u(cell.center()[0], 0., c);
                           });

    return u;
}

/// Computing the ord-norm error for a given field at time t (and for speed c)
template <class Field>
double u_error(Field const& u, double c, double t, double ord = 2)
{
    double error = 0.;

    samurai::for_each_cell(u.mesh(),
                           [&](auto& cell)
                           {
                               error += std::pow(std::abs(u[cell] - exact_u(cell.center()[0], t, c)), ord) * cell.length;
                           });

    return std::pow(error, 1. / ord);
}

/// Saving field in hdf5 format
template <class Field>
void save(const fs::path& path, const std::string& filename, const Field& u, const std::string& suffix = "")
{
    auto mesh   = u.mesh();
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

    samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, u, level_);
}

/// Update face mesh so that each cell (cells id) of the cell mesh has its faces in the face mesh.
template <typename MeshCell, typename MeshFace, typename Field>
void update_face_mesh(MeshCell const& mesh_cell, MeshFace& mesh_face, Field& field)
{
    using cl_type = typename MeshFace::cl_type;
    cl_type cl;
    samurai::for_each_interval(mesh_cell,
                               [&](std::size_t level, const auto& interval, const auto& index)
                               {
                                   // Two lines to match the future nD process
                                   cl[level][index].add_interval(interval);     // Left face has same index
                                   cl[level][index].add_interval(interval + 1); // Right face has same index as the next cell
                               });
    MeshFace new_mesh(cl, mesh_face);
    mesh_face.swap(new_mesh);

    // Creates a new field instead of updating old one (will be filled anyway)
    Field new_field("new_f", mesh_face);

    using std::swap;
    swap(new_field.array(), field.array());
}

/// Compute the flux for each face (cells id) depending on the neighboring cells
template <typename MeshFace, typename U, typename Flux>
void update_flux(MeshFace const& mesh_face, U const& u, double a, Flux& flux)
{
    samurai::for_each_interval(mesh_face,
                               [&](std::size_t level, const auto& interval, const auto& index)
                               {
                                   auto ul = u(level, interval - 1, index); // Left cell has same index as previous face
                                   auto ur = u(level, interval, index);     // Right cell has same index as current face
                                   flux(level, interval, index) = .5 * a * (ul + ur) + .5 * std::abs(a) * (ul - ur); // Left or right cell
                                                                                                                     // depending on speed
                                                                                                                     // (a) sign
                               });
}

int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);

    constexpr std::size_t dim = 1; // cppcheck-suppress unreadVariable
    using Config              = samurai::MRConfig<dim>;

    // Simulation parameters
    double left_box  = -2;
    double right_box = 2;
    bool is_periodic = false;
    double a         = 1.;
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

    CLI::App app{"Finite volume example for the advection equation in 1d "
                 "using multiresolution"};
    app.add_option("--left", left_box, "The left border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--right", right_box, "The right border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_flag("--periodic", is_periodic, "Set the domain periodic")->capture_default_str()->group("Simulation parameters");
    app.add_option("--velocity", a, "The velocity of the advection equation")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--min-level", min_level, "Minimum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--max-level", max_level, "Maximum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--mr-eps", mr_epsilon, "The epsilon used by the multiresolution to adapt the mesh")
        ->capture_default_str()
        ->group("Multiresolution");
    app.add_option("--mr-reg",
                   mr_regularity,
                   "The regularity criteria used by the multiresolution to "
                   "adapt the mesh")
        ->capture_default_str()
        ->group("Multiresolution");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Ouput");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Ouput");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Ouput");
    CLI11_PARSE(app, argc, argv);

    const samurai::Box<double, dim> box({left_box}, {right_box});
    samurai::MRMesh<Config, 1> mesh_cell(box, min_level, max_level, {is_periodic});
    samurai::MRMesh<Config, 0> mesh_face(box, min_level, max_level, {is_periodic}); // Topological parameter (the 3rd template) isn't used
                                                                                    // in this code

    double dt            = cfl / (1 << max_level);
    const double dt_save = Tf / static_cast<double>(nfiles);
    double t             = 0.;

    auto u = init(mesh_cell, a);
    if (!is_periodic)
    {
        const xt::xtensor_fixed<int, xt::xshape<1>> left{-1};
        const xt::xtensor_fixed<int, xt::xshape<1>> right{1};
        samurai::make_bc<samurai::Dirichlet>(u, 0.)->on(left, right);
        // same as (just to test OnDirection instead of Everywhere)
        // samurai::make_bc<samurai::Dirichlet>(u, 0.);
    }
    auto flux = samurai::make_field<double, 1>("flux", mesh_face);

    auto MRadaptation = samurai::make_MRAdapt(u);
    MRadaptation(mr_epsilon, mr_regularity);
    save(path, filename, u, "_init");

    std::size_t nsave = 1;
    std::size_t nt    = 0;

    while (t != Tf)
    {
        MRadaptation(mr_epsilon, mr_regularity);

        t += dt;
        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }

        samurai::update_ghost_mr(u);
        update_face_mesh(mesh_cell, mesh_face, flux);
        update_flux(mesh_face, u, a, flux);

        // Apply one time step
        samurai::for_each_interval(mesh_cell,
                                   [&](std::size_t level, const auto& interval, const auto& index)
                                   {
                                       u(level, interval, index) = u(level, interval, index)
                                                                 - dt * (flux(level, interval + 1, index) - flux(level, interval, index))
                                                                       / samurai::cell_length(level);
                                   });

        double error = u_error(u, a, t);
        std::cout << fmt::format("iteration {}: t = {}, dt = {}, error = {}", nt++, t, dt, error);
        std::cout << ", cells =";
        for (std::size_t l = mesh_cell.get_union().min_level(); l <= mesh_cell.get_union().max_level(); ++l)
        {
            std::cout << " " << mesh_cell.get_union().nb_cells(l);
        }
        std::cout << ", faces =";
        for (std::size_t l = mesh_face.get_union().min_level(); l <= mesh_face.get_union().max_level(); ++l)
        {
            std::cout << " " << mesh_face.get_union().nb_cells(l);
        }
        std::cout << std::endl;

        if (t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
        {
            const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            save(path, filename, u, suffix);
        }
    }
    samurai::finalize();
    return 0;
}
