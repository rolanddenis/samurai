#include <math.h>
#include <vector>
#include <utility>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

#include <samurai/samurai.hpp>
#include "coarsening.hpp"
#include "refinement.hpp"
#include "criteria.hpp"
#include "prediction_map_2d.hpp"


template<class Config>
auto init_f(samurai::Mesh<Config> &mesh, double t)
{

    samurai::BC<2> bc{ {{ {samurai::BCType::neumann, 0.0},
                       {samurai::BCType::neumann, 0.0},
                       {samurai::BCType::neumann, 0.0},
                       {samurai::BCType::neumann, 0.0}
                    }} };

    samurai::Field<Config, double, 2> f("f", mesh, bc);
    f.array().fill(0);

    mesh.for_each_cell([&](auto &cell) {
        auto center = cell.center();
        auto x = center[0];
        auto y = center[1];

        // double f_new = (y < 0.5) ? 0.5 : 1.;

        // if (std::sqrt(std::pow(x - .5, 2.) + std::pow(y - .5, 2.)) < 0.15)  {
        //     f_new = 2.;
        // }

        double f_new = std::exp(-500. * (std::pow(x - .5, 2.) + std::pow(y - .5, 2.)));

        f[cell][0] = f_new;
        f[cell][1] = f_new;


    });

    return f;
}


template<class Field>
void save_solution(Field &f, double eps, std::size_t ite, std::string ext="")
{
    using Config = typename Field::Config;
    auto mesh = f.mesh();
    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    std::stringstream str;
    str << "Finest_Reconstruction_2D_" << ext << "_lmin_" << min_level << "_lmax-" << max_level << "_eps-"
        << eps << "_ite-" << ite;

    auto h5file = samurai::Hdf5(str.str().data());
    h5file.add_mesh(mesh);
    samurai::Field<Config> level_{"level", mesh};

    mesh.for_each_cell([&](auto &cell) {
        level_[cell] = static_cast<double>(cell.level);
    });

    h5file.add_field(f);
    h5file.add_field(level_);
}


// Attention : the number 2 as second template parameter does not mean
// that we are dealing with two fields!!!!
template<class Field, class interval_t, class ordinates_t, class ordinates_t_bis>
xt::xtensor<double, 2> prediction_all(const Field & f, std::size_t level_g, std::size_t level,
                                      const interval_t & k, const ordinates_t & h,
                                      std::map<std::tuple<std::size_t, std::size_t, interval_t, ordinates_t_bis>, xt::xtensor<double, 2>> & mem_map)
{

    // That is used to employ _ with xtensor
    using namespace xt::placeholders;

    // mem_map.clear();

    auto it = mem_map.find({level_g, level, k, h});


    if (it != mem_map.end() && k.size() == (std::get<2>(it->first)).size())    {

        // std::cout<<std::endl<<"From map level_g = "<<level_g<<" level = "<<level<<"  k = "<<std::get<2>(it->first)<<"  h = "<<std::get<3>(it->first)<<"  shape = "<<xt::adapt(it->second.shape())<<std::flush;


        // std::cout<<std::endl<<"*"<<std::flush;

        // std::cout<<std::endl<<"Interval = "<<k<<"   Size = "<<k.size()<<"  Size before returning map = "<<xt::adapt(it->second.shape())<<std::endl;

        return it->second;
    }
    else
    {


    auto mesh = f.mesh();

    // We put only the size in x (k.size()) because in y
    // we only have slices of size 1.
    // The second term (1) should be adapted according to the
    // number of fields that we have.
    std::vector<std::size_t> shape_x = {k.size(), 2};
    xt::xtensor<double, 2> out = xt::empty<double>(shape_x);

    auto mask = mesh.exists(samurai::MeshType::cells_and_ghosts, level_g + level, k, h); // Check if we are on a leaf or a ghost (CHECK IF IT IS OK)

    xt::xtensor<double, 2> mask_all = xt::empty<double>(shape_x);

    xt::view(mask_all, xt::all(), 0) = mask; // We have only this because we only have one field
    xt::view(mask_all, xt::all(), 1) = mask; // We have only this because we only have one field

    // std::cout<<std::endl<<"Inside all - level_g = "<<level_g<<"  level = "<<level<<"   k = "<<k<<"   h = "<<h<<"  mask = "<<mask;

    // Recursion finished
    if (xt::all(mask))
    {
        // std::cout<<std::endl<<"Data found - level_g = "<<level_g<<"  level = "<<level<<"   k = "<<k<<"   h = "<<h;//" Value = "<<xt::adapt(xt::eval(f(0, level_g + level, k, h)).shape());

        return xt::eval(f(level_g + level, k, h));
    }

    // If we cannot stop here

    auto kg = k >> 1;
    kg.step = 1;

    xt::xtensor<double, 2> val = xt::empty<double>(shape_x);


    /*
    --------------------
    NW   |   N   |   NE
    --------------------
     W   | EARTH |   E
    --------------------
    SW   |   S   |   SE
    --------------------
    */

    // std::cout<<std::endl<<"REC - level - 1 = "<<(level-1)<<"  kg= "<<kg<<"  hg = "<<(h>>1)<<std::flush;

    // std::cout<<std::endl<<"level = "<<level<<"   "<<"H"<<std::endl;
    auto earth  = xt::eval(prediction_all(f, level_g, level - 1, kg    , (h>>1)    , mem_map));
    // std::cout<<std::endl<<"level = "<<level<<"   "<<"W"<<std::endl;

    auto W      = xt::eval(prediction_all(f, level_g, level - 1, kg - 1, (h>>1)    , mem_map));
    // std::cout<<std::endl<<"level = "<<level<<"   "<<"E"<<std::endl;

    auto E      = xt::eval(prediction_all(f, level_g, level - 1, kg + 1, (h>>1)    , mem_map));
    // std::cout<<std::endl<<"level = "<<level<<"   "<<"S"<<std::endl;

    auto S      = xt::eval(prediction_all(f, level_g, level - 1, kg    , (h>>1) - 1, mem_map));
    // std::cout<<std::endl<<"level = "<<level<<"   "<<"N"<<std::endl;

    auto N      = xt::eval(prediction_all(f, level_g, level - 1, kg    , (h>>1) + 1, mem_map));
    // std::cout<<std::endl<<"level = "<<level<<"   "<<"SW"<<std::endl;

    auto SW     = xt::eval(prediction_all(f, level_g, level - 1, kg - 1, (h>>1) - 1, mem_map));
    // std::cout<<std::endl<<"level = "<<level<<"   "<<"SE"<<std::endl;

    auto SE     = xt::eval(prediction_all(f, level_g, level - 1, kg + 1, (h>>1) - 1, mem_map));
    // std::cout<<std::endl<<"level = "<<level<<"   "<<"NW"<<std::endl;

    auto NW     = xt::eval(prediction_all(f, level_g, level - 1, kg - 1, (h>>1) + 1, mem_map));
    // std::cout<<std::endl<<"level = "<<level<<"   "<<"NE"<<std::endl;

    auto NE     = xt::eval(prediction_all(f, level_g, level - 1, kg + 1, (h>>1) + 1, mem_map));


    // This is to deal with odd/even indices in the x direction
    std::size_t start_even = (k.start & 1) ?     1         :     0        ;
    std::size_t start_odd  = (k.start & 1) ?     0         :     1        ;
    std::size_t end_even   = (k.end & 1)   ? kg.size()     : kg.size() - 1;
    std::size_t end_odd    = (k.end & 1)   ? kg.size() - 1 : kg.size()    ;

    int delta_y = (h & 1) ? 1 : 0;
    int m1_delta_y = (delta_y == 0) ? 1 : -1; // (-1)^(delta_y)

    // We recall the formula before doing everything
    /*
    f[j + 1][2k + dx][2h + dy] = f[j][k][h] + 1/8 * (-1)^dx * (f[j][k - 1][h] - f[j][k + 1][h])
                                            + 1/8 * (-1)^dy * (f[j][k][h - 1] - f[j][k][h + 1])
                                - 1/64 * (-1)^(dx+dy) * (f[j][k + 1][h + 1] - f[j][k - 1][h + 1]
                                                         f[j][k - 1][h - 1] - f[j][k + 1][h - 1])

    dx = 0, 1
    dy = 0, 1
    */

    // std::cout<<std::endl<<"  Dim H = "<<xt::adapt(earth.shape())
    //                     <<"  Dim W = "<<xt::adapt(W.shape())
    //                     <<"  Dim E = "<<xt::adapt(E.shape())
    //                     <<"  Dim S = "<<xt::adapt(S.shape())
    //                     <<"  Dim N = "<<xt::adapt(N.shape())
    //                     <<"  Dim SW = "<<xt::adapt(SW.shape())
    //                     <<"  Dim SE = "<<xt::adapt(SE.shape())
    //                     <<"  Dim NW = "<<xt::adapt(NW.shape())
    //                     <<"  Dim NE = "<<xt::adapt(NE.shape())<<std::flush;


    // xt::view(val, xt::range(start_even, _, 2)) = xt::view(earth + E + W, xt::range(start_even, _));



    // xt::view(val, xt::range(start_odd, _, 2))  = xt::view(earth + E + W, xt::range(_, end_odd));
    // if (icase == 0) // EARTH
    // {
    //     auto data  = xt::eval(prediction_all(icase, f, level_g, level - 1, kg    , (h>>1)    , mem_map));
    //     xt::view(val, xt::range(start_even, _, 2)) = xt::view(data, xt::range(start_even, _));
    //     xt::view(val, xt::range(start_odd, _, 2)) = xt::view(data, xt::range(_, end_odd));
    // }

    // else if (icase == 1) // W
    // {
    //     auto data = xt::eval(prediction_all(icase, f, level_g, level - 1, kg - 1, (h>>1)    , mem_map));
    //     xt::view(val, xt::range(start_even, _, 2)) = xt::view(1./8*data, xt::range(start_even, _));
    //     xt::view(val, xt::range(start_odd, _, 2)) = xt::view(-1./8*data, xt::range(_, end_odd));
    // }
    // else if (icase == 2) // E
    // {
    //     std::cout<<std::endl<<"E - level - 1 = "<<(level-1)<<"  kg + 1 = "<<(kg + 1)<<"  hg = "<<(h>>1)<<std::flush;
    //     auto data = xt::eval(prediction_all(icase, f, level_g, level - 1, kg + 1, (h>>1)    , mem_map));
    //     xt::view(val, xt::range(start_even, _, 2)) = xt::view(-1./8*data, xt::range(start_even, _));
    //     xt::view(val, xt::range(start_odd, _, 2)) = xt::view(1./8*data, xt::range(_, end_odd));
    // }
    // else if (icase == 3) // S
    // {
    //     std::cout<<std::endl<<"S - level - 1 = "<<(level-1)<<"  kg= "<<kg<<"  hg - 1 = "<<(h>>1) - 1<<std::flush;
    //     auto data = xt::eval(prediction_all(icase, f, level_g, level - 1, kg    , (h>>1) - 1, mem_map));

    //     std::cout<<std::endl<<"   lhs1 = " << xt::adapt(xt::view(val, xt::range(start_even, _, 2)).shape())
    //                         <<"   rhs1 = " << xt::adapt(xt::view(1./8*m1_delta_y*data, xt::range(start_even, end_even)).shape())
    //                         <<"   lhs2 = " << xt::adapt(xt::view(val, xt::range(start_odd, _, 2)).shape())
    //                         <<"   rhs2 = " << xt::adapt(xt::view(1./8*m1_delta_y*data, xt::range(start_odd, end_odd)).shape())<<std::flush;
    //     std::cout<<std::endl<<"start even = "<<start_even<<" end even = "<<end_even
    //                         <<"start odd = "<<start_odd<<" end odd = "<<end_odd<<std::flush;


    //     xt::view(val, xt::range(start_even, _, 2)) = xt::view(1./8*m1_delta_y*data, xt::range(start_even, _));
    //     xt::view(val, xt::range(start_odd, _, 2)) = xt::view(1./8*m1_delta_y*data, xt::range(_, end_odd));
    // }
    // else if (icase == 4) //N
    // {
    //     auto data = xt::eval(prediction_all(icase, f, level_g, level - 1, kg    , (h>>1) + 1, mem_map));
    //     xt::view(val, xt::range(start_even, _, 2)) = xt::view(-1./8*m1_delta_y*data, xt::range(start_even, _));
    //     xt::view(val, xt::range(start_odd, _, 2)) = xt::view(-1./8*m1_delta_y*data, xt::range(_, end_odd));
    // }
    // else if (icase == 5) // SW
    // {
    //     auto data = xt::eval(prediction_all(icase, f, level_g, level - 1, kg - 1, (h>>1) - 1, mem_map));
    //     xt::view(val, xt::range(start_even, _, 2)) = xt::view(-1/64. * m1_delta_y *data, xt::range(start_even, _));
    //     xt::view(val, xt::range(start_odd, _, 2)) = xt::view(1/64. * m1_delta_y *data, xt::range(_, end_odd));
    // }
    // else if (icase == 6) // SE
    // {
    //     auto data = xt::eval(prediction_all(icase, f, level_g, level - 1, kg + 1, (h>>1) - 1, mem_map));
    //     xt::view(val, xt::range(start_even, _, 2)) = xt::view(1/64. * m1_delta_y *data, xt::range(start_even, _));
    //     xt::view(val, xt::range(start_odd, _, 2)) = xt::view(-1/64. * m1_delta_y *data, xt::range(_, end_odd));
    // }
    // else if (icase == 7) // NW
    // {
    //     auto data = xt::eval(prediction_all(icase, f, level_g, level - 1, kg - 1, (h>>1) + 1, mem_map));
    //     xt::view(val, xt::range(start_even, _, 2)) = xt::view(-1/64. * m1_delta_y *data, xt::range(start_even, _));
    //     xt::view(val, xt::range(start_odd, _, 2)) = xt::view(1/64. * m1_delta_y *data, xt::range(_, end_odd));
    // }
    // else if (icase == 8) // NE
    // {
    //     auto data = xt::eval(prediction_all(icase, f, level_g, level - 1, kg + 1, (h>>1) + 1, mem_map));
    //     xt::view(val, xt::range(start_even, _, 2)) = xt::view(1/64. * m1_delta_y *data, xt::range(start_even, _));
    //     xt::view(val, xt::range(start_odd, _, 2)) = xt::view(-1/64. * m1_delta_y *data, xt::range(_, end_odd));
    // }
        // auto SW     = xt::eval(prediction_all(f, level_g, level - 1, kg - 1, (h>>1) - 1, mem_map));
    // auto SE     = xt::eval(prediction_all(f, level_g, level - 1, kg + 1, (h>>1) - 1, mem_map));
    // auto NW     = xt::eval(prediction_all(f, level_g, level - 1, kg - 1, (h>>1) + 1, mem_map));
    // auto NE     = xt::eval(prediction_all(f, level_g, level - 1, kg + 1, (h>>1) + 1, mem_map));

    // std::cout<<std::endl<<"  Dim H = "<<xt::adapt(earth.shape())
    //                     <<"  Dim W = "<<xt::adapt(W.shape())
    //                     <<"  Dim E = "<<xt::adapt(E.shape())
    //                     <<"  Dim S = "<<xt::adapt(S.shape())
    //                     <<"  Dim N = "<<xt::adapt(N.shape())
    //                     <<"  Dim SW = "<<xt::adapt(SW.shape())
    //                     <<"  Dim SE = "<<xt::adapt(SE.shape())
    //                     <<"  Dim NW = "<<xt::adapt(NW.shape())
    //                     <<"  Dim NE = "<<xt::adapt(NE.shape())<<std::flush;

    xt::view(val, xt::range(start_even, _, 2)) = xt::view(                        earth
                                                          + 1./8               * (W - E)
                                                          + 1./8  * m1_delta_y * (S - N)
                                                          - 1./64 * m1_delta_y * (NE - NW - SE + SW), xt::range(start_even, _));



    xt::view(val, xt::range(start_odd, _, 2))  = xt::view(                        earth
                                                          - 1./8               * (W - E)
                                                          + 1./8  * m1_delta_y * (S - N)
                                                          + 1./64 * m1_delta_y * (NE - NW - SE + SW), xt::range(_, end_odd));

    xt::masked_view(out, !mask_all) = xt::masked_view(val, !mask_all);

    for(int k_mask = 0, k_int = k.start; k_int < k.end; ++k_mask, ++k_int)
    {
        if (mask[k_mask])
        {
            xt::view(out, k_mask) = xt::view(f(level_g + level, {k_int, k_int + 1}, h), 0);
        }
    }

    // std::cout<<std::endl<<"Interval = "<<k<<"   Size = "<<k.size()<<"  Size before returning = "<<xt::adapt(out.shape())<<std::endl;



    // It is crucial to use insert and not []
    // in order not to update the value in case of duplicated (same key)
    mem_map.insert(std::make_pair(std::tuple<std::size_t, std::size_t, interval_t, ordinates_t_bis>{level_g, level, k, h}
                                  ,out));


    return out;

    }
}

template<class Config, class Field>
void save_reconstructed(Field & f, samurai::Mesh<Config> & init_mesh,
                        double eps, std::size_t ite, std::string ext="")
{


    auto mesh = f.mesh();
    auto min_level = mesh.min_level();
    auto max_level = mesh.max_level();


    samurai::mr_projection(f);
    f.update_bc();
    samurai::mr_prediction(f);



    samurai::BC<2> bc{ {{ {samurai::BCType::neumann, 0.0},
                       {samurai::BCType::neumann, 0.0},
                       {samurai::BCType::neumann, 0.0},
                       {samurai::BCType::neumann, 0.0}
                    }} };


    samurai::Field<Config, double, 2> f_reconstructed("f_reconstructed", init_mesh, bc);
    f_reconstructed.array().fill(0.);


    // For memoization
    using interval_t  = typename Config::interval_t; // Type in X
    using ordinates_t = typename Config::index_t;    // Type in Y
    std::map<std::tuple<std::size_t, std::size_t, interval_t, ordinates_t>, xt::xtensor<double, 2>> memoization_map;

    memoization_map.clear();

    for (std::size_t level = 0; level <= max_level; ++level)
    {
        auto number_leaves = mesh.nb_cells(level, samurai::MeshType::cells);

        std::cout<<std::endl<<"Level = "<<level<<"   Until the end = "<<(max_level - level)
                            <<"  Num cells = "<<number_leaves<<"  At finest = "<<number_leaves * (1 << (max_level - level))<<std::endl;


        auto leaves_on_finest = samurai::intersection(mesh[samurai::MeshType::cells][level],
                                                    mesh[samurai::MeshType::cells][level]);

        leaves_on_finest.on(max_level)([&](auto& index, auto &interval, auto) {
            auto k = interval[0];
            auto h = index[0];

            std::cout<<std::endl<<"[*****] - level = "<<level<<"  k = "<<k<<"   h = "<<h<<std::endl;

            f_reconstructed(max_level, k, h) = prediction_all(f, level, max_level - level, k, h, memoization_map);

        });
    }


    std::cout<<std::endl;

    std::stringstream str;
    str << "Finest_Reconstruction_2D_reconstructed_" << ext << "_lmin_" << min_level << "_lmax-" << max_level << "_eps-"
        << eps << "_ite-" << ite;

    auto h5file = samurai::Hdf5(str.str().data());
    h5file.add_mesh(init_mesh);
    h5file.add_field(f_reconstructed);

}



int main(int argc, char *argv[])
{
    cxxopts::Options options("...",
                             "...");

    options.add_options()
                       ("min_level", "minimum level", cxxopts::value<std::size_t>()->default_value("2"))
                       ("max_level", "maximum level", cxxopts::value<std::size_t>()->default_value("7"))
                       ("epsilon", "maximum level", cxxopts::value<double>()->default_value("0.0001"))
                       ("log", "log level", cxxopts::value<std::string>()->default_value("warning"))
                       ("h, help", "Help");

    try
    {
        auto result = options.parse(argc, argv);

        if (result.count("help"))
            std::cout << options.help() << "\n";
        else
        {




            std::map<std::string, spdlog::level::level_enum> log_level{{"debug", spdlog::level::debug},
                                                               {"warning", spdlog::level::warn}};
            constexpr size_t dim = 2;
            using Config = samurai::MRConfig<dim, 2>;

            spdlog::set_level(log_level[result["log"].as<std::string>()]);
            std::size_t min_level = result["min_level"].as<std::size_t>();
            std::size_t max_level = result["max_level"].as<std::size_t>();
            double eps = result["epsilon"].as<double>();

            samurai::Box<double, dim> box({0, 0}, {1, 1});
            samurai::Mesh<Config> mesh{box, min_level, max_level};

            auto f = init_f(mesh, 0);

            auto mesh_everywhere_refined(mesh);
            auto f_everywhere_refined = init_f(mesh_everywhere_refined, 0);

            for (std::size_t i=0; i<max_level-min_level; ++i)
            {
                if (coarsening(f, eps, i))
                    break;
            }

            for (std::size_t i=0; i<max_level-min_level; ++i)
            {
                if (refinement(f, eps, 0.0, i))
                    break;
            }

            samurai::mr_prediction_overleaves(f);

            save_solution(f, eps, 0);
            save_solution(f_everywhere_refined, eps, 0, std::string("original"));


            save_reconstructed(f, mesh_everywhere_refined, eps, 0);


        }
    }
    catch (const cxxopts::OptionException &e)
    {
        std::cout << options.help() << "\n";
    }
    return 0;
}
