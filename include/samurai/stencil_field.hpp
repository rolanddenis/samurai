// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <xtensor/xadapt.hpp>
#include <xtensor/xfunction.hpp>
#include <xtensor/xmasked_view.hpp>

#include "cell.hpp"
#include "field_expression.hpp"
#include "kspace/kcellnd.hpp"
#include "operators_base.hpp"

namespace samurai
{
    template <class D>
    class finite_volume : public field_expression<D>
    {
      public:

        using derived_type = D;

        derived_type& derived_cast() & noexcept;
        const derived_type& derived_cast() const& noexcept;
        derived_type derived_cast() && noexcept;

        // TODO: generic operator() (need base generic flux)
        template <class... CT>
        inline auto operator()(Dim<1>, CT&&... e) const
        {
            return (derived_cast().right_flux(std::forward<CT>(e)...) - derived_cast().left_flux(std::forward<CT>(e)...))
                 / derived_cast().dx();
        }

        template <class... CT>
        inline auto operator()(Dim<2>, CT&&... e) const
        {
            return (-derived_cast().left_flux(std::forward<CT>(e)...) + derived_cast().right_flux(std::forward<CT>(e)...)
                    + -derived_cast().down_flux(std::forward<CT>(e)...) + derived_cast().up_flux(std::forward<CT>(e)...))
                 / derived_cast().dx();
        }

        template <class... CT>
        inline auto operator()(Dim<3>, CT&&... e) const
        {
            return (-derived_cast().left_flux(std::forward<CT>(e)...) + derived_cast().right_flux(std::forward<CT>(e)...)
                    + -derived_cast().down_flux(std::forward<CT>(e)...) + derived_cast().up_flux(std::forward<CT>(e)...)
                    + -derived_cast().front_flux(std::forward<CT>(e)...) + derived_cast().back_flux(std::forward<CT>(e)...))
                 / derived_cast().dx();
        }

        template <class T, std::size_t Dimension>
        inline auto left_flux(std::array<double, Dimension> a, const T& u) const
        {
            return derived_cast().template flux<0, -1>(a, u);
        }

        template <class T, std::size_t Dimension>
        inline auto right_flux(std::array<double, Dimension> a, const T& u) const
        {
            return derived_cast().template flux<0, 1>(a, u);
        }

        template <class T>
        inline auto left_flux(double a, const T& u) const
        {
            return derived_cast().template left_flux(std::array<double, 1>{a}, u);
        }

        template <class T>
        inline auto right_flux(double a, const T& u) const
        {
            return derived_cast().template flux<0, 1>(a, u);
        }

        template <class T, std::size_t Dimension>
        inline auto down_flux(std::array<double, Dimension> a, const T& u) const
        {
            return derived_cast().template flux<1, -1>(a, u);
        }

        template <class T, std::size_t Dimension>
        inline auto up_flux(std::array<double, Dimension> a, const T& u) const
        {
            return derived_cast().template flux<1, 1>(a, u);
        }

        template <class T, std::size_t Dimension>
        inline auto front_flux(std::array<double, Dimension> a, const T& u) const
        {
            return derived_cast().template flux<2, -1>(a, u);
        }

        template <class T, std::size_t Dimension>
        inline auto back_flux(std::array<double, Dimension> a, const T& u) const
        {
            return derived_cast().template flux<2, 1>(a, u);
        }

      protected:

        finite_volume() = default;
    };

    template <class D>
    inline auto finite_volume<D>::derived_cast() & noexcept -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D>
    inline auto finite_volume<D>::derived_cast() const& noexcept -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

    template <class D>
    inline auto finite_volume<D>::derived_cast() && noexcept -> derived_type
    {
        return *static_cast<derived_type*>(this);
    }

    /*******************
     * upwind operator *
     *******************/

    template <class TInterval>
    class upwind_op : public field_operator_base<TInterval>,
                      public finite_volume<upwind_op<TInterval>>
    {
      public:

        INIT_OPERATOR(upwind_op)

        template <class T1, class T2>
        inline auto flux(double a, T1&& ul, T2&& ur) const
        {
            // TODO(loic): remove the xt::eval (bug without, see
            // VF_advection_1d)
            return (.5 * a * (std::forward<T1>(ul) + std::forward<T2>(ur)) + .5 * std::abs(a) * (std::forward<T1>(ul) - std::forward<T2>(ur)));
        }

        /** @brief Generic flux function along given direction and way
         *  @tparam Direction   Direction of the flux
         *  @tparam Step        Forward (+1) or backward (-1) flux
         */
        template <std::size_t Direction, std::ptrdiff_t Step, std::size_t Dimension, class T>
        auto flux(std::array<double, Dimension> a, const T& u) const
        {
            static_assert(Dimension > 0, "Positive dimension only");
            static_assert(Step == 1 || Step == -1, "Step should be equal to ±1");

            constexpr auto cell  = make_KCellND<Dimension>();                 // A cell a full-dimension
            constexpr auto face  = cell.template incident<Direction, Step>(); // Its face along the given direction
            constexpr auto cells = face.upperIncident(); // The neighborhood of this face with greater dimension (so, same as cell)

            // Calling u on shifted indices
            auto shift_helper = [this, &u](auto c, auto... idx)
            {
                return c.shift(u, level, idx...);
            };

            return cells.apply(
                [this, &a, &shift_helper](auto... c)
                {
                    return flux(a[Direction], this->template call_with_indices<Dimension>(shift_helper, c)...);
                });
        }

        template <std::size_t Direction, std::size_t Step, class T>
        auto flux(double a, const T& u) const
        {
            return flux(std::array<double, 1>{a}, u);
        }
    };

    template <class... CT>
    inline auto upwind(CT&&... e)
    {
        return make_field_operator_function<upwind_op>(std::forward<CT>(e)...);
    }

    /*******************
     * upwind operator for the scalar Burgers equation *
     *******************/
    template <class TInterval>
    class upwind_scalar_burgers_op : public field_operator_base<TInterval>,
                                     public finite_volume<upwind_scalar_burgers_op<TInterval>>
    {
      public:

        INIT_OPERATOR(upwind_scalar_burgers_op)

        template <class T1, class T2>
        inline auto flux(double a, const T1& ul, const T2& ur) const
        {
            auto out = xt::xarray<double>::from_shape(ul.shape());
            out.fill(0);

            auto mask1 = (a * ul < a * ur);
            auto mask2 = (ul * ur > 0.0);

            auto min = xt::eval(xt::minimum(xt::abs(ul), xt::abs(ur)));
            auto max = xt::eval(xt::maximum(xt::abs(ul), xt::abs(ur)));

            xt::masked_view(out, mask1 && mask2) = .5 * min * min;
            xt::masked_view(out, !mask1)         = .5 * max * max;

            return out;
        }

        /** @brief Generic flux function along given direction and way
         *  @tparam Direction   Direction of the flux
         *  @tparam Step        Forward (+1) or backward (-1) flux
         */
        template <std::size_t Direction, std::ptrdiff_t Step, std::size_t Dimension, class T>
        auto flux(std::array<double, Dimension> a, const T& u) const
        {
            static_assert(Dimension > 0, "Positive dimension only");
            static_assert(Step == 1 || Step == -1, "Step should be equal to ±1");

            constexpr auto cell  = make_KCellND<Dimension>();                 // A cell a full-dimension
            constexpr auto face  = cell.template incident<Direction, Step>(); // Its face along the given direction
            constexpr auto cells = face.upperIncident(); // The neighborhood of this face with greater dimension (so, same as cell)

            // Calling u on shifted indices
            auto shift_helper = [this, &u](auto c, auto... idx)
            {
                return c.shift(u, level, idx...);
            };

            return cells.apply(
                [this, &a, &shift_helper](auto... c)
                {
                    return flux(a[Direction], this->template call_with_indices<Dimension>(shift_helper, c)...);
                });
        }

        template <std::size_t Direction, std::size_t Step, class T>
        auto flux(double a, const T& u) const
        {
            return flux(std::array<double, 1>{a}, u);
        }
    };

    template <class... CT>
    inline auto upwind_scalar_burgers(CT&&... e)
    {
        return make_field_operator_function<upwind_scalar_burgers_op>(std::forward<CT>(e)...);
    }
} // namespace samurai
