// DO NOT EDIT! This file was auto-generated by crates/re_types_builder/src/codegen/cpp/mod.rs
// Based on "crates/re_types/definitions/rerun/components/draw_order.fbs".

#pragma once

#include "../result.hpp"

#include <cstdint>
#include <memory>

namespace arrow {
    /// \private
    template <typename T>
    class NumericBuilder;

    class Array;
    class DataType;
    class FloatType;
    using FloatBuilder = NumericBuilder<FloatType>;
} // namespace arrow

namespace rerun::components {
    /// **Component**: Draw order of 2D elements. Higher values are drawn on top of lower values.
    ///
    /// An entity can have only a single draw order component.
    /// Within an entity draw order is governed by the order of the components.
    ///
    /// Draw order for entities with the same draw order is generally undefined.
    struct DrawOrder {
        float value;

      public:
        DrawOrder() = default;

        DrawOrder(float value_) : value(value_) {}

        DrawOrder& operator=(float value_) {
            value = value_;
            return *this;
        }
    };
} // namespace rerun::components

namespace rerun {
    template <typename T>
    struct Loggable;

    /// \private
    template <>
    struct Loggable<components::DrawOrder> {
        static constexpr const char Name[] = "rerun.components.DrawOrder";

        /// Returns the arrow data type this type corresponds to.
        static const std::shared_ptr<arrow::DataType>& arrow_datatype();

        /// Serializes an array of `rerun::components::DrawOrder` into an arrow array.
        static Result<std::shared_ptr<arrow::Array>> to_arrow(
            const components::DrawOrder* instances, size_t num_instances
        );

        /// Fills an arrow array builder with an array of this type.
        static rerun::Error fill_arrow_array_builder(
            arrow::FloatBuilder* builder, const components::DrawOrder* elements, size_t num_elements
        );
    };
} // namespace rerun
