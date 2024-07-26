// DO NOT EDIT! This file was auto-generated by crates/re_types_builder/src/codegen/cpp/mod.rs
// Based on "crates/re_types/definitions/rerun/components/blob.fbs".

#pragma once

#include "../collection.hpp"
#include "../result.hpp"

#include <cstdint>
#include <memory>
#include <utility>

namespace arrow {
    class Array;
    class DataType;
    class ListBuilder;
} // namespace arrow

namespace rerun::components {
    /// **Component**: A binary blob of data.
    struct Blob {
        rerun::Collection<uint8_t> data;

      public:
        Blob() = default;

        Blob(rerun::Collection<uint8_t> data_) : data(std::move(data_)) {}

        Blob& operator=(rerun::Collection<uint8_t> data_) {
            data = std::move(data_);
            return *this;
        }
    };
} // namespace rerun::components

namespace rerun {
    template <typename T>
    struct Loggable;

    /// \private
    template <>
    struct Loggable<components::Blob> {
        static constexpr const char Name[] = "rerun.components.Blob";

        /// Returns the arrow data type this type corresponds to.
        static const std::shared_ptr<arrow::DataType>& arrow_datatype();

        /// Serializes an array of `rerun::components::Blob` into an arrow array.
        static Result<std::shared_ptr<arrow::Array>> to_arrow(
            const components::Blob* instances, size_t num_instances
        );

        /// Fills an arrow array builder with an array of this type.
        static rerun::Error fill_arrow_array_builder(
            arrow::ListBuilder* builder, const components::Blob* elements, size_t num_elements
        );
    };
} // namespace rerun
