// DO NOT EDIT! This file was auto-generated by crates/re_types_builder/src/codegen/cpp/mod.rs
// Based on "crates/re_types/definitions/rerun/components/triangle_indices.fbs".

#pragma once

#include "../datatypes/uvec3d.hpp"
#include "../result.hpp"

#include <array>
#include <cstdint>
#include <memory>

namespace rerun::components {
    /// **Component**: The three indices of a triangle in a triangle mesh.
    struct TriangleIndices {
        rerun::datatypes::UVec3D indices;

      public:
        // Extensions to generated type defined in 'triangle_indices_ext.cpp'

        /// Construct TriangleIndices from v0/v1/v2 values.
        TriangleIndices(uint32_t v0, uint32_t v1, uint32_t v2) : indices{v0, v1, v2} {}

        /// Construct UVec3D from v0/v1/v2 uint32_t pointer.
        explicit TriangleIndices(const uint32_t* indices_)
            : indices{indices_[0], indices_[1], indices_[2]} {}

      public:
        TriangleIndices() = default;

        TriangleIndices(rerun::datatypes::UVec3D indices_) : indices(indices_) {}

        TriangleIndices& operator=(rerun::datatypes::UVec3D indices_) {
            indices = indices_;
            return *this;
        }

        TriangleIndices(std::array<uint32_t, 3> xyz_) : indices(xyz_) {}

        TriangleIndices& operator=(std::array<uint32_t, 3> xyz_) {
            indices = xyz_;
            return *this;
        }

        /// Cast to the underlying UVec3D datatype
        operator rerun::datatypes::UVec3D() const {
            return indices;
        }
    };
} // namespace rerun::components

namespace rerun {
    static_assert(sizeof(rerun::datatypes::UVec3D) == sizeof(components::TriangleIndices));

    /// \private
    template <>
    struct Loggable<components::TriangleIndices> {
        static constexpr const char Name[] = "rerun.components.TriangleIndices";

        /// Returns the arrow data type this type corresponds to.
        static const std::shared_ptr<arrow::DataType>& arrow_datatype() {
            return Loggable<rerun::datatypes::UVec3D>::arrow_datatype();
        }

        /// Serializes an array of `rerun::components::TriangleIndices` into an arrow array.
        static Result<std::shared_ptr<arrow::Array>> to_arrow(
            const components::TriangleIndices* instances, size_t num_instances
        ) {
            return Loggable<rerun::datatypes::UVec3D>::to_arrow(&instances->indices, num_instances);
        }
    };
} // namespace rerun
