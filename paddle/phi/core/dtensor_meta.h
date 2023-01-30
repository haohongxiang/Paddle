/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <vector>

#include "paddle/phi/core/ddim.h"
#include "paddle/utils/any.h"

namespace phi {

enum class PlacementType : uint8_t {
  Shard = 0,
  Partial,
  Replicate,
};

class Placement {
 public:
  explicit Placement(const PlacementType& placement) : p_type_(placement) {}
  Placement(const PlacementType& placement, const int64_t dim)
      : p_type_(placement), dim_(dim) {}

  bool is_shard() { return p_type_ == PlacementType::Shard; }
  bool is_partial() { return p_type_ == PlacementType::Partial; }
  bool is_replicate() { return p_type_ == PlacementType::Replicate; }

 private:
  PlacementType p_type_{PlacementType::Replicate};
  int64_t dim_{-1};
};

class DeviceMesh {
 public:
  DeviceMesh(std::string device_type, std::vector<std::vector<int64_t>> mesh)
      : device_type_(device_type), mesh_(mesh) {}

 private:
  std::string device_type_;
  std::vector<std::vector<int64_t>> mesh_;
};

struct DTensorMeta {
  DTensorMeta() = default;
  explicit DTensorMeta(const DeviceMesh& device_mesh)
      : device_mesh_(device_mesh), placement_(PlacementType::Replicate) {}
  DTensorMeta(const DeviceMesh& device_mesh, const Placement& placement)
      : device_mesh_(device_mesh), placement_(placement) {}

  DeviceMesh device_mesh_;
  Placement placement_;
};

}  // namespace phi
