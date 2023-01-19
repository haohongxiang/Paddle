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

#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/dtensor_meta.h"
#include "paddle/phi/core/stream.h"
#include "paddle/phi/core/tensor_base.h"
#include "paddle/phi/core/tensor_meta.h"

namespace phi {

class DTensor : public DenseTensor, public TypeInfoTraits<TensorBase, DTensor> {
 public:
  DTensor();

  DTensor(Allocator* a,
          const DenseTensorMeta& meta,
          const DTensorMeta& dist_meta);

  DTensor(const std::shared_ptr<phi::Allocation>& holder,
          const DenseTensorMeta& meta,
          const DTensorMeta& dist_meta);

  DTensor(DTensor&& other) = default;

  DTensor(const DTensor& other);

  virtual ~DTensor() = default;

  DTensor& operator=(const DTensor& other);

  DTensor& operator=(DTensor&& other);

  static const char* name() { return "DTensor"; }

 private:
  DTensorMeta dist_meta_;
};

}  // namespace phi
