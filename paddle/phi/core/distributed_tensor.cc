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

#include "paddle/phi/core/distributed_tensor.h"
#include "paddle/fluid/memory/malloc.h"

namespace phi {
DTensor::DTensor(Allocator* a,
                 const DenseTensorMeta& meta,
                 const DTensorMeta& dist_meta)
    : dist_meta_(dist_meta) {
  meta_ = meta;
  holder_ = a->Allocate(SizeOf(dtype()) * numel());
}

DTensor::DTensor(const std::shared_ptr<phi::Allocation>& holder,
                 const DenseTensorMeta& meta,
                 const DTensorMeta& dist_meta)
    : dist_meta_(dist_meta) {
  meta_ = meta;
  holder_ = holder;
}

DTensor::DTensor(const DTensor& other)
    : DenseTensor(), dist_meta_(other.dist_meta_) {
  meta_ = other.meta_;
  holder_ = other.holder_;
  dist_meta_ = other.dist_meta_;
}

DTensor& DTensor::operator=(const DTensor& other) {
  meta_ = other.meta_;
  holder_ = other.holder_;
  dist_meta_ = other.dist_meta_;
  return *this;
}

DTensor& DTensor::operator=(DTensor&& other) {
  meta_ = std::move(other.meta_);
  std::swap(holder_, other.holder_);
  dist_meta_ = std::move(other.dist_meta_);
  return *this;
}

}  // namespace phi
