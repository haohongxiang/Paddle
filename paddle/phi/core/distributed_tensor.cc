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

DistTensor::DistTensor(const DenseTensor& other, const DTensorMeta& dist_meta)
    : dist_meta_(dist_meta) {
  value_ = std::make_shared<DenseTensor>(other);
}

DistTensor::DistTensor(const DistTensor& other) {
  value_ = other.value_;
  dist_meta_ = other.dist_meta_;
}

DistTensor& DistTensor::operator=(const DistTensor& other) {
  value_ = other.value_;
  dist_meta_ = other.dist_meta_;
  return *this;
}

DistTensor& DistTensor::operator=(DistTensor&& other) {
  std::swap(value_, other.value_);
  dist_meta_ = std::move(other.dist_meta_);
  return *this;
}

void* DistTensor::AllocateFrom(Allocator* allocator,
                               DataType dtype,
                               size_t requested_size,
                               bool fake_alloc) {
  return value_->AllocateFrom(allocator, dtype, requested_size, fake_alloc);
}

}  // namespace phi
