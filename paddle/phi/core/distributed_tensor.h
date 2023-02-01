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

class DistTensor : public TensorBase,
                   public TypeInfoTraits<TensorBase, DistTensor> {
 public:
  DistTensor() { value_.reset(new DenseTensor()); }

  DistTensor(DistTensor&& other) = default;

  DistTensor(const DenseTensor& other, const DTensorMeta& dist_meta);

  DistTensor(const DistTensor& other);

  virtual ~DistTensor() = default;

  DistTensor& operator=(const DistTensor& other);

  DistTensor& operator=(DistTensor&& other);

  void* AllocateFrom(Allocator* allocator,
                     DataType dtype,
                     size_t requested_size = 0,
                     bool fake_alloc = false);

  static const char* name() { return "DistTensor"; }

  const DenseTensor& value() const { return *value_; }

  DenseTensor* mutable_value() { return value_.get(); }

  int64_t numel() const { return value_->numel(); }

  const DDim& dims() const noexcept { return value_->dims(); }

  DataType dtype() const noexcept { return value_->dtype(); }

  DataLayout layout() const noexcept { return value_->layout(); }

  const Place& place() const { return value_->place(); }

  bool valid() const noexcept { return value_->valid(); }

  bool initialized() const { return value_->initialized(); }

 private:
  friend class DenseTensorUtils;

 protected:
  std::shared_ptr<DenseTensor> value_{nullptr};
  DTensorMeta dist_meta_{DTensorMeta()};
};

}  // namespace phi
