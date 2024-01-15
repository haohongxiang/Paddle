/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void BitwiseAndKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      DenseTensor* out);

template <typename T, typename Context>
void BitwiseOrKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     DenseTensor* out);

template <typename T, typename Context>
void BitwiseXorKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      DenseTensor* out);

template <typename T, typename Context>
void BitwiseNotKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      DenseTensor* out);

template <typename T, typename Context>
void BitwiseLeftShiftKernel(const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& y,
                            bool is_arithmetic,
                            DenseTensor* out);

template <typename T, typename Context>
void BitwiseRightShiftKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& y,
                             bool is_arithmetic,
                             DenseTensor* out);

}  // namespace phi
