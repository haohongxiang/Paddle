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

#include <memory>

#include "paddle/phi/api/include/context_pool.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/device_context.h"

namespace phi {

class PADDLE_API DISTContext
    : public DeviceContext,
      public TypeInfoTraits<DeviceContext, DISTContext> {
 public:
  DISTContext();

  DISTContext(DISTContext&&);
  DISTContext& operator=(DISTContext&&);

  explicit DISTContext(const Place&);
  virtual ~DISTContext();

  const Place& GetPlace() const override;

  static const char* name() { return "DISTContext"; }

  // Special Methods
  const phi::DeviceContext* Get(const Place& place);

  template <AllocationType T>
  const typename DefaultDeviceContextType<T>::TYPE* GetExecuteDevCtx(
      const Place& place) {
    return reinterpret_cast<const typename DefaultDeviceContextType<T>::TYPE*>(
        Get(place));
  }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace phi
