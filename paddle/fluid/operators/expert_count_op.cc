// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/expert_count_op.h"

namespace paddle {
namespace operators {

class ExpertCountOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("gate_idx"), "Input", "gate_idx",
                   "ExpertCount");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "expert_count",
                   "ExpertCount");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    // the dtype of the gate_idx should be same as int64
    auto gate_idx_dtype =
        OperatorWithKernel::IndicateVarDataType(ctx, "gate_idx");
    // auto gate_idx_data_dtype =
    // static_cast<framework::proto::VarType::Type>(gate_idx_dtype);
    PADDLE_ENFORCE_EQ(
        gate_idx_dtype == framework::proto::VarType::INT64, 1,
        platform::errors::InvalidArgument(
            "The dtype of the gate_idx_dtype should be same as int64"));
    return framework::OpKernelType(gate_idx_dtype, ctx.GetPlace());
  }
};

class ExpertCountOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("gate_idx", "(Tensor) The input gate index tensor.");
    AddOutput("Out", "(Tensor) The output expert count tensor.");
    AddAttr<int>("n_expert", "（int), The number of experts.");

    AddComment(R"DOC(expert_count Operator.count gate indices.)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CPU_KERNEL(expert_count, ops::ExpertCountOpCPUKernel<int>,
                       ops::ExpertCountOpCPUKernel<int64_t>);

REGISTER_OP_WITHOUT_GRADIENT(expert_count, ops::ExpertCountOp,
                             ops::ExpertCountOpMaker);
