# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle.distributed.utils import expert_count, assign_pos
from paddle.fluid.framework import in_dygraph_mode
from paddle.fluid.layer_helper import LayerHelper

def _alltoall(in_tensor_list, group=None, use_calc_stream=True):
    if group is not None and not group.is_member():
        return
    ring_id = 0 if group is None else group.id

    if in_dygraph_mode():
        return paddle._C_ops.alltoall(in_tensor_list, 'use_calc_stream',
                                      use_calc_stream, 'ring_id', ring_id)

    op_type = 'alltoall'
    helper = LayerHelper(op_type, **locals())
    out = helper.create_variable_for_type_inference(
        dtype=in_tensor_list.dtype)

    helper.append_op(
        type=op_type,
        inputs={'X': [in_tensor_list]},
        outputs={'Out': [out]},
        attrs={
            'ring_id': ring_id,
            'use_calc_stream': use_calc_stream,
            })
    return out


def count_by_gate(gate, num_expert, world_size, require_pos=True, group=None):
    total_expert_count = num_expert * world_size
    with paddle.no_grad():
        local_expert_count = expert_count(gate, total_expert_count)

        if world_size > 1:
            global_expert_count = _alltoall(local_expert_count, group=group)
        else:
            global_expert_count = local_expert_count
        if not require_pos:
            pos = None
        else:
            lec_cum = paddle.cumsum(local_expert_count, axis=0)
            pos = assign_pos(gate, lec_cum)
    return pos, local_expert_count, global_expert_count


def limit_by_capacity(topk_idx, num_expert, world_size, capacity, group=None):
    with paddle.no_grad():
        capacity = paddle.ones(
            shape=[num_expert], dtype=paddle.int64) * capacity
        pos, lec, gec = count_by_gate(
            topk_idx, num_expert, world_size, require_pos=False, group=group)
        new_gec = paddle.distributed.utils.limit_by_capacity(gec, capacity,
                                                             world_size)
        if world_size > 1:
            assert group.nranks == world_size
            new_lec = _alltoall(new_gec, group=group)
        else:
            new_lec = new_gec

        topk_idx = paddle.distributed.utils.prune_gate_by_capacity(
            topk_idx, new_lec, num_expert, world_size)

    return new_lec, new_gec, topk_idx
