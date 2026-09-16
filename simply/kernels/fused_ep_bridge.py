# Copyright 2026 The Simply Authors
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

"""The fused expert-parallel MoE kernel's entry points for the MoE layer.

The kernel is the `simply.kernels.fused_ep` package: the routing tables, the
row transport between the expert shards, the streamed expert FFN and the
combine, as Pallas programs, entered through `fused_ep_moe`. This module is
the layer's view of it: the mesh axis the kernel splits the experts over, the
kernel's Config for a layer shape, and the entry call.

The kernel splits the experts over the mesh axis the layer names, the
expert-parallel axis of its sharding config, and holds a replica of them on
every other mesh axis. This path is inference-only: the kernel has no
gradient.
"""

import dataclasses
import types

from simply.kernels import fused_ep

# Rows per FFN tile (a multiple of 8); 128 is the measured choice.
TILE_ROWS = 128


def package() -> types.ModuleType:
  """The `fused_ep` package."""
  return fused_ep


def fused_ep_moe(*args, **kwargs):
  """One MoE layer under expert parallelism, through the kernel."""
  return package().fused_ep_moe(*args, **kwargs)


def weight_format(name: str):
  """The kernel's weight format named `name` ('bf16', 'fp8' or 'fp4')."""
  return package().WeightFormat(name)


def config_for(
    *,
    hidden: int,
    inter: int,
    experts_per_shard: int,
    top_k: int,
    rows: str = 'fp8',
    activation_block: int = 0,
):
  """The kernel's Config for a layer shape, with its row numerics chosen.

  `rows` is 'fp8' (the kernel's default: the token rows, the intermediate
  and the result rows rounded to fp8 with a scale per row) or 'bf16'
  (nothing rounded past bfloat16). `activation_block` > 0 lets an expert
  too large for the whole-expert build stream in column blocks of that
  width (512 fits every shape gated so far; 0 keeps the kernel's
  whole-expert build, which refuses when the local experts' weights do not
  fit VMEM). Everything else is the kernel's own choice for the shape
  (Config.for_shape).
  """
  if rows not in ('fp8', 'bf16'):
    raise ValueError(f"fused_ep_rows is {rows!r}; 'fp8' or 'bf16'")
  config = package().Config.for_shape(
      hidden=hidden,
      inter=inter,
      experts_per_shard=experts_per_shard,
      top_k=top_k,
  )
  return dataclasses.replace(
      config,
      token_rows=rows,
      intermediate=rows,
      result_rows=rows,
      activation_block=int(activation_block),
  )
