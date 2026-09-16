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

"""The fused expert-parallel MoE kernel: fused_ep_moe, its Config, the
weight formats, the VMEM accounting, and the layout facts a caller has to
match.

MESH_AXIS is the default name of the expert-parallel mesh axis; `fused_ep_moe`
takes the axis name as `expert_axis`.
"""

from . import device, ffn, vmem
from .config import Config
from .env import config_from_env
from .formats import (
    WEIGHT_BLOCK_DEFAULT,
    WEIGHT_FORMS,
    WEIGHT_FORMAT_NAMES,
    PACKED_ROW_TILE,
    WeightFormat,
    weight_form,
    weight_format_of_dtype,
)
from .layout import LANES, MESH_AXIS, ROW_BLOCK
from .layer import fused_ep_moe
from .routing_tables import ALIGNMENT_SLOT_FIELD, routing_block

ACTIVATIONS = ffn.ACTIVATIONS
SUPPORTED_GENERATIONS = device.SUPPORTED_GENERATIONS
chip_generation = device.chip_generation
supported_on = device.supported_on
vmem_estimate_bytes = vmem.estimate_bytes
vmem_limit = device.vmem_limit

__all__ = [
    "ACTIVATIONS",
    "ALIGNMENT_SLOT_FIELD",
    "Config",
    "LANES",
    "MESH_AXIS",
    "PACKED_ROW_TILE",
    "ROW_BLOCK",
    "SUPPORTED_GENERATIONS",
    "WEIGHT_BLOCK_DEFAULT",
    "WEIGHT_FORMS",
    "WEIGHT_FORMAT_NAMES",
    "WeightFormat",
    "chip_generation",
    "config_from_env",
    "fused_ep_moe",
    "routing_block",
    "supported_on",
    "vmem_estimate_bytes",
    "vmem_limit",
    "weight_form",
    "weight_format_of_dtype",
]
