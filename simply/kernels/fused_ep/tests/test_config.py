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

"""Config: its defaults, its constraints and the variables that set it.
No device needed."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp

from simply.kernels.fused_ep import Config, config_from_env
from simply.kernels.fused_ep.config import (
    SORTED_ROWS_EXPERTS_PER_SHARD,
    SORTED_ROWS_TOKENS_PER_SHARD,
    sorted_rows_by_shape,
)


class ConfigTest(parameterized.TestCase):

  def test_defaults(self):
    config = Config()
    self.assertEqual(config.activation_block, 0)
    self.assertEqual(config.result_rows, "fp8")
    self.assertEqual(config.tables_blocks_per_step, 0)
    self.assertEqual(config.elementwise_dtype, "bfloat16")
    self.assertEqual(config.combine_dtype, "float32")
    self.assertIsNone(config.token_block)
    self.assertIsNone(config.sorted_rows)
    self.assertEqual(config.token_rows_block, 0)
    self.assertEqual(Config(activation_block=512).token_rows_block, 512)
    self.assertEqual(
        Config(activation_block=512, token_block=0).token_rows_block, 0
    )
    self.assertEqual(config_from_env({}), config)

  def test_one_variable(self):
    config = config_from_env({"FUSED_EP_ACTIVATION_BLOCK": "512"})
    self.assertEqual(config, Config(activation_block=512))

  def test_every_field_is_overridable(self):
    config = config_from_env(
        {
            "FUSED_EP_RESULT_SLOTS": "4",
            "FUSED_EP_ROW_AHEAD": "3",
            "FUSED_EP_WEIGHT_SLOTS": "2",
            "FUSED_EP_WEIGHT_AHEAD": "1",
            "FUSED_EP_SCALE_BF16": "1",
            "FUSED_EP_BOUNDS_CHECKS": "1",
            "FUSED_EP_TABLES_BLOCKS": "4",
            "FUSED_EP_STREAM_BLOCK": "1024",
            "FUSED_EP_STREAM_ROWS": "512",
            "FUSED_EP_VMEM_FRACTION": "0.9",
            "FUSED_EP_ACTIVATION_BLOCK": "512",
            "FUSED_EP_RESULT_ROWS": "bf16",
            "FUSED_EP_ELEMENTWISE": "bfloat16",
            "FUSED_EP_COMBINE": "bfloat16",
            "FUSED_EP_GATHER_BLOCK": "32",
            "FUSED_EP_GATHER_RING": "8",
            "FUSED_EP_SCATTER_CHUNK": "512",
            "FUSED_EP_SCATTER_RING": "2",
            "FUSED_EP_COMBINE_TILE": "256",
            "FUSED_EP_COMBINE_UNROLL": "16",
            "FUSED_EP_ACTIVATION_COLUMNS": "1024",
            "FUSED_EP_TABLES_ZERO_WORDS": "4096",
            "FUSED_EP_WEIGHT_DMA_PRIORITY": "0",
        }
    )
    self.assertEqual(
        config,
        Config(
            result_slots=4,
            row_fetch_ahead=3,
            weight_slots=2,
            weight_prefetch=1,
            row_scale_dtype=jnp.bfloat16,
            bounds_checks=True,
            tables_blocks_per_step=4,
            stream_block=1024,
            stream_rows=512,
            vmem_fraction=0.9,
            activation_block=512,
            result_rows="bf16",
            elementwise_dtype="bfloat16",
            combine_dtype="bfloat16",
            gather_block_rows=32,
            gather_ring=8,
            scatter_chunk_pairs=512,
            scatter_ring=2,
            combine_tokens_per_tile=256,
            combine_unroll=16,
            activation_columns_per_pass=1024,
            tables_zero_words=4096,
            weight_dma_priority=0,
        ),
    )

  def test_every_field_has_a_variable(self):
    # Every Config field is reachable from the environment: setting a
    # value off its default changes the parsed Config's field.
    fields = list(Config.__dataclass_fields__)
    self.assertLen(fields, 28)
    environ = {
        "FUSED_EP_RESULT_SLOTS": "4",
        "FUSED_EP_ROW_AHEAD": "3",
        "FUSED_EP_WEIGHT_SLOTS": "2",
        "FUSED_EP_WEIGHT_AHEAD": "1",
        "FUSED_EP_SCALE_BF16": "1",
        "FUSED_EP_BOUNDS_CHECKS": "1",
        "FUSED_EP_TABLES_BLOCKS": "4",
        "FUSED_EP_STREAM_BLOCK": "1024",
        "FUSED_EP_STREAM_ROWS": "512",
        "FUSED_EP_VMEM_FRACTION": "0.9",
        "FUSED_EP_ACTIVATION_BLOCK": "512",
        "FUSED_EP_TOKEN_BLOCK": "0",
        "FUSED_EP_RESULT_ROWS": "bf16",
        "FUSED_EP_ELEMENTWISE": "float32",
        "FUSED_EP_COMBINE": "bfloat16",
        "FUSED_EP_GATHER_BLOCK": "32",
        "FUSED_EP_GATHER_RING": "8",
        "FUSED_EP_SCATTER_CHUNK": "512",
        "FUSED_EP_SCATTER_RING": "2",
        "FUSED_EP_COMBINE_TILE": "256",
        "FUSED_EP_COMBINE_UNROLL": "16",
        "FUSED_EP_ACTIVATION_COLUMNS": "1024",
        "FUSED_EP_TABLES_ZERO_WORDS": "4096",
        "FUSED_EP_WEIGHT_DMA_PRIORITY": "0",
        "FUSED_EP_REGION_PUSH": "1",
        "FUSED_EP_SORTED_ROWS": "0",
        "FUSED_EP_INTERMEDIATE": "bf16",
        "FUSED_EP_TOKEN_ROWS": "bf16",
    }
    self.assertLen(environ, len(fields))
    changed = config_from_env(environ)
    default = Config()
    for name in fields:
      self.assertNotEqual(getattr(changed, name), getattr(default, name), name)

  @parameterized.parameters(
      dict(result_slots=1),
      dict(result_slots=3, row_fetch_ahead=3),
      dict(row_fetch_ahead=0),
      dict(weight_slots=1),
      dict(weight_slots=3, weight_prefetch=3),
      dict(row_scale_dtype=jnp.float16),
      dict(activation_block=100),
      dict(token_block=100),
      dict(token_block=-128),
      dict(tables_blocks_per_step=-1),
      dict(stream_block=100),
      dict(stream_rows=0),
      dict(stream_rows=12),
      dict(vmem_fraction=0.0),
      dict(vmem_fraction=1.5),
      dict(result_rows="fp16"),
      dict(elementwise_dtype="float16"),
      dict(combine_dtype="int8"),
      dict(gather_block_rows=12),
      dict(gather_ring=0),
      dict(scatter_chunk_pairs=100),
      dict(scatter_ring=0),
      dict(combine_tokens_per_tile=100),
      dict(combine_unroll=48),
      dict(activation_columns_per_pass=100),
      dict(tables_zero_words=100),
      dict(weight_dma_priority=2),
      dict(sorted_rows="yes"),
      dict(sorted_rows=1),
  )
  def test_refused(self, **fields):
    with self.assertRaises(ValueError):
      Config(**fields)

  @parameterized.parameters(
      ({"FUSED_EP_RESULT_SLOTS": "2", "FUSED_EP_ROW_AHEAD": "2"},),
      ({"FUSED_EP_BOUNDS_CHECKS": "yes"},),
      ({"FUSED_EP_RESULT_ROWS": "fp16"},),
      ({"FUSED_EP_TABLES_BLOCKS": "-1"},),
      ({"FUSED_EP_ELEMENTWISE": "float16"},),
      ({"FUSED_EP_SORTED_ROWS": "yes"},),
  )
  def test_refused_from_env(self, environ):
    with self.assertRaises(ValueError):
      config_from_env(environ)

  def test_sorted_rows_variable(self):
    # auto (the default) leaves the choice to the shape; 0 and 1 force it.
    self.assertIsNone(config_from_env({}).sorted_rows)
    self.assertIsNone(
        config_from_env({"FUSED_EP_SORTED_ROWS": "auto"}).sorted_rows
    )
    self.assertIs(
        config_from_env({"FUSED_EP_SORTED_ROWS": "1"}).sorted_rows, True
    )
    self.assertIs(
        config_from_env({"FUSED_EP_SORTED_ROWS": "0"}).sorted_rows, False
    )

  def test_sorted_rows_by_shape(self):
    # Sorted from 32 experts per shard up at any token count, and at 256
    # tokens per shard or fewer for any expert count; per-row otherwise.
    self.assertEqual(
        (SORTED_ROWS_EXPERTS_PER_SHARD, SORTED_ROWS_TOKENS_PER_SHARD), (32, 256)
    )
    for experts, tokens, want in (
        (64, 1024, True),
        (32, 1024, True),
        (48, 512, True),
        (16, 256, True),
        (4, 8, True),
        (16, 512, False),
        (16, 1024, False),
        (8, 512, False),
        (31, 257, False),
    ):
      self.assertEqual(
          sorted_rows_by_shape(experts, tokens), want, (experts, tokens)
      )

  def test_for_shape(self):
    # Without an entry the record is the defaults; with one, the entry's
    # fields sit on top of them and nothing else moves.
    shape = dict(hidden=4096, inter=1024, experts_per_shard=64, top_k=10)
    self.assertEqual(Config.for_shape(**shape), Config())
    from simply.kernels.fused_ep import config as config_module

    saved = dict(config_module.TUNED_BY_SHAPE)
    try:
      config_module.TUNED_BY_SHAPE[(4096, 1024, 64, 10)] = {
          "gather_ring": 4,
          "combine_unroll": 16,
      }
      self.assertEqual(
          Config.for_shape(**shape), Config(gather_ring=4, combine_unroll=16)
      )
      self.assertEqual(Config.for_shape(**dict(shape, top_k=8)), Config())
    finally:
      config_module.TUNED_BY_SHAPE.clear()
      config_module.TUNED_BY_SHAPE.update(saved)

  def test_hashable(self):
    self.assertEqual(hash(Config()), hash(Config()))
    self.assertNotEqual(Config(), Config(result_slots=4))


if __name__ == "__main__":
  absltest.main()
