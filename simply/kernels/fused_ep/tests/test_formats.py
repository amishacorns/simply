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

"""The weight-format table, the xla mode's routing message, and the private jax
surfaces the package binds."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp

from simply.kernels.fused_ep import formats, layer
from simply.kernels.fused_ep.formats import (
    FP4,
    WEIGHT_FORMS,
    WEIGHT_FORMAT_NAMES,
    WeightFormat,
    weight_form,
    weight_format_of_dtype,
)
from simply.kernels.fused_ep.rowquant import FP8


class WeightFormatTableTest(parameterized.TestCase):

  def test_the_dtype_to_format_map_is_exact(self):
    # Two forms may share an element type (the per-channel and the
    # block-scaled fp8 forms both store float8_e4m3fn): the map answers
    # with the first form in table order, and the scale layout is what
    # selects the block form. Every form's dtype maps to A form of that
    # dtype, and to itself when it is the first with it.
    first = {}
    for name, form in WEIGHT_FORMS.items():
      first.setdefault(jnp.dtype(form.weight_dtype), name)
    for name, form in WEIGHT_FORMS.items():
      self.assertEqual(
          weight_format_of_dtype(form.weight_dtype),
          first[jnp.dtype(form.weight_dtype)],
      )
    for dtype in (jnp.int8, jnp.float16, jnp.float8_e5m2, jnp.float32):
      self.assertIsNone(weight_format_of_dtype(dtype))

  def test_a_format_outside_the_table_raises(self):
    for name in WEIGHT_FORMAT_NAMES:
      self.assertIs(weight_form(name), WEIGHT_FORMS[name])
    with self.assertRaises(ValueError) as caught:
      weight_form("fp6")
    message = str(caught.exception)
    self.assertIn("fp6", message)
    for name in WEIGHT_FORMAT_NAMES:
      self.assertIn(str(name), message)

  def test_the_scale_layout_follows_the_format(self):
    self.assertFalse(WEIGHT_FORMS[WeightFormat.FP8].block_scaled)
    self.assertTrue(WEIGHT_FORMS[WeightFormat.FP4].block_scaled)
    self.assertEqual(formats.PACKED_ROW_TILE, 64)

  @parameterized.parameters((WeightFormat.FP8, FP4), (WeightFormat.FP4, FP8))
  def test_the_layer_checks_the_weights_against_the_format(
      self, weight_format, wrong_dtype
  ):
    struct = jax.ShapeDtypeStruct
    with self.assertRaisesRegex(ValueError, "expert weights"):
      layer.fused_ep_moe(
          struct((64, 256), jnp.bfloat16),
          struct((4, 256, 512), wrong_dtype),
          struct((4, 256, 512), wrong_dtype),
          struct((4, 512), jnp.float32),
          struct((4, 256), jnp.float32),
          struct((64, 4), jnp.float32),
          top_k=2,
          renormalize=True,
          mesh=None,
          tile_rows=128,
          weight_format=weight_format,
      )


class PrivateJaxSurfacesTest(absltest.TestCase):

  def test_the_ref_level_bitcast_is_still_there(self):
    from jax._src.state import types as state_types

    self.assertTrue(hasattr(state_types.AbstractRef, "bitcast"))

  def test_the_core_map_lowering_is_still_there(self):
    from simply.kernels.fused_ep import row_gather_kernel

    self.assertEqual(len(row_gather_kernel._jax_private()), 5)


if __name__ == "__main__":
  absltest.main()
