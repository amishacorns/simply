# Copyright 2024 The Simply Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests of the fused expert-parallel bridge and the restore-time weight fusing."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import jax.sharding as js
import numpy as np
from simply.kernels import fused_ep_bridge
from simply.utils import checkpoint_lib
from simply.utils import sharding


class ConfigForTest(absltest.TestCase):

  def test_sets_the_row_numerics_and_the_block(self):
    config = fused_ep_bridge.config_for(
        hidden=256,
        inter=128,
        experts_per_shard=4,
        top_k=2,
        rows='bf16',
        activation_block=512,
    )
    self.assertEqual(
        (
            config.token_rows,
            config.intermediate,
            config.result_rows,
            config.activation_block,
        ),
        ('bf16', 'bf16', 'bf16', 512),
    )

  def test_default_rows_are_fp8(self):
    config = fused_ep_bridge.config_for(
        hidden=256, inter=128, experts_per_shard=4, top_k=2
    )
    self.assertEqual((config.token_rows, config.result_rows), ('fp8', 'fp8'))

  def test_refuses_other_rows(self):
    with self.assertRaisesRegex(ValueError, "'fp8' or 'bf16'"):
      fused_ep_bridge.config_for(
          hidden=256, inter=128, experts_per_shard=4, top_k=2, rows='int8'
      )


class FuseExpertGateUpTest(absltest.TestCase):

  def _replicated(self, shape):
    mesh = sharding.get_default_mesh()
    return jax.ShapeDtypeStruct(
        shape, jnp.float32, sharding=js.NamedSharding(mesh, js.PartitionSpec())
    )

  def test_fuses_gate_first_and_drops_the_halves(self):
    with sharding.mesh_context(mesh_shape=[1, 1, 1], dcn_mesh_shape=[1]):
      gate = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
      up = -gate
      down = np.zeros((2, 4, 3), np.float32)
      # The fusing frees the two halves it consumes, so the state holds
      # device copies and the checks below read the host originals.
      state = {
          'layer': {
              'ffn_0_gate': {'w': jnp.asarray(gate)},
              'ffn_0': {'w': jnp.asarray(up)},
              'ffn_1': {'w': jnp.asarray(down)},
          }
      }
      target = {
          'layer': {
              'ffn_0_fused': {'w': self._replicated((2, 3, 8))},
              'ffn_1': {'w': self._replicated((2, 4, 3))},
          }
      }
      fused = checkpoint_lib.fuse_expert_gate_up(state, target)
    self.assertEqual(set(fused['layer']), {'ffn_0_fused', 'ffn_1'})
    np.testing.assert_array_equal(
        fused['layer']['ffn_0_fused']['w'][..., :4], gate
    )
    np.testing.assert_array_equal(
        fused['layer']['ffn_0_fused']['w'][..., 4:], up
    )
    np.testing.assert_array_equal(fused['layer']['ffn_1']['w'], down)

  def test_passes_a_tree_without_fused_targets_through(self):
    state = {'layer': {'ffn_0': {'w': jnp.ones((2, 3, 4))}}}
    target = {'layer': {'ffn_0': {'w': self._replicated((2, 3, 4))}}}
    with sharding.mesh_context(mesh_shape=[1, 1, 1], dcn_mesh_shape=[1]):
      self.assertIs(checkpoint_lib.fuse_expert_gate_up(state, target), state)


if __name__ == '__main__':
  absltest.main()


class FusedEpLayerErrorsTest(absltest.TestCase):
  """The named errors of the fused path, raised before any device work."""

  def _layer(self, **overrides):
    from simply import config_lib  # pylint: disable=g-import-not-at-top
    from simply import model_lib  # pylint: disable=g-import-not-at-top

    sharding_config = overrides.pop(
        'sharding_config', config_lib.moe_sharding()
    )
    sharding.set_default_mesh_shape(
        mesh_shape=(1, 1, 1, 1), axis_names=sharding_config.mesh_axis_names
    )
    kwargs = dict(
        model_dim=32,
        expand_factor=2,
        sharding_config=sharding_config,
        num_experts=4,
        num_experts_per_token=2,
        ep_method='fused_ep',
        ffn_use_bias=False,
        use_gated_activation_in_ffn=True,
        activation_dtype=jnp.bfloat16,
    )
    kwargs.update(overrides)
    return model_lib.MoEFeedForward(**kwargs)

  def test_weight_quant_is_refused_by_name(self):
    layer = self._layer(weight_quant='int8')
    params = layer.init(jax.random.key(0))
    with self.assertRaisesRegex(NotImplementedError, 'weight_quant'):
      layer.quantize(params)

  def test_an_ungated_ffn_is_refused_by_name(self):
    layer = self._layer(use_gated_activation_in_ffn=False)
    params = layer.init(jax.random.key(0))
    with self.assertRaisesRegex(NotImplementedError, 'gated'):
      layer.quantize(params)

  def test_an_unfused_tree_names_quantize(self):
    layer = self._layer()
    params = layer.init(jax.random.key(0))  # not passed through quantize()
    inputs = jnp.zeros((1, 8, 32), jnp.bfloat16)
    with self.assertRaisesRegex(ValueError, 'quantize'):
      layer.apply(params, inputs)

  def test_experts_over_several_axes_are_refused(self):
    import dataclasses  # pylint: disable=g-import-not-at-top
    from simply import config_lib  # pylint: disable=g-import-not-at-top

    two_axes = dataclasses.replace(
        config_lib.moe_sharding(),
        ffn0_partition=(('seq', 'data'), None, None),
        ffn1_partition=(('seq', 'data'), None, None),
    )
    layer = self._layer(sharding_config=two_axes)
    params = layer.quantize(layer.init(jax.random.key(0)))
    inputs = jnp.zeros((1, 8, 32), jnp.bfloat16)
    with self.assertRaisesRegex(ValueError, 'one mesh axis'):
      layer.apply(params, inputs)
