# fused_ep, the fused expert-parallel MoE kernel

One mixture-of-experts layer under expert parallelism. Every shard of the
mesh owns `experts / num_shards` experts. On each layer call, every token's
top-k routed rows travel to the shards that own their experts, each expert's
FFN runs over the rows routed to it, the results come back to the token's own
shard as fp8 rows with one f32 scale each, and that shard sums them with the
router's weights. The package is one entry point, `fused_ep_moe` (layer.py),
its build-time settings, `Config` (config.py), and the programs behind them.

## The layer in one table

| | |
|---|---|
| mesh | a power-of-two mesh of 1 to 32 shards (8 measured; 16 and 32 span hosts) |
| weights | fp8, fp4, bf16 |
| token exchange | the transport's own start call, hosted across three programs |
| routing tables | the shard-tables kernel, which also emits every shard table the FFN kernel takes |
| row tables | one sparse-core program |
| combine | a sparse-core gather and a TensorCore weighted sum |
| in one line | no XLA collective, scatter or gather left in the layer |

Every program is the package's own. A shape the layer cannot take is
refused by name (the limits below), never served by a slower path instead.

## Design

Every expert-parallel design has two costs: a fixed cost to plan and launch
its transport, and a per-token cost to move each token's results back to
the core that owns it. Designs that pay per token fall behind as batches
grow, because a transfer's cost is mostly fixed. This package plans its
transport once and returns the results in one transfer per contiguous run
of tokens, so its per-token cost is low and it is fastest at large batches;
at small batches a fixed transport plan costs more than per-token
collectives save.

Two choices set the layer apart. The results travel in fp8 with one f32
scale per row, so the return transfer moves a quarter of the bytes of a
bf16 combine, and the output sits within one bf16 rounding of the
unquantized layer rather than bit-identical to it (see Numerics). And the
select, the transport, the routing tables, the row tables, the expert FFN
and the combine are all programs of this package, on the TensorCore and the
sparse cores, so the layer holds no XLA collective, scatter or gather at
all.

### What the layer takes

| | this package |
|---|---|
| expert selection | in the layer, top-k of 1 or more |
| weight formats | fp8, fp4, bf16 |
| activation quantization | fp8 per token |
| per-expert biases, clamped SwiGLU, no-gate form | yes |
| empty experts | skipped |
| `hidden` | a multiple of 128; the VMEM estimate decides how wide |
| expert weight buffers | up to three whole-expert slots; the layer drops to two when three do not fit the VMEM budget (a 4096 x 1536 fp8 expert, Qwen3-235B), and streams the expert in column blocks when two do not (a 4096 x 16384 expert) |
| expert-parallel width | a power of two from 1 to 32 |
| tensor parallelism inside an expert | no |
| tokens per shard | any (rows padded to 8; the message and the combine see the tokens padded to 128) |
| routing | the kernel's softmax top-k over the logits, or the caller's own indices and weights (`routing=`), for sigmoid, biased or grouped routers |

Every shape outside those limits raises a ValueError naming the operand
(the section below lists them), never a wrong answer.

## The layer, program by program

On one shard, in the order the programs run:

1. **Select** (`select_kernel.select_top_k`, TensorCore). Top-k over the
   router logits by repeated max-and-mask, the renormalized weights computed
   in place, the expert indices written straight into the routing message,
   slot-major.
2. **Transport start** (`transport.start_transport`, TensorCore and the
   DMA engine). The shard's rows quantized to fp8 a tile at a time and
   shared with the pair mate as they go, the row scales written into the
   routing message behind the indices, the routing message exchanged with
   every peer, round 1 of the token exchange issued. It returns with the copies still in
   flight: the DMA engine keeps moving bytes across a program boundary and
   the semaphores keep their counts.
3. **Routing tables** (`routing_tables_kernel.shard_tables_kernel`,
   TensorCore). One sealed grid whose steps run in order: the expert
   histogram and within-block rank of every pair block (the histogram
   phase), the
   prefix chain once, the three table lookups per block (the lookup
   phase). Every
   shard table the FFN kernel needs comes out of the prefix step, the two
   row tables the scatter fills are zeroed on the first steps, and the last
   step forwards the transport's middle rounds.
4. **Row tables** (`row_tables_kernel.scatter_row_tables`, sparse cores).
   All 32 vector subcores stream the routed pairs and scatter each routed
   row's token and scale word by index-driven DMA, the scale read straight
   out of the routing message.
5. **Expert FFN** (`ffn_kernel.py`, TensorCore). The transport's final
   waits, then the visit loop: per local expert with rows, the weight
   matrices prefetched `weight_prefetch` experts ahead, tiles of
   `tile_rows` rows fetched by token index `row_fetch_ahead` tiles ahead,
   the two matmuls and the activation, the rows requantized to fp8,
   committed to the outgoing rows, and pushed to their destination shard at
   true length.
6. **Combine** (`combine.combine`, a `row_gather_kernel` program on the
   sparse cores and then a TensorCore sum). The token's top-k arrival rows
   and their scales gathered in pair order, then summed with the router
   weights in the order and precision of the reference layer's combine.

`layer._pallas_step` is the per-shard body that runs these in order.

An expert whose matrices do not fit two whole-expert weight slots runs
streamed (`Config.stream_block`): the weight slots hold column blocks of
one expert, each block's up columns (both halves with a gate) with their
scales and bias and its down rows; a group of `stream_rows` token rows
and their scale windows stay resident, double-buffered across groups,
with the group's float32 running sum in VMEM. Every block adds its
partial down product for every tile of the group; the last block finishes
each tile (down scales, bias, quantization) into the same result staging,
commit and push the whole-expert build uses. The refill sequence is
(expert, group, block) in visit order, prefetched `weight_prefetch` units
ahead, so an expert boundary needs no special case; each group reads the
expert's weights once. The intermediate is rounded per activation block,
one block at a time, so a streamed expert at a stream block equal to the
activation block computes bit for bit what the whole-expert build computes
at that activation block (tests/test_layer.py holds it to that).


## Modules

| file | holds |
|---|---|
| `layer.py` | `fused_ep_moe`: operand validation, the kernel build and cache, the shard_map, the per-shard body |
| `config.py` | `Config`: the pipeline depths, the tables step, the rounding settings, the diagnostic build, the programs' geometry (gather, scatter, combine, activation pass, zeroing, DMA queue) |
| `layout.py` | the layout facts: the lane count, the row block, the 32-bit sublane tile, the default mesh axis name, the collective ids |
| `device.py` | the supported TPU generations, the VMEM budget, the chip reads |
| `formats.py` | the weight formats, fp8 per output channel and fp4 per contraction block |
| `ffn.py` | one tile's expert FFN as plain jnp: the matmuls, the activation, the epilogue |
| `ffn_kernel.py` | the expert FFN program's build: the operand, output, scratch and semaphore records, the checks, the VMEM fit, the program and its `run` |
| `ffn_body.py` | the program body both expert steps share: the build record, the context, row streams, commits, staging, the down product, the push, the drains |
| `ffn_expert_whole.py` | the whole-expert step: an expert's matrices in one weight slot, rows a tile at a time |
| `ffn_expert_streamed.py` | the streamed step: an expert in column blocks, a group of rows resident with a float32 running sum |
| `vmem.py` | the FFN kernel's VMEM residents, and whether a build fits |
| `routing_tables.py` | the routing tables in their reference form (dense jnp), which the tests hold the shard-tables kernel to |
| `routing_tables_kernel.py` | `shard_tables_kernel`: every shard table as one Pallas program |
| `transport.py` | the token exchange, an all-gather over a hypercube of shard pairs |
| `row_tables_kernel.py`, `row_gather_kernel.py` | the sparse-core programs: the row-table scatter and the combine's row gather |
| `combine.py` | the combine: the gather and the weighted sum |
| `select_kernel.py` | `select_top_k` |
| `rowquant.py` | per-row fp8 quantization, the one set of lines every transferred row is quantized by |
| `reference.py` | the layer's arithmetic in plain jnp with the same quantization steps, for the tests |
| `env.py` | `config_from_env`: the one place the environment is read, for a caller at a process boundary |
| `__init__.py` | the package's public names: the entry `fused_ep_moe`, `Config`, `WeightFormat`, the layout constants and the device facts |
| `tests/` | the device tests: the layer against the reference, every configurable value, the refusals |

## Settings

`fused_ep_moe(..., config=Config(...))`. Nothing inside the package reads
the environment. `env.config_from_env()` is the one function that does,
for a caller at a process boundary, such as a serving adapter.
Every field has a default, and every field is overridable by its variable.

| field | default | variable | meaning and constraint |
|---|---|---|---|
| `result_slots` | 3 | `FUSED_EP_RESULT_SLOTS` | buffers a tile's result rows are staged in, at least 2 |
| `row_fetch_ahead` | 2 | `FUSED_EP_ROW_AHEAD` | tiles ahead of compute a tile's rows are fetched, smaller than `result_slots` |
| `weight_slots` | 3 | `FUSED_EP_WEIGHT_SLOTS` | the most buffers an expert's weight matrices stream into, at least 2; the layer takes the largest count that fits the VMEM budget |
| `weight_prefetch` | 2 | `FUSED_EP_WEIGHT_AHEAD` | experts ahead a weight refill is issued, smaller than `weight_slots` |
| `stream_block` | `0` | `FUSED_EP_STREAM_BLOCK` | intermediate columns per streamed weight block. 0 = whole-expert weight slots when they fit the VMEM budget, else the widest block that fits; a count forces the streamed build. A streamed expert rounds its intermediate per activation block, so it needs `activation_block` set, dividing the stream block |
| `stream_rows` | `1024` | `FUSED_EP_STREAM_ROWS` | token rows per group of a streamed expert, whose float32 running sum is resident; the layer halves it down to the tile height until the group fits |
| `vmem_fraction` | `0.98` | `FUSED_EP_VMEM_FRACTION` | the share of the chip's VMEM the kernels plan for: the weight-slot fit, the streaming choice and each program's limit |
| `tables_blocks_per_step` | `0` | `FUSED_EP_TABLES_BLOCKS` | routing blocks of 128 pairs the tables kernels handle per grid step; 0 = per call, the largest count up to 8 that pads the shard no further than the gather's unit |
| `activation_block` | `0` | `FUSED_EP_ACTIVATION_BLOCK` | values per fp8 scale of a row, for the token rows and the intermediate: 0 = one scale per row; 512 fits every shape gated so far; a multiple of 128 dividing hidden and inter (the weight block, with four-bit weights). Block 128 at hidden 4096 (32 scale planes) does not compile in Mosaic; 256 and 512 do at every measured shape, 128 at hidden 2048 |
| `token_block` | unset | `FUSED_EP_TOKEN_BLOCK` | the token rows' own rounding block when it differs from `activation_block`: unset follows it, except that a streamed fp8 or bf16 expert takes 0 on its own (3 to 16 percent less layer time on every streamed shape measured); 0 is one scale per row into the up projection (one plain matmul and a single rescale) while the intermediate keeps its block, which is what a streamed expert needs; a multiple of 128 dividing hidden is one scale per that many values (with four-bit weights, the weight block) |
| `elementwise_dtype` | `"bfloat16"` | `FUSED_EP_ELEMENTWISE` | the element type of the fp8 scaling products (row quantizer, intermediate, result epilogue). bf16 is a packed multiply whose second rounding the compiler places, so two compilations can differ by an fp8 step on a few percent of values; it measured 1.5 percent less layer time than float32 at 8192 tokens. `"float32"` rounds each product once: every compilation, the reference and the kernels agree to the bit |
| `combine_dtype` | `"float32"` | `FUSED_EP_COMBINE` | the element type of the combine's weighted sum; bf16 measured slower |
| `row_scale_dtype` | float32 | `FUSED_EP_SCALE_BF16=1` | rounding of each row's quantization scale. The model eval judges this one |
| `bounds_checks` | off | `FUSED_EP_BOUNDS_CHECKS=1` | a diagnostic build with Mosaic's bounds checks on every dynamic index. Slower, changes no value |
| `gather_block_rows` | 16 | `FUSED_EP_GATHER_BLOCK` | rows per index-driven DMA of the sparse-core gather behind the combine, a whole number of 8-row blocks |
| `gather_ring` | 6 | `FUSED_EP_GATHER_RING` | the gather's DMAs in flight per subcore; halved while the ring's staging (block rows x row bytes x ring) exceeds the default geometry's at 4096 lanes |
| `scatter_chunk_pairs` | 640 | `FUSED_EP_SCATTER_CHUNK` | the most routed pairs per chunk of the sparse-core row-tables scatter, a multiple of 128; the layer takes the largest count up to it that divides a subcore's run |
| `scatter_ring` | 4 | `FUSED_EP_SCATTER_RING` | the scatter's chunks in flight per subcore |
| `combine_tokens_per_tile` | 128 | `FUSED_EP_COMBINE_TILE` | tokens per grid step of the combine's weighted sum, a whole number of 8-row blocks; the routing message pads the tokens to a multiple of it |
| `combine_unroll` | 32 | `FUSED_EP_COMBINE_UNROLL` | the weighted sum's token loop unroll, dividing the tile |
| `activation_columns_per_pass` | 512 | `FUSED_EP_ACTIVATION_COLUMNS` | columns of the intermediate the FFN kernel activates and requantizes per pass, a multiple of 128 |
| `sorted_rows` | `None` (chosen by shape; `True` or `False` forces it) | `FUSED_EP_SORTED_ROWS` | the routed rows sorted into expert order by a sparse-core gather before the FFN kernel (exact; exposed, about 34 us a call at 8192 tokens on a 512-expert 4096 x 1024 shape), so the kernel fetches each tile as one copy instead of one per row (10,240 copies a call); off, the copies hide under the result push and the sort would not, which is why the copies are the default; on, the form a smaller result push stands on; whole-expert builds only |
| `tables_zero_words` | 8192 | `FUSED_EP_TABLES_ZERO_WORDS` | the fewest words of a row table the shard-tables kernel zeroes per grid step, a whole number of 1024-word tiles |
| `weight_dma_priority` | 1 | `FUSED_EP_WEIGHT_DMA_PRIORITY` | the DMA queue of the expert weight refills: 1 is off the token rows' in-order queue, 0 the same queue |
| `token_rows` | `"fp8"` | `FUSED_EP_TOKEN_ROWS` | the token rows on the wire: "fp8", one scale per row (or per token block), the largest of the kernel's roundings; "bf16" sends them as they are, twice the bytes, and the up matmul runs in bfloat16 (whole-expert builds) |
| `intermediate` | `"fp8"` | `FUSED_EP_INTERMEDIATE` | the rows between the two matmuls: "fp8" requantizes them with one scale per row (or per activation block) and runs the down matmul in fp8; "bf16" keeps them as they are and runs the down matmul in bfloat16 (whole-expert builds) |
| `result_rows` | `"fp8"` | `FUSED_EP_RESULT_ROWS` | each expert's result rows between shards: "fp8" with one scale per row; "bf16" twice the bytes and no third rounding |
| `region_push` | off | `FUSED_EP_REGION_PUSH=1` | how the results go home: off, one copy per (expert, destination) run after each expert; on, one copy per destination of the whole outgoing region after the last expert |

Every value of every field above is tested on device (`tests/test_layer.py`).
The tests read one more variable, `FUSED_EP_TEST_BATCHES`, a comma-separated
list of the token counts the shape suite runs at (unset: every count).
The defaults are the measured recipe on the models gated so far; every field is
a pipeline depth, a tables step or a rounding choice.

## Operands, and the shapes refused

Every condition below is refused with a `ValueError` naming the operand.
`python -O` strips asserts, so none of these is one.

- Weights `[experts, hidden, 2 * inter]` and `[experts, inter, hidden]` in
  the element type of the named format. Scales per output channel
  (`[experts, N]`) for fp8, per contraction block (`[experts, blocks, N]`)
  for fp4. Optional per-expert biases `[experts, 1, N]`.
- The expert count divides the mesh width. The routing tables pack an
  alignment slot next to an arrival position in one 32-bit word, and the
  slot field widens with the mesh (64 up to 10 shards, 128 at 16, 256 at
  32), so a wider mesh lowers the arrival-row count an int32 holds.
- `hidden` is a multiple of 128 (the VMEM estimate refuses a row staging that does not fit). `inter` is at
  least 128. An fp4 block is a multiple of 64 rows and divides both
  contractions.
- The tile height `tile_rows` is a multiple of 8. The arrival buffer and
  the routed-rows bound are derived from it and the batch.
- The batch has at least one token, divides by the mesh width, and admits a
  routing block of at least 8 pairs.
- A power-of-two mesh of 1 to 32 shards, and `hidden` a whole number of
  128-lane blocks. A width that is not a multiple of 1024 is staged,
  pushed and gathered with zero blocks up to the next multiple (the
  combine's sum stores eight blocks at a time and the sparse-core gather
  moves rows as 32-bit words); the sum writes the true width, and a width
  that is a multiple of 1024 builds the same program as before. The layer
  pads a shard's rows to whole 8-row blocks only (the
  transport moves 8-row blocks and quantizes 128-row tiles with a partial
  last tile; the FFN kernel sees the true rows). The select kernel and the
  combine, both cheap, work on the tokens padded to 128, and the routed
  pair list is padded on its own to whole tables steps and gather blocks
  (1024 pairs) with pairs that route nowhere: their index is
  `num_experts`, the tables give them no routed row, the gather takes
  them at arrival row zero and the combine at weight zero. The padding
  rows are dropped from the result, so any batch that divides by the mesh
  width runs.

## Numerics

Where the layer rounds, and the settings that move it:

| rounding point | default | setting |
|---|---|---|
| token rows into the up matmul | fp8, one scale per row | `activation_block`: one scale per that many values of the row (512 fits every shape gated so far) |
| the intermediate into the down matmul | fp8, one scale per row | the same setting, the same block |
| expert output between shards | fp8 rows, one scale per row | `result_rows="bf16"`: bf16 rows, no third rounding, twice the bytes |
| the scales themselves | float32 | `row_scale_dtype=bfloat16` rounds them, by fp8 requantization noise |

Each form's distance from the layer computed in float32 with nothing
rounded, on identical inputs at a 4096 x 1024 expert geometry with 512
experts and top-10 routing: one scale per
row with fp8 return 5.3e-2, with bf16 return 4.6e-2; block 512 with bf16
return (the form of a bf16 combine) 4.6e-2. The distance between two forms is
their two errors added, so matching the reference path's rounding points is
what brings the output close to it.

The fp8 scaling product is formed in float32 and rounded once
(`rowquant.apply_row_scale`), so the jitted layer, the eager reference and
the Mosaic kernels quantize alike to the bit; the f32 accumulation order
inside the matmuls and the top-k sum is the only compilation-dependent
arithmetic left, at the 1e-5 level.

The output is compared against the reference layer at the same routing.
`rel_l2 <= 1e-3` and `max_abs <= 2^-6`, one bf16 rounding of the output, is
the bound a change must stay within.
Every setting is tested against the reference at the same rounding; the
shard tables are held bit for bit to their reference form, and every
pipeline-depth setting bit for bit to the default's output. The reference
tolerance alone (8e-2) does not tell a wrong rounding scheme from a right
one, which is why the rounding is defined in float32 (rowquant.py) and
the bitwise checks exist.

## Gating a change

Every change to the package runs, on an idle device:

- `tests/`: the layer against `reference.py` at small shapes, every
  configurable value, the refusals, and the shard-tables kernel bit for
  bit against its reference form. `pytest simply/kernels/fused_ep/tests`
  on eight devices. The CPU-side files run without a TPU
  (`JAX_PLATFORMS=cpu`, about 20 seconds); the device suite at 64 and 8192
  tokens (`FUSED_EP_TEST_BATCHES=64,8192`) takes about 45 minutes; the full
  shape suite at every token count longer.
- the served model against the ra2a path on the same checkpoint and the same
  prompts, greedy, before a served number is recorded (the served numbers
  are kept outside the repository).

## Toolchain

Built and gated on jax 0.10.2 with libtpu 0.0.43, on TPU v7 (Ironwood) only.
`device.SUPPORTED_GENERATIONS` states it, and another generation is refused
by name until it has been tested there. Two places reach into jax's private
API and refuse by name on a jax that moved them, the four-bit weight stream's
ref-level bitcast (ffn_kernel.py) and the sparse-core programs' core-map
lowering (row_gather_kernel.py).
