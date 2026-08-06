# DeToNation Integration: Bug Fixes & Benchmark Rewrite

## Summary

This branch (`feat-deto`) merges the DeToNation replicator implementation into
Megatron-LM's `MegatronFSDP` gradient pipeline.  Six bugs were identified by
comparing against the reference DeToNATION repository
(https://github.com/schneiderkamplab/DeToNATION), fixed, and verified with a
full 2-GPU training loop for all six replicator strategies.

---

## Bug Fixes

### 1. `full.py` — Broken async all-reduce with one-step delay (Critical)

**File:** `megatron/core/distributed/fsdp/src/megatron_fsdp/replicators/full.py`

**Problem:** The original code had a syntax error on the `wait_pending`
signature and dead code where the resolved all-reduce result was never copied
back into the gradient buffer.  The async one-step delay logic was also
incorrectly structured.

**Fix:** Complete rewrite:
- Fixed `wait_pending` to properly resolve pending async all-reduce work and
  copy the result into the gradient buffer.
- Implemented proper async all-reduce with one-step delay: `pre_step()` resolves
  the previous step's pending work, the current step launches a new async
  `all_reduce`, and `post_step()` stores the work handle for next-step
  resolution.
- The one-step delay is intentional — it allows communication to overlap
  with computation, which is the value-add of the Megatron integration over
  the reference's synchronous approach.

### 2. `compression_utils.py` — Per-chunk top-k not implemented (Critical)

**File:** `megatron/core/distributed/fsdp/src/megatron_fsdp/replicators/compression_utils.py`

**Problem:** `DCTBufferCompress.compress()` did not accept a `chunk_size`
parameter.  The reference implementation performs top-k selection **per chunk**
(not globally), which is essential for the DCT compression strategy to work
correctly.

**Fix:**
- Added `chunk_size` parameter to `compress()`.
- When `chunk_size > 0`: reshape the buffer to `(num_chunks, chunk_size)`,
  call `torch.topk` per row, then convert local indices back to global
  indices.
- When `chunk_size == 0`: falls back to global top-k (original behavior).

### 3. `demo.py` — Missing `chunk_size` argument in compress call (Critical)

**File:** `megatron/core/distributed/fsdp/src/megatron_fsdp/replicators/demo.py`

**Problem:** `replicate_bucket_group()` called `self._transform.compress()`
without passing the chunk size, so per-chunk top-k was never activated even
after the `compression_utils.py` fix.

**Fix:** Updated the call to pass
`chunk_size=self._transform.chunk_sizes[bucket_id]`.

### 4. `random.py` — Inverted delta mask and wrong update ordering (Critical)

**File:** `megatron/core/distributed/fsdp/src/megatron_fsdp/replicators/random.py`

**Problem:** Two bugs:
1. **Inverted delta mask:** The code zeroed the selected elements instead of
   keeping them: `delta[~mask] = 0` should be `delta[mask] = 0` (we keep the
   randomly selected elements and zero the rest).
2. **Wrong delta update ordering:** The delta buffer was updated *after*
   resolving pending work, but it should be updated *before*, so that the
   current step's delta reflects the accumulated gradient history.

**Fix:** Complete rewrite:
- Fixed delta mask: `delta[mask] = 0` (zero unselected, keep selected).
- Fixed ordering: delta update happens in `replicate_bucket_group()` before
  resolving pending work.
- Simplified `wait_pending` to a stub (random replicator is synchronous, no
  pending work to resolve).

### 5. `megatron_fsdp.py` — Standalone replication path never triggered (Critical)

**File:** `megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py`

**Problem:** Two call sites in `MegatronFSDP` (lines ~782 and ~898) passed
`outer_fsdp_group_grad_reduce=self.dist_index.use_hybrid_fsdp` to
`GradReducePipeline.reduce_gradients()`.  In standalone replication mode
(non-HSDP, `dp_outer_dim=None`), `use_hybrid_fsdp` is `False`, so the
`outer_fsdp_group_grad_reduce` flag was never set to `True`.  This meant the
replicator path in `GradReducePipeline._bucket_group_gradient_reduce()` (line
3717-3737) was never executed, even though the `GradReducePipeline.__init__`
correctly set `self.outer_fsdp_group_grad_reduce = True` for standalone mode.

**Fix:** Updated both call sites to also check `self.standalone_replication_group is not None`:

```python
outer_fsdp_group_grad_reduce=(
    (self.dist_index.use_hybrid_fsdp or self.standalone_replication_group is not None)
    and (is_last_microbatch or self.model_auto_sync)
)
```

This ensures the replicator path is activated when a standalone replication
group exists, even without HSDP.

---

## Benchmark Rewrite

### `run_benchmark.py` — Full rewrite to use Megatron-FSDP APIs

**File:** `tests/detonation_benchmarks/run_benchmark.py`

**Problem:** The original benchmark script had four critical integration gaps:
1. Used `DistributedDataParallel` directly, which has no replicator path.
2. Imported `from megatron.bridge import AutoBridge` — a module that doesn't
   exist in this codebase.
3. Never called `fully_shard_optimizer()`, so the learning rate provider stayed
   at 0.0, meaning gradients never accumulated (delta = decay * delta + 0 * grad).
4. `data_parallel_sharding_strategy` defaulted to `no_shard`, but FSDP requires
   `optim_grads_params` for the full gradient pipeline.

**Fix:** Complete rewrite using the native Megatron-FSDP API:
- Uses `fully_shard_model()` + `fully_shard_optimizer()` from
  `megatron.core.distributed.fsdp.src.megatron_fsdp`.
- No HuggingFace download — uses a native `FakeModel` (embedding + 2 MLP layers
  + output projection, ~4.2M params) with `fsdp_unit_modules=[FakeMLP]`.
- `zero_dp_strategy="optim_grads_params"` (ZeRO-3) exercises the full FSDP
  path: parameter sharding, gradient reduce-scatter, and outer-DP replication.
- No `device_mesh` provided → `fully_shard_model` creates a 1-D mesh
  `(world_size, 1)` with `dp_outer_dim=None` → standalone replication mode.
- `fully_shard_optimizer()` wraps the optimizer to call `finish_grad_sync()`
  (→ `replicator.pre_step()`) and `zero_grad_buffer()` (→
  `replicator.post_step()`), and updates the learning rate provider before
  each step.

---

## Test Results

### Standalone replicator test (`test_replicators.py`)

```
torchrun --nproc-per-node=2 tests/detonation_benchmarks/test_replicators.py
```

All 6/6 strategies PASS: none, full, demo, slicing, striding, random.

### Full training loop benchmark (`run_benchmark.py`)

```
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. torchrun --nproc-per-node=2 \
    tests/detonation_benchmarks/run_benchmark.py \
    --config tests/detonation_benchmarks/configs/experiments/smoke_test.yaml \
    --replicator-idx <N>
```

| Idx | Strategy  | Status | Throughput (iters/s) |
|-----|-----------|--------|----------------------|
| 0   | none      | PASS   | 102.76               |
| 1   | full      | PASS   | 101.42               |
| 2   | demo      | PASS   | 75.44                |
| 0*  | slicing   | PASS   | 90.83                |
| 1*  | striding  | PASS   | 84.47                |
| 2*  | random    | PASS   | 18.18                |

\* slicing/striding/random tested with a custom sweep config.

---

## Files Changed

| File | Type | Description |
|------|------|-------------|
| `replicators/full.py` | Bug fix | Rewritten: async all-reduce with one-step delay |
| `replicators/compression_utils.py` | Bug fix | Per-chunk top-k in `DCTBufferCompress.compress()` |
| `replicators/demo.py` | Bug fix | Pass `chunk_size` to `compress()` |
| `replicators/random.py` | Bug fix | Rewritten: fixed inverted mask, fixed delta ordering |
| `megatron_fsdp.py` | Bug fix | Enable `outer_fsdp_group_grad_reduce` for standalone replication |
| `tests/detonation_benchmarks/run_benchmark.py` | Rewrite | Use `fully_shard_model()` + `fully_shard_optimizer()` API |
| `tests/detonation_benchmarks/test_replicators.py` | New | Standalone 2-GPU replicator test (6/6 PASS) |

## Files Unchanged (Verified Correct)

- `replicators/base.py` — `BucketReplicator` ABC, `DeltaBufferManager`
- `replicators/slicing.py` — slicing strategy
- `replicators/striding.py` — striding strategy
- `replicators/__init__.py` — `get_replicator()` factory
