# Upstream Sync Tracker

Living document for tracking merges from upstream llama.cpp (and the borrow
source) into this fork. Use this instead of digging through git history to
recover context. Update it on every merge batch.

## How to use

1. When merging upstream commits, add a row to the batch table with the
   upstream commit hash, subject, and what we verified after the merge.
2. Keep the "Pending" tables as the to-do queue.
3. After a full sync batch, append a "Sync history" entry and re-baseline the
   "Current baseline" section.

## Current baseline

| Item | Value |
|---|---|
| Branch | `feature/model-swap-rebased` |
| Rebase base | `a7a6d0d26` (b10217, May 13 2026) |
| Upstream compared against | `ggml/master` (fetched 2026-08-09, `08659901c`) |
| Commits behind upstream | 111 (13 absorbed, 98 missing) |
| Borrow source | Indras-Mirror/llama.cpp-turboq-mtp (`origin`) - fully contained in our `master`, 0 commits we lack |
| Push remote | `myfork` (jtrefon/llama.cpp-turboq-mtp) |

## Merged from upstream

### Batch 1 - Aug 2 2026 (absorbed via sync/absorb-dsv4, already in HEAD)

| Upstream commit | Subject | Notes |
|---|---|---|
| `596a5795b` | DeepseekV4 MTP + DSpark (#25784) | Core DSV4 support |
| `f5919bf45` | chat: add qwen3 specialized parser (#26252) | |
| `75587a05b` | model: load MiMo V2 MTP tensors only if used (#26412) | |
| `bb4e0e1b3` | common: support DSpark sidecar resolution (#26458) | |
| `3581ba0cf` | convert: option to create separate dspark GGUF (#26452) | |
| `fffbcbdb9` | metal: DeepSeek V4 hyper-connections (#26459) | not used (CUDA) |
| `0ab9d6fed` | opencl: GLU workgroup limit (#26383) | not used |
| `9d21b57f2` | metal: F16 bin ops (#26465) | not used |
| `221f0f635` | metal: SILU_BACK (#25982) | not used |
| `272700b36` | sycl: iGPU classification (#26105) | not used |
| `7a2db1a0c` | ggml-webgpu: f16 repeat (#26307) | not used |
| `c745be2a2` | opencl: ref_count fix (#26162) | not used |
| `11924d4c1` | test: fix CI errors (#26415) | |

### Batch 2 - planned (see Pending below)

## Pending - merge candidates (2026-08-09)

### Must merge - bug fixes that hit us directly

| Upstream commit | Subject | Applies? | Merged | Verified |
|---|---|---|---|---|
| `9a688e51e` | fit: Fix memory allocation for MTP layers (#26605) | clean | | |
| `69bf64379` | CUDA: fix thread/block count in quantized cpy kernel launches (#26731) | clean | | |
| `9bd4c09ea` | CUDA: fix SMEM data-races in block_reduce (norm/softmax) (#26385) | clean | | |
| `1269cb1ff` | model: allow reshape of tensors during load (#26531) | clean | | |

### Should merge - maintenance value

| Upstream commit | Subject | Applies? | Merged | Verified |
|---|---|---|---|---|
| `7bd8282c3` | speculative: refactor common_speculative_init (#26510) | clean | | |
| `a035a8887` | server: spec-decode counters on /metrics (#26389) | clean | | |
| `935cad649` | llama: move n_vocab to penalty_sampler (#26520) | CONFLICT (drift) | | |
| `a6aa6f545` | sampler: remove "full-context windows" from history samplers (#26524) | CONFLICT (drift) | | |

### Optional

| Upstream commit | Subject | Applies? | Merged | Verified |
|---|---|---|---|---|
| `3db4ff877` | model-loader: fix quantized reshaped tensor strides (#26672) | CONFLICT (our DSpark mapping) | | |
| `f9e832c10` | server: harden file_glob_search directory walk (#26626) | CONFLICT (drift) | | |
| `96278e39f` | CUDA: backend sampler for penalties sampler (#25262) | clean, large | | |

### Skipped - not relevant

SYCL / Vulkan / Metal / WebGPU, mtmd / OCR / TTS, UI / webui, CI, convert
scripts, vendor bumps (BoringSSL, cpp-httplib), ggml version bumps, grammar,
security docs, tests. Full list: 87 of the 98 missing commits.
`0b14b87d7` (upstream default port 8080 -> 9931 notice) skipped - we pin 8081.

## Conflict resolution notes

- `935cad649` / `a6aa6f545`: sampler refactors; our only sampler additions are
  ftype enums (`LLAMA_FTYPE_MOSTLY_TBQ3_0/TBQ4_0`) + unrelated API, no semantic
  overlap - conflicts are pure line drift, resolve by taking upstream hunks.
- `3db4ff877`: upstream strides fix collides with our `llama_model_loader_apply_dspark_mapping`
  (180 added lines in llama-model-loader.cpp) - line shift only.
- `f9e832c10`: we did NOT modify server-tools.cpp; conflict is upstream drift
  in surrounding code.

## Fork-local changes to re-verify after any sync

- `common/speculative.cpp`: MTP loop breaker (FULL_ACCEPT_LIMIT), pending_h
  reset on reload/position discontinuity - lives at lines ~1277-1600, does NOT
  overlap upstream's `common_speculative_init` (~2505).
- `common/arg.cpp`: "--spec-type none" resets types (preset override).
- `tools/server/server-context.cpp`: in-process swap (stale-alias fix at swap
  handler, spec-type none in `[*]`).
- `models.ini`: per-model ctk/ctv, spec-type, ctx-checkpoints.
- `ggml/src/ggml-cuda/`: TBQ4/TBQ3 fused FA kernels (fattn-mma-tbq*.cuh),
  cpy-planar-iso, concat, gated_delta_net_pipe.

## Sync history

| Date | From | Commits | Result |
|---|---|---|---|
| 2026-08-02 | upstream (absorb) | 13 | in HEAD |
| (pending) | upstream | 4 must + 4 should + 3 optional | |
