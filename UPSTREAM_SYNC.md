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

### Batch 2 - merged 2026-08-09 (15 commits, all verified)

| Upstream commit | Subject | Merged | Verified |
|---|---|---|---|
| `9a688e51e` | fit: Fix memory allocation for MTP layers (#26605) | yes | swap+MTP ok |
| `69bf64379` | CUDA: fix thread/block count in quantized cpy kernel launches (#26731) | yes | build ok |
| `9bd4c09ea` | CUDA: fix SMEM data-races in block_reduce (norm/softmax) (#26385) | yes | build ok |
| `1269cb1ff` | model: allow reshape of tensors during load (#26531) | yes | build ok |
| `3db4ff877` | model-loader: fix quantized reshaped tensor strides (#26672) | yes | build ok |
| `7bd8282c3` | speculative: refactor common_speculative_init (#26510) | yes | MTP ok |
| `a035a8887` | server: spec-decode counters on /metrics (#26389) | yes | build ok |
| `96278e39f` | CUDA: backend sampler for penalties sampler (#25262) | yes | swap ok |
| `935cad649` | llama: move n_vocab to penalty_sampler (#26520) | yes | swap ok |
| `a6aa6f545` | sampler: remove "full-context windows" from history samplers (#26524) | yes | swap ok |
| `f2b52a87e` | server: (tools) add x-tool-cwd header (#26420) | yes | build ok |
| `99111b19c` | server: add get_info tool (#26522) | yes | build ok |
| `2f56fc343` | ui: CWD for agent (#26518) | yes | build ok |
| `4308a4f03` | server: decode Windows OEM output to UTF-8 in built-in tools (#26597) | yes | build ok |
| `f9e832c10` | server: harden file_glob_search directory walk (#26626) | yes | build ok |

Notes: sampler chain applied in upstream order (96278e39f -> 935cad649 ->
a6aa6f545); conflicts were pure line drift, resolved by taking upstream hunks
and keeping our reasoning-budget clamp in server-schema.cpp. server-tools
commits applied as a chain (we never modified that file).

## Pending - remaining candidates

| Upstream commit | Subject | Applies? | Decision |
|---|---|---|---|
| `dd2c7c447` | server: docker tool isolation (#26507) | CONFLICT | SKIP - new subsystem, not a fix |
| `18f7ad7fc` | server, ui: working dir only when a tool reads it (#26762) | CONFLICT | SKIP - depends on isolation |
| `7ba604f1c` | server: report isolate working directory (#26773) | clean | SKIP - depends on isolation |
| `0b14b87d7` | server: port 8080 -> 9931 notice (#26508) | clean | SKIP - we pin 8081 |

### Skipped - not relevant

SYCL / Vulkan / Metal / WebGPU, mtmd / OCR / TTS, UI / webui, CI, convert
scripts, vendor bumps (BoringSSL, cpp-httplib), ggml version bumps, grammar,
security docs, tests. 87 of the 98 originally-missing commits; the remainder
are tracked above (4 isolation/port commits).

## Conflict resolution notes

- `935cad649` / `a6aa6f545`: sampler refactors; our only sampler additions are
  ftype enums (`LLAMA_FTYPE_MOSTLY_TBQ3_0/TBQ4_0`) + unrelated API, no semantic
  overlap - conflicts were pure line drift, resolved by taking upstream hunks.
  NOTE: `96278e39f` (CUDA backend sampler) must be applied FIRST - it is a
  prerequisite for `935cad649` (n_vocab lives in the sampler struct upstream).
  Keep our reasoning-budget clamp in server-schema.cpp when resolving
  `a6aa6f545` (upstream deletes the `-1 -> ctx-size` fallback there; samplers
  now clamp `-1 -> 0` at init).
- `3db4ff877`: upstream strides fix collides with our `llama_model_loader_apply_dspark_mapping`
  (180 added lines in llama-model-loader.cpp) - line shift only; apply AFTER
  `1269cb1ff` (reshape commit) and it applies cleanly.
- `f9e832c10`: we did NOT modify server-tools.cpp; apply the 5-commit tool
  chain (f2b52a87e -> 99111b19c -> 2f56fc343 -> 4308a4f03 -> f9e832c10) in
  order instead of cherry-picking the last commit alone.

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
| 2026-08-09 | upstream | 15 | merged, build + swap + MTP verified, pushed e0858931d |
| 2026-08-09 | borrow source (origin) | 0 | fully contained, nothing to merge |
