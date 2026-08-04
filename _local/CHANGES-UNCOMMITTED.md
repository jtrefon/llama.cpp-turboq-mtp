# Uncommitted Changes & Revert Guide

> **Scope:** All changes listed below are **UNCOMMITTED working-tree edits** on branch
> `master` (HEAD `aa8ee2d`). The 4 earlier fixes (MTP OOB crash, fused chunk pipeline,
> message-aware trimming, MTP docs) ARE committed and are **not** reverted by this guide.
>
> The running production binary (systemd `llama-turboq.service`) was built from these
> edits, so reverting source requires a **rebuild + service restart** to take effect.

## Global one-shot revert (returns tree to HEAD `aa8ee2d`)

```sh
cd /home/jack/llm/llama-turboq
git restore README.md common/common.cpp common/common.h include/llama.h \
            src/llama-context.cpp src/llama-context.h \
            src/llama-kv-cache.cpp src/models/delta-net-base.cpp
rm -f ARCHITECTURE-pcore-binding.md
cmake --build build -j"$(nproc)" --target llama-server
sudo systemctl restart llama-turboq
```

---

## Change 1 — Async checkpoint capture (NVIDIA-dedicated / RTX 4090)

- **Purpose:** overlap D2H state capture with prefill compute. Measured **+14% prefill
  throughput** (600 vs 524 tok/s) via the binary's own `timings.prompt_per_second`.
- **Files touched:**
  - `README.md` (+30) — NVIDIA-dedicated banner + async checkpoint docs
  - `common/common.cpp` (+269) — `ckpt_staging_pool`, `ckpt_host_cb`, async update/load/clear/move/copy, dtor
  - `common/common.h` (+31) — async members; `checkpoint_every_nt` (currently 8192)
  - `include/llama.h` (+20) — `llama_state_seq_get_data_ext_async`, `llama_state_seq_capture_wait`
  - `src/llama-context.cpp` (+137) — `state_seq_get_data_async`, `llama_io_write_host_async`, `llama_state_seq_capture_wait`, `CUDA_CHECK`
  - `src/llama-context.h` (+5) — `state_seq_get_data_async` decl
- **Revert (source):**
  ```sh
  git restore README.md common/common.cpp common/common.h include/llama.h \
              src/llama-context.cpp src/llama-context.h
  cmake --build build -j"$(nproc)" --target llama-server
  sudo systemctl restart llama-turboq
  ```
- **Note:** IDE/LSP may flag `state_seq_get_data_async` as private in `llama_context`;
  the binary **compiles and runs** (build verified). This is a stale-index warning, not a
  build error.

---

## Change 2 — KV pad floor (this session)

- **Purpose:** reduce CUDA-graph re-capture frequency. **Measured marginal**: re-captures
  138 → 115 (~17% fewer) over 4096 tokens, but **no tail-latency improvement** (max spike
  ~50 ms in both) because re-captures are dominated by **MTP draft-graph variation**, not
  n_kv padding.
- **File:** `src/llama-kv-cache.cpp:1134` — `n_pad_cur = std::max(n_pad, n_pad_min)` where
  `n_pad_min` defaults to **4096** (was hardcoded 256), overridable via env
  `LLAMA_KV_PAD_MIN`.
- **Revert without rebuild** (keep source, restore original behavior): set the env for the
  service:
  ```ini
  # in /etc/systemd/system/llama-turboq.service [Service]
  Environment=LLAMA_KV_PAD_MIN=256
  ```
  then `sudo systemctl daemon-reload && sudo systemctl restart llama-turboq`.
- **Revert with rebuild** (remove the edit entirely):
  ```sh
  git restore src/llama-kv-cache.cpp
  cmake --build build -j"$(nproc)" --target llama-server
  sudo systemctl restart llama-turboq
  ```

---

## Change 3 — `delta-net-base.cpp` comment (doc-only, NO-OP)

- **Purpose:** clarifying comment only; **no functional change**.
- **File:** `src/models/delta-net-base.cpp` (+2 lines inside `build_delta_net_base`,
  `fused_gdn_ch` branch).
- **Revert:** `git restore src/models/delta-net-base.cpp` (no rebuild strictly required,
  but rebuild+restart to be safe).

---

## Change 4 — `ARCHITECTURE-pcore-binding.md` (untracked proposal doc)

- **Purpose:** the P-core binding architecture proposal (not code).
- **File:** `ARCHITECTURE-pcore-binding.md` (untracked).
- **Revert:** `rm -f ARCHITECTURE-pcore-binding.md`.

---

## How to confirm a revert took effect
- Source revert: `git status --short` should be empty (except possibly the untracked doc).
- Behavior revert: check the running binary's `timings` / verbose log, or re-run the
  prefill benchmark (`prompt_per_second` ~524 = reverted async; ~600 = async present) and
  the KV-pad A/B (re-capture count ~138 at pad 256 vs ~115 at pad 4096).
