# MTP `ctx_mtp` Desync — Root Cause & Fix

**Status:** Fixed (implemented, rebuilding, pending restart + test)
**Component:** `common/speculative.cpp` (`common_speculative_state_mtp`) + `src/llama-context.cpp` (`handle_mtp_for_ubatch`)
**Symptom in logs:** flood of
```
handle_mtp_for_ubatch: MTP hook SKIP — ctx_mtp is ahead of target (target pos=…, ctx_mtp pos_max=…, gap=1..3)
```
every ubatch during decode, with gaps cycling 1→2→3 (== `--spec-draft-n-max`).

---

## 1. Architecture (how MTP works in this fork)

The MTP draft model runs in a **separate context `ctx_mtp`** registered on the
target via `llama_set_mtp(ctx_tgt, ctx_mtp)` (`common/speculative.cpp:417`).
`ctx_mtp`'s KV cache must mirror the **target's real hidden states** so the draft
head can predict the next token from the correct context.

Two code paths write `ctx_mtp`'s KV:

- **Writer A — the streaming hook** `handle_mtp_for_ubatch` (`src/llama-context.cpp:3592`),
  invoked from `process_ubatch` **during the target's own decode**. It mirrors the
  target's `t_h_pre_norm` hidden states into `ctx_mtp` position-by-position. This
  is the intended *source of truth*.
- **Writer B — the draft auto-regression loop**
  `common_speculative_state_mtp::draft` (`common/speculative.cpp`), which calls
  `llama_decode(ctx_mtp, …)` and **extends `ctx_mtp` ahead of the target** by up to
  `n_max` positions *before* the target has decoded those tokens.

### Per-step pipeline order (`tools/server/server-context.cpp`)
1. `common_speculative_draft()` — `draft()` runs **first**, extending `ctx_mtp` to
   `T + n_draft` (ahead of target `T`).  (`:2354`)
2. Target re-decodes the verification batch (anchor + draft tokens). The hook
   (`Writer A`) fires per ubatch.  (`:3096`)
3. `common_speculative_accept()` drops the un-accepted draft tail from `ctx_mtp`.  (`:3296`)

---

## 2. Root cause

The hook's guard was *skip-on-ahead*:

```cpp
const llama_pos pos_max_mtp = llama_memory_seq_pos_max(llama_get_memory(mtp.ctx_mtp), 0);
if (pos_start <= pos_max_mtp) {
    LLAMA_LOG_WARN("%s: MTP hook SKIP — ctx_mtp is ahead of target …");
    return;   // <-- never mirrors, never heals
}
```

Because `draft()` (Writer B) **always runs before** the target decode and pushes
`ctx_mtp` ahead, the target's verification decode finds `ctx_mtp` already ahead →
the hook **permanently SKIPs**. `ctx_mtp` therefore never receives the target's
*true* hidden states at the accepted positions; it keeps the draft model's own
self-predicted hidden states.

The per-step reconciliation in `accept()` (`common/speculative.cpp`) compounded it
with an off-by-one:

```cpp
const int32_t n_to_drop = std::max(0, n_drafted_last - (int32_t) n_accepted - 1);  // WRONG
```

It dropped one *fewer* row than it should, leaving `ctx_mtp` exactly **one
position ahead** of the target's real last position after every step. Both writers
advance at the same rate, so the gap stays a constant 1–3 (`spec-draft-n-max`),
which is exactly the `gap=1..3` emitted on every ubatch.

Commit `245213f` ("fix MTP shadow-context synchronization") only mirrors
`seq_rm`/`seq_add` from the target context. That helps the *partial-accept /
context-trim* paths (`server-context.cpp:3261`) but the **normal full-accept path
never calls `seq_rm` on the target**, so it still relied on the broken `accept()`
drop. That is why the desync returned.

**Net effect:** `ctx_mtp` permanently runs 1–3 positions ahead; its KV at the
accepted positions holds the draft model's self-predicted hidden states instead of
the target's real ones → degraded / degenerate drafting (the repetitive-loop
signature in the agent workload).

> Note: the *separate* observation in the logs — context growing monotonically
> (`107874 → 111986`, ~344 tok/request) with no idle between turns — is the
> upstream client driving the model in a tight request loop and never trimming
> history. That is a driver/agent concern, **not** the inference crash. The MTP
> desync above is the actual crash/instability sign.

---

## 3. Fix (root cause, not a band-aid)

1. **Hook is now authoritative & self-healing** (`src/llama-context.cpp`,
   `handle_mtp_for_ubatch`). When `ctx_mtp` is ahead, *truncate* it to
   `pos_start - 1` (`llama_memory_seq_rm(ctx_mtp, 0, pos_start, -1)`) and fall
   through to re-mirror the target's real hidden states, instead of skipping.
   This converges `ctx_mtp` to the target's truth on every decode regardless of
   what the draft loop wrote. The old `LLAMA_LOG_WARN` flooding is removed.

2. **Correct the `accept()` reconciliation** (`common/speculative.cpp`):
   ```cpp
   const int32_t n_to_drop = std::max(0, n_drafted_last - (int32_t) n_accepted);
   ```
   Drops the *entire* un-accepted tail, leaving `ctx_mtp == [0, T+a]` (the
   target's real frontier) at rest.

3. **Re-anchor `ctx_mtp` at the start of each `draft()`**
   (`common/speculative.cpp`, replacing the old buggy tail-drop). Instead of
   assuming exactly one accepted draft (`n_drafted_last - 1`, which could discard
   *accepted* positions), reset `ctx_mtp` to the target's real frontier
   (`dp.n_past`) before extending:
   ```cpp
   const llama_pos pos_frontier = dp.n_past;
   const llama_pos pos_max      = llama_memory_seq_pos_max(llama_get_memory(ctx_mtp), 0);
   if (pos_max >= pos_frontier) {
       llama_memory_seq_rm(llama_get_memory(ctx_mtp), 0, pos_frontier, -1);
   }
   ```
   In the normal path this is a no-op (ctx_mtp already at the frontier); it only
   corrects desync / the `n_accepted == 0` edge case.

The MTP math (pending-hook scheme, draft sampling) is unchanged, and the GPU /
Flash-Attention / TBQ4 KV path is untouched.

---

## 4. Verification

- **Build:** `cmake --build build -j"$(nproc)" --target llama-server`
- **Restart:** `sudo systemctl restart llama-turboq`
- **Pass criteria:**
  - `journalctl -u llama-turboq.service` shows **no** `MTP hook SKIP` lines.
  - `draft acceptance rate` remains in its previous range (~0.7–0.99) — fix should
    not regress acceptance; ideally it improves as `ctx_mtp` KV is now correct.
  - Long run (the agent loop) shows `ctx_mtp` position staying *behind-or-equal*
    the target, i.e. desync gone.
- **Rollback (source):** `git restore src/llama-context.cpp common/speculative.cpp`
  then rebuild + restart. (Note: the prior "MTP hook SKIP" `LLAMA_LOG_WARN` was an
  *uncommitted* diagnostic; this fix removes it.)

---

## 5. Out of scope

- Adaptive draft depth (`--spec-adapt`) — unrelated, left as-is.
- Agent-side request loop / unbounded context growth — separate driver concern.
- EAGLE3 / FastMTP model-side acceptance improvements — see `PROPOSAL-mtp-acceptance.md`.
