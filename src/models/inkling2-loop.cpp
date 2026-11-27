#include "models.h"
#include "../llama-memory-hybrid-iswa.h"

// relative positions for the Inkling 2 full-attention layers, shared by the original and the looped graph
class llm_graph_input_inkling2_loop_pos : public llm_graph_input_i {
public:
    const llama_kv_cache_context * cache[2];
    ggml_tensor * distance[2];
    uint32_t extent[2];

    bool can_reuse(const llm_graph_params & params) override {
        const auto * memory = static_cast<const llama_memory_hybrid_iswa_context *>(params.mctx);
        cache[0] = memory->get_attn()->get_base();
        cache[1] = memory->get_attn()->get_swa();
        for (int i = 0; i < 2; ++i) {
            if (distance[i]->ne[0] != cache[i]->get_n_kv() || distance[i]->ne[1] != params.ubatch.n_tokens) {
                return false;
            }
        }
        return true;
    }

    void set_input(const llama_ubatch * ubatch) override {
        for (int i = 0; i < 2; ++i) {
            if (distance[i]->buffer) {
                cache[i]->set_input_relative_pos(distance[i], ubatch, extent[i]);
            }
        }
    }
};

int llama_model_inkling2_loop::layer_phys(int il) const {
    return il < n_phys ? il : core_start + (il - n_phys) % (core_end - core_start);
}

void llama_model_inkling2_loop::load_arch_hparams(llama_model_loader & ml) {
    llama_model_inkling2::load_arch_hparams(ml);

    ml.get_key(LLM_KV_LOOP_START,     core_start);
    ml.get_key(LLM_KV_LOOP_END,       core_end);
    ml.get_key(LLM_KV_LOOP_COUNT,     n_passes);
    ml.get_key(LLM_KV_LOOP_GATE_RANK, gate_rank);
    ml.get_key(LLM_KV_LOOP_VARYING,   varying, false);

    n_phys = hparams.n_layer();
    const uint32_t block = hparams.attn_res_block_size;
    // a fixed loop needs at least 2 passes, a variable loop at least 1
    if (n_passes < (varying ? 1u : 2u) || !gate_rank || !core_start || core_start >= core_end || core_end > (uint32_t) n_phys ||
        core_start % block || core_end % block || hparams.n_layer_nextn) {
        throw std::runtime_error(format("inkling2_loop: invalid loop core [%u, %u) with block size %u, %u passes or gate rank %u for %d layers",
                core_start, core_end, block, n_passes, gate_rank, n_phys));
    }
    const size_t n_slot = n_phys + (size_t) (n_passes - 1)*(core_end - core_start);
    if (n_slot > LLAMA_MAX_LAYERS) {
        throw std::runtime_error(format("inkling2_loop: %zu cache slots exceed the maximum of %d", n_slot, LLAMA_MAX_LAYERS));
    }

    // expand to the cache slots before load_tensors() allocates the layers and the memory
    hparams.n_layer_all = n_slot;
    for (int il = n_phys; il < (int) n_slot; ++il) {
        const int p = layer_phys(il);
        hparams.n_head_arr[il]        = hparams.n_head_arr[p];
        hparams.n_head_kv_arr[il]     = hparams.n_head_kv_arr[p];
        hparams.n_ff_arr[il]          = hparams.n_ff_arr[p];
        hparams.n_ff_exp_arr[il]      = hparams.n_ff_exp_arr[p];
        hparams.n_expert_used_arr[il] = hparams.n_expert_used_arr[p];
        hparams.is_swa_impl[il]       = hparams.is_swa_impl[p];
        hparams.is_recr_impl[il]      = hparams.is_recr_impl[p];
    }
}

void llama_model_inkling2_loop::load_arch_tensors(llama_model_loader & ml) {
    llama_model_inkling2::load_arch_tensors(ml);

    const int64_t n_embd = hparams.n_embd;
    const int64_t rank   = gate_rank;
    const llm_tensor down[LOOP_CTRL_COUNT]   = { LLM_TENSOR_LOOP_Q_CTRL_DOWN,   LLM_TENSOR_LOOP_K_CTRL_DOWN,   LLM_TENSOR_LOOP_V_CTRL_DOWN,   LLM_TENSOR_LOOP_FFN_CTRL_DOWN   };
    const llm_tensor gain[LOOP_CTRL_COUNT]   = { LLM_TENSOR_LOOP_Q_CTRL_GAIN,   LLM_TENSOR_LOOP_K_CTRL_GAIN,   LLM_TENSOR_LOOP_V_CTRL_GAIN,   LLM_TENSOR_LOOP_FFN_CTRL_GAIN   };
    const llm_tensor forget[LOOP_CTRL_COUNT] = { LLM_TENSOR_LOOP_Q_CTRL_FORGET, LLM_TENSOR_LOOP_K_CTRL_FORGET, LLM_TENSOR_LOOP_V_CTRL_FORGET, LLM_TENSOR_LOOP_FFN_CTRL_FORGET };

    loop_layers.resize(core_end - core_start);
    for (uint32_t il = core_start; il < core_end; ++il) {
        auto & l = loop_layers[il - core_start];
        // the q, k, v and ffn message queries
        l.msg_query = create_tensor(tn(LLM_TENSOR_LOOP_MSG_QUERY, "weight", il), {n_embd, LOOP_CTRL_COUNT}, 0);
        for (int c = 0; c < LOOP_CTRL_COUNT; ++c) {
            l.ctrl_down[c]   = create_tensor(tn(down[c],   "weight", il), {2*n_embd, rank}, 0);
            l.ctrl_gain[c]   = create_tensor(tn(gain[c],   "weight", il), {rank, n_embd}, 0);
            l.ctrl_forget[c] = create_tensor(tn(forget[c], "weight", il), {rank, n_embd}, 0);
        }
    }
}

void llama_model_inkling2_loop::post_arch_tensors(llama_model_loader &) {
    // share the physical weights, each cache slot keeps its own KV and conv state
    for (int il = n_phys; il < (int) hparams.n_layer(); ++il) {
        layers[il] = layers[layer_phys(il)];
    }
}

std::unique_ptr<llm_graph_context> llama_model_inkling2_loop::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

ggml_tensor * llama_model_inkling2_loop::graph::clip(ggml_tensor * cur, ggml_tensor * scale, float threshold) {
    if (scale) {
        cur = ggml_mul(ctx0, cur, scale);
    }
    auto * norm = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, cur)));
    norm = ggml_scale_bias(ctx0, norm, 1.0f, 1e-6f);
    auto * factor = ggml_div(ctx0, ggml_fill(ctx0, norm, threshold), norm);
    return ggml_mul(ctx0, cur, ggml_clamp(ctx0, factor, 0.0f, 1.0f));
}

ggml_tensor * llama_model_inkling2_loop::graph::mix(const std::vector<ggml_tensor *> & blocks, ggml_tensor * cur, ggml_tensor * query) {
    auto * states = ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens);
    for (auto * block : blocks) {
        states = ggml_concat(ctx0, ggml_reshape_3d(ctx0, block, n_embd, 1, n_tokens), states, 1);
    }
    states = ggml_rms_norm(ctx0, states, 1e-6f);
    auto * queries = ggml_repeat(ctx0, query, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, query->ne[1], n_tokens));
    auto * scores = ggml_mul_mat(ctx0, states, queries);
    scores = ggml_soft_max_ext(ctx0, scores, nullptr, 1.0f/sqrtf(n_embd), 0.0f);
    // [n_queries, n_embd, n_tokens], the queries live in dim 0 so views stay inside the source
    return ggml_mul_mat(ctx0, scores, ggml_cont(ctx0, ggml_transpose(ctx0, states)));
}

ggml_tensor * llama_model_inkling2_loop::graph::convolve(ggml_tensor * cur, ggml_tensor * weight, ggml_tensor * dynamic,
        ggml_tensor * state, ggml_tensor * history, llm_graph_input_rs * inp, int64_t offset) {
    const int64_t kernel = hparams.n_shortconv_l_cache;
    const int64_t channels = cur->ne[0];
    const int64_t nt = ubatch.n_seq_tokens;
    const int64_t ns = ubatch.n_seqs;
    auto * past = ggml_view_3d(ctx0, history, kernel - 1, channels, ns, history->nb[1], history->nb[2], offset*history->nb[1]);
    auto * x = ggml_transpose(ctx0, ggml_reshape_3d(ctx0, cur, channels, nt, ns));
    x = ggml_concat(ctx0, past, x, 0);
    build_conv_state_store(inp, state, x, kernel - 1, offset*(kernel - 1)*ggml_element_size(state));
    auto * out = ggml_ssm_conv(ctx0, x, weight);
    out = ggml_reshape_2d(ctx0, out, channels, n_tokens);
    if (dynamic) {
        const int64_t heads = dynamic->ne[0]/kernel;
        dynamic = ggml_reshape_3d(ctx0, dynamic, kernel, heads, n_tokens);
        for (int64_t lag = 0; lag < kernel; ++lag) {
            auto * tap = ggml_view_3d(ctx0, dynamic, 1, heads, n_tokens, dynamic->nb[1], dynamic->nb[2], lag*dynamic->nb[0]);
            auto * window = ggml_view_3d(ctx0, x, nt, channels, ns, x->nb[1], x->nb[2], (kernel - 1 - lag)*x->nb[0]);
            window = ggml_cont(ctx0, ggml_transpose(ctx0, window));
            window = ggml_reshape_3d(ctx0, window, channels/heads, heads, n_tokens);
            auto * term = ggml_reshape_2d(ctx0, ggml_mul(ctx0, window, tap), channels, n_tokens);
            out = ggml_add(ctx0, out, term);
        }
    }
    return ggml_scale(ctx0, ggml_add(ctx0, cur, out), 1.0f/sqrtf((dynamic ? 2 : 1)*kernel + 1));
}

// mix a cross-pass message into a looped layer input: (1 - forget)*cur + gain*message, both gated on cur and message
ggml_tensor * llama_model_inkling2_loop::graph::control(ggml_tensor * cur, ggml_tensor * message, const loop_layer & loop, int ctrl) {
    // cur and message as two rows of one rms_norm, then side by side: [norm(cur); norm(message)]
    ggml_tensor * gate = ggml_concat(ctx0, ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens), ggml_reshape_3d(ctx0, message, n_embd, 1, n_tokens), 1);
    gate = ggml_reshape_2d(ctx0, ggml_rms_norm(ctx0, gate, 1e-6f), 2*n_embd, n_tokens);
    gate = build_lora_mm(loop.ctrl_down[ctrl], gate);

    ggml_tensor * gain   = ggml_tanh(ctx0, build_lora_mm(loop.ctrl_gain[ctrl],   gate));
    ggml_tensor * forget = ggml_tanh(ctx0, build_lora_mm(loop.ctrl_forget[ctrl], gate));

    return ggml_add(ctx0, ggml_sub(ctx0, cur, ggml_mul(ctx0, forget, cur)), ggml_mul(ctx0, gain, message));
}

ggml_tensor * llama_model_inkling2_loop::graph::build_inputs() {
    inp  = build_inp_mem_hybrid_iswa();
    attn = inp->get_attn();
    recr = inp->get_recr();

    auto distances_in = std::make_unique<llm_graph_input_inkling2_loop_pos>();
    distances_in->extent[0] = hparams.n_relative_extent;
    distances_in->extent[1] = hparams.n_swa;
    for (int i = 0; i < 2; ++i) {
        distances_in->cache[i] = (i == 0 ? attn->mctx->get_base() : attn->mctx->get_swa());
        distances[i] = distances_in->distance[i] = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, distances_in->cache[i]->get_n_kv(), n_tokens);
        ggml_set_input(distances[i]);
    }
    res->add_input(std::move(distances_in));

    auto * positions = build_inp_pos();
    inp_out_ids = build_inp_out_ids();
    auto * tau_in = ggml_scale_bias(ctx0, ggml_cast(ctx0, positions, GGML_TYPE_F32), 1.0f, 1.0f);
    tau_in = ggml_log(ctx0, ggml_clamp(ctx0, ggml_scale(ctx0, tau_in, 1.0f/hparams.n_log_scale_floor), 1.0f, INFINITY));
    tau = ggml_reshape_3d(ctx0, ggml_scale_bias(ctx0, tau_in, hparams.f_log_scale_alpha, 1.0f), 1, 1, n_tokens);

    auto * cur = build_inp_embd(model.tok_embd);
    // important: do not normalize raw embeddings input (i.e. encoded image embeddings)
    if (ubatch.token) {
        cur = build_norm(cur, model.tok_norm, nullptr, LLM_NORM_RMS, -1);
    }
    return cur;
}

ggml_tensor * llama_model_inkling2_loop::graph::build_layer(int il, int ip, ggml_tensor * cur,
        std::vector<ggml_tensor *> & blocks, ggml_tensor * messages, const loop_layer * loop) {
    GGML_ASSERT(!messages == !loop);

    const auto & l = model.layers[il];
    const bool swa = hparams.is_swa(il);
    const int64_t nh = hparams.n_head(il);
    const int64_t nk = hparams.n_head_kv(il);
    const int64_t hd = hparams.n_embd_head_k(il);
    const int64_t kv = nk*hd;
    auto * state = recr->mctx->get_r_l(il);
    auto * history = build_rs(recr, state, hparams.n_embd_r(), ubatch.n_seqs);
    history = ggml_reshape_3d(ctx0, history, hparams.n_shortconv_l_cache - 1, hparams.n_sconv_channels, ubatch.n_seqs);
    auto * residual = cur;
    ggml_tensor * inputs[3] = {cur, cur, cur};
    if (l.attn_res_query) {
        auto * mixed = mix(blocks, cur, l.attn_res_query);
        for (int j = 0; j < 3; ++j) {
            inputs[j] = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_view_3d(ctx0, mixed,
                            1, n_embd, n_tokens, mixed->nb[1], mixed->nb[2], (swa ? 0 : j)*mixed->nb[0])), n_embd, n_tokens);
        }
    }
    if (messages) {
        // the message of controller ctrl, [n_embd, n_tokens]
        auto message = [&](int ctrl) {
            return ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_view_3d(ctx0, messages, 1, n_embd, n_tokens,
                            messages->nb[1], messages->nb[2], ctrl*messages->nb[0])), n_embd, n_tokens);
        };
        // the messages make the Q, K and V inputs of a SWA layer differ
        for (int j = 0; j < 3; ++j) {
            inputs[j] = control(inputs[j], message(LOOP_CTRL_Q + j), *loop, LOOP_CTRL_Q + j);
        }
    }
    if (swa && messages) {
        for (int j = 0; j < 3; ++j) {
            inputs[j] = build_norm(inputs[j], l.attn_norm, nullptr, LLM_NORM_RMS, il);
        }
    } else if (swa) {
        inputs[0] = build_norm(inputs[0], l.attn_norm, nullptr, LLM_NORM_RMS, il);
        inputs[1] = inputs[2] = inputs[0];
    } else {
        inputs[0] = build_norm(inputs[0], l.attn_q_inp_norm, nullptr, LLM_NORM_RMS, il);
        inputs[1] = build_norm(inputs[1], l.attn_k_inp_norm, nullptr, LLM_NORM_RMS, il);
        inputs[2] = build_norm(inputs[2], l.attn_v_inp_norm, nullptr, LLM_NORM_RMS, il);
    }
    auto * q = build_lora_mm(l.wq, inputs[0]);
    auto * k = convolve(build_lora_mm(l.wk, inputs[1]), l.attn_k_conv,
            swa ? nullptr : build_lora_mm(l.attn_k_conv_proj, inputs[1]), state, history, recr, 0);
    auto * v = convolve(build_lora_mm(l.wv, inputs[2]), l.attn_v_conv,
            swa ? nullptr : build_lora_mm(l.attn_v_conv_proj, inputs[2]), state, history, recr, kv);
    auto * r = build_lora_mm(l.attn_rel, inputs[0]);
    q = clip(ggml_reshape_3d(ctx0, q, hd, nh, n_tokens), l.attn_q_clip, hparams.f_qk_clip);
    k = clip(ggml_reshape_3d(ctx0, k, hd, nk, n_tokens), nullptr, hparams.f_qk_clip);
    v = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, v, hd, nk, n_tokens), hparams.f_norm_rms_eps);
    r = clip(ggml_reshape_3d(ctx0, r, hparams.n_relative, nh, n_tokens), l.attn_rel_clip, hparams.f_relative_clip);
    if (!swa) {
        q = ggml_mul(ctx0, q, tau);
        r = ggml_mul(ctx0, r, tau);
    }
    auto * mask = swa ? attn->self_attn_mask_swa.raw : attn->self_attn_mask.raw;
    const int64_t streams = mask->ne[3];
    const int64_t n_kv = distances[swa]->ne[0];
    auto * sinks = build_lora_mm(l.attn_sink_proj, inputs[0]);
    // project r on the bias table first, then gather by distance: only the gathered bias is n_kv x n_tokens x nh
    auto * table = ggml_mul_mat(ctx0, ggml_pad(ctx0, l.attn_rel_bias, 0, 1, 0, 0), r);
    ggml_tensor * fa_mask = nullptr;
    if (cparams.flash_attn) {
        // exp(s - t)/(1 + sum(exp(s - t))) = exp(s)/(exp(t) + sum(exp(s))).
        table = ggml_sub(ctx0, table, ggml_reshape_3d(ctx0, sinks, 1, nh, n_tokens));
    } else {
        // build_attn scales the bias along with QK; FA adds the mask after scaling QK.
        table = ggml_scale(ctx0, table, sqrtf(hd));
    }
    table = ggml_cont(ctx0, ggml_permute(ctx0, table, 1, 0, 2, 3));
    auto * bias = ggml_get_rows(ctx0, table, distances[swa]);
    bias = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_4d(ctx0, bias, nh, n_kv, n_tokens/streams, streams), 2, 0, 1, 3));
    if (cparams.flash_attn) {
        fa_mask = ggml_cast(ctx0, ggml_add(ctx0, bias, ggml_cast(ctx0, mask, GGML_TYPE_F32)), GGML_TYPE_F16);
        sinks = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, nh), 0.0f);
        bias = nullptr;
    } else {
        sinks = ggml_reshape_4d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0,
                ggml_reshape_3d(ctx0, sinks, nh, n_tokens/streams, streams), 1, 0, 2, 3)), 1, n_tokens/streams, nh, streams);
    }
    cur = build_attn(attn, nullptr, nullptr, nullptr, q, k, v, bias, sinks, nullptr, 1.0f/sqrtf(hd), il, fa_mask);
    cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, build_lora_mm(l.wqkv_gate, inputs[0])));
    cur = convolve(build_lora_mm(l.wo, cur), l.attn_post_conv, nullptr, state, history, recr, 2*kv);
    if (ip % hparams.attn_res_block_size == 0) {
        blocks.push_back(residual);
    } else {
        cur = ggml_add(ctx0, residual, cur);
    }
    residual = cur;
    cur = ggml_reshape_2d(ctx0, mix(blocks, cur, l.ffn_res_query), n_embd, n_tokens);
    if (messages) {
        auto message = [&](int ctrl) {
            return ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_view_3d(ctx0, messages, 1, n_embd, n_tokens,
                            messages->nb[1], messages->nb[2], ctrl*messages->nb[0])), n_embd, n_tokens);
        };
        cur = control(cur, message(LOOP_CTRL_FFN), *loop, LOOP_CTRL_FFN);
    }
    cur = build_norm(cur, l.ffn_norm, nullptr, LLM_NORM_RMS, il);
    cur = build_moe_ffn(cur, l.ffn_gate_inp, nullptr, nullptr, l.ffn_down_exps, l.ffn_exp_probs_b,
            hparams.n_expert, hparams.n_expert_used(il), LLM_FFN_SITU, true, hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID, il, nullptr, l.ffn_gate_up_exps,
            l.ffn_up_exps_s, l.ffn_gate_exps_s, l.ffn_down_exps_s);
    cur = convolve(cur, l.ffn_post_conv, nullptr, state, history, recr, 2*kv + n_embd);
    cur = ggml_add(ctx0, residual, cur);
    cb(cur, "l_out", il);
    return cur;
}

void llama_model_inkling2_loop::graph::build_output(ggml_tensor * cur, const std::vector<ggml_tensor *> & blocks) {
    cur = ggml_reshape_2d(ctx0, mix(blocks, cur, model.output_res_query), n_embd, n_tokens);
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    // DSpark target features: the last one is the final norm output of every token
    res->t_layer_inp[n_layer] = cur;

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }
    res->t_embd = cur;
    cur = build_lora_mm(model.output, cur);
    ggml_prec_set_acc(cur, GGML_PREC_F32);
    const float cap = hparams.f_final_logit_softcapping;
    cur = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/cap)), cap);
    cur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, hparams.n_vocab_unpadded, cur->ne[1], cur->nb[1], 0));
    res->t_logits = cur;
    cb(cur, "result_output", -1);
}

llama_model_inkling2_loop::graph::graph(const llm_graph_params & params) : llm_graph_context(params) {
    const auto & m = static_cast<const llama_model_inkling2_loop &>(model);
    const int start = m.core_start;
    const int end   = m.core_end;

    auto * cur = build_inputs();
    std::vector<ggml_tensor *> blocks;

    // DSpark target features are indexed by layer visit, not by cache slot
    int visit = 0;

    for (int il = 0; il < start; ++il) {
        res->t_layer_inp[visit++] = cur;
        cur = build_layer(il, il, cur, blocks);
    }

    // every pass starts from the prelude output
    auto * prelude = cur;
    const auto prelude_blocks = blocks;
    for (uint32_t pass = 0; pass < m.n_passes; ++pass) {
        // the bank of the previous pass: its blocks after the first core block, and its output
        std::vector<ggml_tensor *> bank;
        ggml_tensor * bank_out = nullptr;
        if (pass > 0) {
            bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
            bank_out = cur;
            cur = prelude;
            blocks = prelude_blocks;
        }
        for (int ip = start; ip < end; ++ip) {
            const int il = pass == 0 ? ip : m.n_phys + (pass - 1)*(end - start) + (ip - start);
            res->t_layer_inp[visit++] = cur;
            if (pass == 0) {
                cur = build_layer(il, ip, cur, blocks);
                continue;
            }
            const auto & loop = m.loop_layers[ip - start];
            cur = build_layer(il, ip, cur, blocks, mix(bank, bank_out, loop.msg_query), &loop);
        }
    }

    for (int il = end; il < m.n_phys; ++il) {
        res->t_layer_inp[visit++] = cur;
        cur = build_layer(il, il, cur, blocks);
    }
    GGML_ASSERT(visit == n_layer);

    build_output(cur, blocks);
}
