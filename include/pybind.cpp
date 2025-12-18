#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> llama_decoder_layer_sm90(
    torch::Tensor input,
    torch::Tensor weight_qkv,
    torch::Tensor weight_o,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor rms_input_weight,
    torch::Tensor cos,
    torch::Tensor sin
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> llama_decoder_layer_sglang_sm90(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight_qkv,
    torch::Tensor weight_o,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor rms_input_weight,
    float eps,
    torch::Tensor cos,
    torch::Tensor sin
);

void llama_decoder_layer_batch_sglang_sm90(
    torch::Tensor output,
    torch::Tensor residual_output,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight_qkv,
    torch::Tensor weight_o,
    torch::Tensor paged_kv_indptr,
    torch::Tensor paged_kv_indices,
    torch::Tensor k_cache_ptrs,
    torch::Tensor v_cache_ptrs,
    int layer_id,
    torch::Tensor rms_input_weight,
    float eps,
    torch::Tensor positions,
    torch::Tensor cos_sin
);

torch::Tensor deepseek_decoder_layer(
    torch::Tensor input,
    torch::Tensor weight_q_nope,
    torch::Tensor weight_q_pe,
    torch::Tensor weight_uk,
    torch::Tensor weight_kv_nope,
    torch::Tensor weight_k_pe,
    torch::Tensor weight_uv,
    torch::Tensor weight_o,
    torch::Tensor ckv_cache,
    torch::Tensor rms_input_weight,
    torch::Tensor rms_ckv_weight,
    torch::Tensor cos,
    torch::Tensor sin
);

torch::Tensor rmsnorm(
    torch::Tensor input,
    torch::Tensor weight
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> llama_decoder_layer_sm120(
    torch::Tensor input,
    torch::Tensor weight_qkv,
    torch::Tensor weight_o,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor rms_input_weight,
    torch::Tensor cos,
    torch::Tensor sin
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> llama_decoder_layer_sglang_sm120(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight_qkv,
    torch::Tensor weight_o,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor rms_input_weight,
    float eps,
    torch::Tensor cos,
    torch::Tensor sin
);

void llama_decoder_layer_batch_sglang_sm120(
    torch::Tensor output,
    torch::Tensor residual_output,
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight_qkv,
    torch::Tensor weight_o,
    torch::Tensor paged_kv_indptr,
    torch::Tensor paged_kv_indices,
    torch::Tensor k_cache_ptrs,
    torch::Tensor v_cache_ptrs,
    int layer_id,
    torch::Tensor rms_input_weight,
    float eps,
    torch::Tensor positions,
    torch::Tensor cos_sin
);

// Pythia-2.8B decoder layer
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> pythia_2b8_decoder_layer_sm120(
    torch::Tensor input,
    torch::Tensor weight_qkv,
    torch::Tensor bias_qkv,           // QKV projection bias
    torch::Tensor weight_o,
    torch::Tensor bias_o,             // Output projection bias
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor layernorm_weight,
    torch::Tensor layernorm_bias,     // LayerNorm bias
    torch::Tensor cos,
    torch::Tensor sin,
    // MLP weights
    torch::Tensor post_ln_weight,
    torch::Tensor post_ln_bias,
    torch::Tensor mlp_up_weight,
    torch::Tensor mlp_up_bias,
    torch::Tensor mlp_down_weight,
    torch::Tensor mlp_down_bias,
    int64_t current_seq_len           // Current sequence length
);

// Pythia-6.9B decoder layer
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> pythia_6b9_decoder_layer_sm120(
    torch::Tensor input,
    torch::Tensor weight_qkv,
    torch::Tensor bias_qkv,           // QKV projection bias
    torch::Tensor weight_o,
    torch::Tensor bias_o,             // Output projection bias
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor layernorm_weight,
    torch::Tensor layernorm_bias,     // LayerNorm bias
    torch::Tensor cos,
    torch::Tensor sin,
    // MLP weights
    torch::Tensor post_ln_weight,
    torch::Tensor post_ln_bias,
    torch::Tensor mlp_up_weight,
    torch::Tensor mlp_up_bias,
    torch::Tensor mlp_down_weight,
    torch::Tensor mlp_down_bias,
    int64_t current_seq_len           // Current sequence length
);

// CUDA Graph support functions for Pythia-2.8B
void pythia_2b8_create_graph_context_sm120(
    int64_t context_id,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor weight_qkv,
    torch::Tensor weight_o,
    torch::Tensor mlp_up_weight,
    torch::Tensor mlp_down_weight,
    int64_t max_seq_len
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> pythia_2b8_graph_decode_step_sm120(
    int64_t context_id,
    torch::Tensor input,
    torch::Tensor layernorm_weight,
    torch::Tensor layernorm_bias,
    torch::Tensor qkv_bias,
    torch::Tensor o_bias,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor post_ln_weight,
    torch::Tensor post_ln_bias,
    torch::Tensor mlp_up_bias,
    torch::Tensor mlp_down_bias,
    int64_t current_seq_len
);

void pythia_2b8_destroy_graph_context_sm120(int64_t context_id);

// CUDA Graph support functions for Pythia-6.9B
void pythia_6b9_create_graph_context_sm120(
    int64_t context_id,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor weight_qkv,
    torch::Tensor weight_o,
    torch::Tensor mlp_up_weight,
    torch::Tensor mlp_down_weight,
    int64_t max_seq_len
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> pythia_6b9_graph_decode_step_sm120(
    int64_t context_id,
    torch::Tensor input,
    torch::Tensor layernorm_weight,
    torch::Tensor layernorm_bias,
    torch::Tensor qkv_bias,
    torch::Tensor o_bias,
    torch::Tensor cos,
    torch::Tensor sin,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor post_ln_weight,
    torch::Tensor post_ln_bias,
    torch::Tensor mlp_up_bias,
    torch::Tensor mlp_down_bias,
    int64_t current_seq_len
);

void pythia_6b9_destroy_graph_context_sm120(int64_t context_id);

#ifdef COMPILE_SM90
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("llama_decoder_layer", &llama_decoder_layer_sm90, "");
    m.def("llama_decoder_layer_sglang", &llama_decoder_layer_sglang_sm90, "");
    m.def("llama_decoder_layer_batch_decode_sglang", &llama_decoder_layer_batch_sglang_sm90, "");
    m.def("deepseek_decoder_layer", &deepseek_decoder_layer, "");
    m.def("rmsnorm", &rmsnorm, "");
}
#endif
#ifdef COMPILE_SM120
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("llama_decoder_layer", &llama_decoder_layer_sm120, "");
    m.def("llama_decoder_layer_sglang", &llama_decoder_layer_sglang_sm120, "");
    m.def("llama_decoder_layer_batch_decode_sglang", &llama_decoder_layer_batch_sglang_sm120, "");
    
    // Pythia-2.8B
    m.def("pythia_2b8_decoder_layer", &pythia_2b8_decoder_layer_sm120, "Pythia-2.8B decoder layer");
    m.def("pythia_2b8_create_graph_context", &pythia_2b8_create_graph_context_sm120, "Create graph context for Pythia-2.8B");
    m.def("pythia_2b8_graph_decode_step", &pythia_2b8_graph_decode_step_sm120, "Graph decode step for Pythia-2.8B");
    m.def("pythia_2b8_destroy_graph_context", &pythia_2b8_destroy_graph_context_sm120, "Destroy graph context for Pythia-2.8B");
    
    // Pythia-6.9B
    m.def("pythia_6b9_decoder_layer", &pythia_6b9_decoder_layer_sm120, "Pythia-6.9B decoder layer");
    m.def("pythia_6b9_create_graph_context", &pythia_6b9_create_graph_context_sm120, "Create graph context for Pythia-6.9B");
    m.def("pythia_6b9_graph_decode_step", &pythia_6b9_graph_decode_step_sm120, "Graph decode step for Pythia-6.9B");
    m.def("pythia_6b9_destroy_graph_context", &pythia_6b9_destroy_graph_context_sm120, "Destroy graph context for Pythia-6.9B");
}
#endif