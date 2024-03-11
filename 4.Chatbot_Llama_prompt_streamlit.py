import streamlit as st
from langchain_community.llms import LlamaCpp

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load the LlamaCpp language model, adjust GPU usage based on your hardware
llm = LlamaCpp(
    model_path = "e:/models/llama/llama-2-7b-chat.Q6_K.gguf",
    n_gpu_layers=40,
    n_batch=512,  # Batch size for model processing
    # verbose=False,  # Enable detailed logging for debugging
)

# Define the prompt template with a placeholder for the question
template = """
Question: {question}

Answer:
"""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Create an LLMChain to manage interactions with the prompt and model
llm_chain = LLMChain(prompt=prompt, llm=llm)

st.title("Chatbot Interface")

question = st.text_input("You:", "")

if st.button("Ask"):
    if question:
        answer = llm_chain.run(question)
        st.text_area("Bot:", value=answer, height=200)
    else:
        st.warning("Please enter a question.")
        
#Output:
# 
# PS A:\Chat-AI> & D:/Python312/python.exe a:/Chat-AI/1.1_Chatbot_Llama_st.py
# ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
# ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
# ggml_init_cublas: found 1 CUDA devices:
#   Device 0: NVIDIA GeForce GTX 1070, compute capability 6.1, VMM: yes
# llama_model_loader: loaded meta data with 19 key-value pairs and 291 tensors from e:/models/llama/llama-2-7b.Q2_K.gguf (version GGUF V2)
# llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
# llama_model_loader: - kv   0:                       general.architecture str              = llama
# llama_model_loader: - kv   1:                               general.name str              = LLaMA v2
# llama_model_loader: - kv   2:                       llama.context_length u32              = 4096
# llama_model_loader: - kv   3:                     llama.embedding_length u32              = 4096
# llama_model_loader: - kv   4:                          llama.block_count u32              = 32
# llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 11008
# llama_model_loader: - kv   6:                 llama.rope.dimension_count u32              = 128
# llama_model_loader: - kv   7:                 llama.attention.head_count u32              = 32
# llama_model_loader: - kv   8:              llama.attention.head_count_kv u32              = 32
# llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
# llama_model_loader: - kv  10:                          general.file_type u32              = 10
# llama_model_loader: - kv  11:                       tokenizer.ggml.model str              = llama
# llama_model_loader: - kv  12:                      tokenizer.ggml.tokens arr[str,32000]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
# llama_model_loader: - kv  13:                      tokenizer.ggml.scores arr[f32,32000]   = [0.000000, 0.000000, 0.000000, 0.0000...
# llama_model_loader: - kv  14:                  tokenizer.ggml.token_type arr[i32,32000]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
# llama_model_loader: - kv  15:                tokenizer.ggml.bos_token_id u32              = 1
# llama_model_loader: - kv  16:                tokenizer.ggml.eos_token_id u32              = 2
# llama_model_loader: - kv  17:            tokenizer.ggml.unknown_token_id u32              = 0
# llama_model_loader: - kv  18:               general.quantization_version u32              = 2
# llama_model_loader: - type  f32:   65 tensors
# llama_model_loader: - type q2_K:   65 tensors
# llama_model_loader: - type q3_K:  160 tensors
# llama_model_loader: - type q6_K:    1 tensors
# llm_load_vocab: special tokens definition check successful ( 259/32000 ).
# llm_load_print_meta: format           = GGUF V2
# llm_load_print_meta: arch             = llama
# llm_load_print_meta: vocab type       = SPM
# llm_load_print_meta: n_vocab          = 32000
# llm_load_print_meta: n_merges         = 0
# llm_load_print_meta: n_ctx_train      = 4096
# llm_load_print_meta: n_embd           = 4096
# llm_load_print_meta: n_head           = 32
# llm_load_print_meta: n_head_kv        = 32
# llm_load_print_meta: n_layer          = 32
# llm_load_print_meta: n_rot            = 128
# llm_load_print_meta: n_embd_head_k    = 128
# llm_load_print_meta: n_embd_head_v    = 128
# llm_load_print_meta: n_gqa            = 1
# llm_load_print_meta: n_embd_k_gqa     = 4096
# llm_load_print_meta: n_embd_v_gqa     = 4096
# llm_load_print_meta: f_norm_eps       = 0.0e+00
# llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
# llm_load_print_meta: f_clamp_kqv      = 0.0e+00
# llm_load_print_meta: f_max_alibi_bias = 0.0e+00
# llm_load_print_meta: n_ff             = 11008
# llm_load_print_meta: n_expert         = 0
# llm_load_print_meta: n_expert_used    = 0
# llm_load_print_meta: rope scaling     = linear
# llm_load_print_meta: freq_base_train  = 10000.0
# llm_load_print_meta: freq_scale_train = 1
# llm_load_print_meta: n_yarn_orig_ctx  = 4096
# llm_load_print_meta: rope_finetuned   = unknown
# llm_load_print_meta: model type       = 7B
# llm_load_print_meta: model ftype      = Q2_K - Medium
# llm_load_print_meta: model params     = 6.74 B
# llm_load_print_meta: model size       = 2.63 GiB (3.35 BPW)
# llm_load_print_meta: general.name     = LLaMA v2
# llm_load_print_meta: BOS token        = 1 '<s>'
# llm_load_print_meta: EOS token        = 2 '</s>'
# llm_load_print_meta: UNK token        = 0 '<unk>'
# llm_load_print_meta: LF token         = 13 '<0x0A>'
# llm_load_tensors: ggml ctx size =    0.22 MiB