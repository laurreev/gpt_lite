#include "llama.h"
#include <jni.h>
#include <string>
#include <map>
#include <vector>
#include <android/log.h>

#define LOG_TAG "LlamaCpp"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Global storage for models, contexts, and samplers
static std::map<int64_t, llama_model*> models;
static std::map<int64_t, llama_context*> contexts;
static std::map<int64_t, llama_sampler*> samplers;
static int64_t next_id = 1;
static bool backend_initialized = false;

extern "C" {

// Initialize the backend (call once)
JNIEXPORT void JNICALL
Java_com_example_gpt_1lite_LlamaCppPlugin_initBackend(JNIEnv *env, jobject /* this */) {
    if (!backend_initialized) {
        LOGI("Initializing llama.cpp backend");
        llama_backend_init();
        backend_initialized = true;
    }
}

// JNI exports for Flutter
JNIEXPORT jlong JNICALL
Java_com_example_gpt_1lite_LlamaCppPlugin_loadModel(JNIEnv *env, jobject /* this */, jstring model_path) {
    const char *path = env->GetStringUTFChars(model_path, 0);
    LOGI("Loading model from: %s", path);
    
    auto params = llama_model_default_params();
    params.n_gpu_layers = 0; // CPU only for now
    params.use_mmap = true;
    params.use_mlock = false;
    
    llama_model* model = llama_load_model_from_file(path, params);
    env->ReleaseStringUTFChars(model_path, path);
    
    if (!model) {
        LOGE("Failed to load model from: %s", path);
        return 0; // Failed to load
    }
    
    int64_t model_id = next_id++;
    models[model_id] = model;
    LOGI("Model loaded successfully with ID: %lld", model_id);
    return model_id;
}

JNIEXPORT jlong JNICALL
Java_com_example_gpt_1lite_LlamaCppPlugin_createContext(JNIEnv *env, jobject /* this */, jlong model_id) {
    if (models.find(model_id) == models.end()) {
        LOGE("Model ID %lld not found", model_id);
        return 0;
    }
      auto params = llama_context_default_params();
    params.n_ctx = 2048;
    params.n_batch = 512;
    // Note: seed is not a member of llama_context_params in current API
    
    llama_context* context = llama_new_context_with_model(models[model_id], params);
    if (!context) {
        LOGE("Failed to create context for model ID: %lld", model_id);
        return 0;
    }
    
    // Create a sampler for this context
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler* sampler = llama_sampler_chain_init(sparams);
    
    // Add sampling components    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.9f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(-1)); // Random seed
    
    int64_t context_id = next_id++;
    contexts[context_id] = context;
    samplers[context_id] = sampler;
    LOGI("Context created successfully with ID: %lld", context_id);
    return context_id;
}

JNIEXPORT jstring JNICALL
Java_com_example_gpt_1lite_LlamaCppPlugin_generateText(JNIEnv *env, jobject /* this */, 
                                                       jlong context_id, jstring input_text, jint max_tokens) {
    if (contexts.find(context_id) == contexts.end() || samplers.find(context_id) == samplers.end()) {
        LOGE("Context ID %lld not found", context_id);
        return env->NewStringUTF("");
    }
    
    const char *input = env->GetStringUTFChars(input_text, 0);
    llama_context *ctx = contexts[context_id];
    llama_sampler *sampler = samplers[context_id];
    
    LOGI("Generating text for input: %.50s...", input);
      // Tokenize input
    std::vector<llama_token> tokens(512);
    const int n_tokens = llama_tokenize(
        llama_model_get_vocab(llama_get_model(ctx)), 
        input, 
        strlen(input),
        tokens.data(), 
        tokens.size(), 
        true,   // add_special (BOS)
        false   // parse_special
    );
    
    env->ReleaseStringUTFChars(input_text, input);
    
    if (n_tokens < 0) {
        LOGE("Tokenization failed with error: %d", n_tokens);
        return env->NewStringUTF("");
    }
    
    tokens.resize(n_tokens);
    LOGI("Tokenized input into %d tokens", n_tokens);
    
    // Create batch for input tokens
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    
    for (int i = 0; i < n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = false; // Only need logits for last token
    }
    batch.logits[n_tokens - 1] = true; // We need logits for the last token
    batch.n_tokens = n_tokens;
    
    // Decode input tokens
    if (llama_decode(ctx, batch) != 0) {
        LOGE("Failed to decode input tokens");
        llama_batch_free(batch);
        return env->NewStringUTF("");
    }
    
    // Generate response tokens
    std::string result;
    int n_generated = 0;
    
    for (int i = 0; i < max_tokens; ++i) {
        // Sample next token
        const llama_token id = llama_sampler_sample(sampler, ctx, -1);
        
        // Accept the token (update sampler state)
        llama_sampler_accept(sampler, id);
          // Convert token to string
        char token_str[256];
        const int token_len = llama_token_to_piece(
            llama_model_get_vocab(llama_get_model(ctx)),
            id,
            token_str,
            sizeof(token_str),
            0,
            false
        );
        
        if (token_len < 0) {
            LOGE("Failed to convert token to string");
            break;
        }
        
        token_str[token_len] = '\0';
        result += std::string(token_str);
        n_generated++;
          // Check for end of sequence
        if (id == llama_vocab_eos(llama_model_get_vocab(llama_get_model(ctx)))) {
            LOGI("Generated EOS token, stopping");
            break;
        }
        
        // Prepare next batch with the new token
        batch.n_tokens = 1;
        batch.token[0] = id;
        batch.pos[0] = n_tokens + i;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;
        
        // Decode the new token
        if (llama_decode(ctx, batch) != 0) {
            LOGE("Failed to decode generated token");
            break;
        }
    }
    
    llama_batch_free(batch);
    LOGI("Generated %d tokens, result length: %zu", n_generated, result.length());
    
    return env->NewStringUTF(result.c_str());
}

JNIEXPORT void JNICALL
Java_com_example_gpt_1lite_LlamaCppPlugin_freeContext(JNIEnv *env, jobject /* this */, jlong context_id) {
    auto ctx_it = contexts.find(context_id);
    if (ctx_it != contexts.end()) {
        llama_free(ctx_it->second);
        contexts.erase(ctx_it);
    }
    
    auto smp_it = samplers.find(context_id);
    if (smp_it != samplers.end()) {
        llama_sampler_free(smp_it->second);
        samplers.erase(smp_it);
    }
    
    LOGI("Freed context and sampler with ID: %lld", context_id);
}

JNIEXPORT void JNICALL
Java_com_example_gpt_1lite_LlamaCppPlugin_freeModel(JNIEnv *env, jobject /* this */, jlong model_id) {
    auto it = models.find(model_id);
    if (it != models.end()) {
        llama_free_model(it->second);
        models.erase(it);
    }
}

} // extern "C"
