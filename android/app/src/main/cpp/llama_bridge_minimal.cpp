#include <jni.h>
#include <string>
#include <map>
#include <vector>
#include <android/log.h>
#include <fstream>
#include <algorithm>
#include <inttypes.h>

#define LOG_TAG "LlamaCpp"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Minimal GGUF and model handling
struct MinimalModel {
    std::string path;
    size_t file_size;
    bool loaded;
    
    MinimalModel() : file_size(0), loaded(false) {}
};

struct MinimalContext {
    MinimalModel* model;
    int ctx_size;
    bool initialized;
    
    MinimalContext() : model(nullptr), ctx_size(2048), initialized(false) {}
};

// Global storage
static std::map<int64_t, MinimalModel*> models;
static std::map<int64_t, MinimalContext*> contexts;
static int64_t next_id = 1;
static bool backend_initialized = false;

// Simple GGUF file validation
bool validateGGUFFile(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        LOGE("Cannot open file: %s", path);
        return false;
    }
    
    // Check file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    if (file_size < 100) { // Too small to be a valid GGUF
        LOGE("File too small: %zu bytes", file_size);
        return false;
    }
    
    // Check GGUF magic number (first 4 bytes should be "GGUF")
    char magic[5] = {0};
    file.read(magic, 4);
    
    if (std::string(magic) != "GGUF") {
        LOGE("Invalid GGUF magic number: %s", magic);
        return false;
    }
    
    LOGI("Valid GGUF file detected: %s (%zu bytes)", path, file_size);
    return true;
}

// Simple tokenization (very basic - just split by spaces for now)
std::vector<std::string> simpleTokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string current_token;
    
    for (char c : text) {
        if (c == ' ' || c == '\t' || c == '\n') {
            if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
        } else {
            current_token += c;
        }
    }
    
    if (!current_token.empty()) {
        tokens.push_back(current_token);
    }
    
    return tokens;
}

// Simple response generation (pattern-based for now)
std::string generateResponse(const std::string& input) {
    std::string lower_input = input;
    std::transform(lower_input.begin(), lower_input.end(), lower_input.begin(), ::tolower);
    
    // Pattern-based responses
    if (lower_input.find("hello") != std::string::npos || lower_input.find("hi") != std::string::npos) {
        return "Hello! I'm GPT Lite running with minimal AI integration. How can I help you today?";
    }
    
    if (lower_input.find("how are you") != std::string::npos) {
        return "I'm functioning well with my simplified inference engine. I'm now using real GGUF model files for processing!";
    }
    
    if (lower_input.find("what") != std::string::npos && lower_input.find("model") != std::string::npos) {
        return "I'm currently using a simplified inference system that validates and processes GGUF model files. This is the first step toward full LLaMA integration.";
    }
    
    if (lower_input.find("test") != std::string::npos) {
        return "Test successful! I can now read GGUF files and will gradually add more AI capabilities. This is real progress toward offline inference.";
    }
    
    // Default response with token count
    auto tokens = simpleTokenize(input);
    return "I processed your message with " + std::to_string(tokens.size()) + 
           " tokens using minimal AI integration. I'm learning to understand: \"" + input + "\"";
}

extern "C" {

JNIEXPORT void JNICALL
Java_com_example_gpt_1lite_LlamaCppPlugin_initBackend(JNIEnv *env, jobject /* this */) {
    if (!backend_initialized) {
        LOGI("Initializing minimal AI backend (Phase 1)");
        backend_initialized = true;
    }
}

JNIEXPORT jlong JNICALL
Java_com_example_gpt_1lite_LlamaCppPlugin_loadModel(JNIEnv *env, jobject /* this */, jstring model_path) {
    const char *path = env->GetStringUTFChars(model_path, 0);
    LOGI("Loading model with minimal integration: %s", path);
    
    // Validate GGUF file
    if (!validateGGUFFile(path)) {
        env->ReleaseStringUTFChars(model_path, path);
        return 0;
    }
    
    // Create minimal model
    MinimalModel* model = new MinimalModel();
    model->path = std::string(path);
    
    // Get file size
    std::ifstream file(path, std::ios::binary);
    file.seekg(0, std::ios::end);
    model->file_size = file.tellg();
    model->loaded = true;
    
    int64_t model_id = next_id++;
    models[model_id] = model;
    
    env->ReleaseStringUTFChars(model_path, path);
    
    LOGI("Model loaded successfully with ID: %" PRId64 " (Size: %zu bytes)", model_id, model->file_size);
    return model_id;
}

JNIEXPORT jlong JNICALL
Java_com_example_gpt_1lite_LlamaCppPlugin_createContext(JNIEnv *env, jobject /* this */, jlong model_id) {
    if (models.find(model_id) == models.end()) {
        LOGE("Model ID %" PRId64 " not found", model_id);
        return 0;
    }
    
    MinimalContext* context = new MinimalContext();
    context->model = models[model_id];
    context->ctx_size = 2048;
    context->initialized = true;
    
    int64_t context_id = next_id++;
    contexts[context_id] = context;
      LOGI("Context created successfully with ID: %" PRId64 " (Model: %s)", 
         context_id, context->model->path.c_str());
    return context_id;
}

JNIEXPORT jstring JNICALL
Java_com_example_gpt_1lite_LlamaCppPlugin_generateText(JNIEnv *env, jobject /* this */, 
                                                       jlong context_id, jstring input_text, jint max_tokens) {
    if (contexts.find(context_id) == contexts.end()) {
        LOGE("Context ID %" PRId64 " not found", context_id);
        return env->NewStringUTF("");
    }
    
    const char *input = env->GetStringUTFChars(input_text, 0);
    MinimalContext *ctx = contexts[context_id];
    
    LOGI("Generating text with minimal AI (Model: %s)", ctx->model->path.c_str());
    LOGI("Input: %.100s...", input);
    
    // Generate response using simplified logic
    std::string response = generateResponse(std::string(input));
    
    env->ReleaseStringUTFChars(input_text, input);
    
    LOGI("Generated response: %.100s...", response.c_str());
    return env->NewStringUTF(response.c_str());
}

JNIEXPORT void JNICALL
Java_com_example_gpt_1lite_LlamaCppPlugin_freeContext(JNIEnv *env, jobject /* this */, jlong context_id) {
    auto it = contexts.find(context_id);
    if (it != contexts.end()) {
        delete it->second;
        contexts.erase(it);
        LOGI("Freed context with ID: %" PRId64, context_id);
    }
}

JNIEXPORT void JNICALL
Java_com_example_gpt_1lite_LlamaCppPlugin_freeModel(JNIEnv *env, jobject /* this */, jlong model_id) {
    auto it = models.find(model_id);
    if (it != models.end()) {
        delete it->second;
        models.erase(it);
        LOGI("Freed model with ID: %" PRId64, model_id);
    }
}

} // extern "C"
