#include <jni.h>
#include <string>
#include <map>
#include <vector>
#include <android/log.h>
#include <unistd.h>

#define LOG_TAG "LlamaCpp"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Stub implementations for testing
static std::map<int64_t, std::string> loaded_models;
static std::map<int64_t, int64_t> contexts; // context_id -> model_id
static int64_t next_id = 1;
static bool backend_initialized = false;

extern "C" {

// Initialize the backend (call once)
JNIEXPORT void JNICALL
Java_com_example_gpt_1lite_LlamaCppPlugin_initBackend(JNIEnv *env, jobject /* this */) {
    if (!backend_initialized) {
        LOGI("Initializing llama.cpp backend (STUB MODE)");
        backend_initialized = true;
    }
}

// JNI exports for Flutter
JNIEXPORT jlong JNICALL
Java_com_example_gpt_1lite_LlamaCppPlugin_loadModel(JNIEnv *env, jobject /* this */, jstring model_path) {
    const char *path = env->GetStringUTFChars(model_path, 0);
    LOGI("Loading model from: %s (STUB MODE)", path);
    
    // Simulate loading delay
    usleep(1000000); // 1 second
    
    int64_t model_id = next_id++;
    loaded_models[model_id] = std::string(path);
    
    env->ReleaseStringUTFChars(model_path, path);
    
    LOGI("Model loaded successfully with ID: %ld (STUB)", model_id);
    return model_id;
}

JNIEXPORT jlong JNICALL
Java_com_example_gpt_1lite_LlamaCppPlugin_createContext(JNIEnv *env, jobject /* this */, jlong model_id) {
    if (loaded_models.find(model_id) == loaded_models.end()) {
        LOGE("Model ID %ld not found", model_id);
        return 0;
    }
    
    int64_t context_id = next_id++;
    contexts[context_id] = model_id;
    
    LOGI("Context created successfully with ID: %ld (STUB)", context_id);
    return context_id;
}

JNIEXPORT jstring JNICALL
Java_com_example_gpt_1lite_LlamaCppPlugin_generateText(JNIEnv *env, jobject /* this */, 
                                                       jlong context_id, jstring input_text, jint max_tokens) {
    if (contexts.find(context_id) == contexts.end()) {
        LOGE("Context ID %ld not found", context_id);
        return env->NewStringUTF("");
    }
    
    const char *input = env->GetStringUTFChars(input_text, 0);
    LOGI("Generating text for input: %.50s... (STUB MODE)", input);
    
    // Simulate some processing time
    usleep(500000); // 0.5 seconds
    
    // Generate a simple response based on input
    std::string response;
    std::string input_str(input);
    
    if (input_str.find("hello") != std::string::npos || input_str.find("hi") != std::string::npos) {
        response = "Hello! I'm a stub AI assistant. How can I help you today?";
    } else if (input_str.find("how are you") != std::string::npos) {
        response = "I'm doing well, thank you! I'm running in stub mode, so I can't do real AI inference yet.";
    } else if (input_str.find("what") != std::string::npos) {
        response = "That's an interesting question! In stub mode, I can only provide simple preset responses.";
    } else if (input_str.find("code") != std::string::npos || input_str.find("program") != std::string::npos) {
        response = "I'd love to help with coding! Once the real llama.cpp integration is complete, I'll be able to assist with programming tasks.";
    } else {
        response = "I understand you said: \"" + input_str + "\". I'm currently running in stub mode with limited responses.";
    }
    
    env->ReleaseStringUTFChars(input_text, input);
    
    LOGI("Generated response: %.100s... (STUB)", response.c_str());
    return env->NewStringUTF(response.c_str());
}

JNIEXPORT void JNICALL
Java_com_example_gpt_1lite_LlamaCppPlugin_freeContext(JNIEnv *env, jobject /* this */, jlong context_id) {
    auto it = contexts.find(context_id);
    if (it != contexts.end()) {
        contexts.erase(it);
        LOGI("Freed context with ID: %ld (STUB)", context_id);
    }
}

JNIEXPORT void JNICALL
Java_com_example_gpt_1lite_LlamaCppPlugin_freeModel(JNIEnv *env, jobject /* this */, jlong model_id) {
    auto it = loaded_models.find(model_id);
    if (it != loaded_models.end()) {
        loaded_models.erase(it);
        LOGI("Freed model with ID: %ld (STUB)", model_id);
    }
}

} // extern "C"
