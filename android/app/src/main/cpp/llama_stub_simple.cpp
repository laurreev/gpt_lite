#include <jni.h>
#include <android/log.h>

#define LOG_TAG "LlamaCppFlutter"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" {

// Simple JNI exports that do nothing for now - just to make the build pass
JNIEXPORT jlong JNICALL
Java_com_example_gpt_1lite_LlamaCppPlugin_loadModel(JNIEnv *env, jobject /* this */, jstring model_path) {
    LOGE("loadModel called - stub implementation");
    return 1; // Return a dummy model ID
}

JNIEXPORT jlong JNICALL
Java_com_example_gpt_1lite_LlamaCppPlugin_createContext(JNIEnv *env, jobject /* this */, jlong model_id) {
    LOGE("createContext called - stub implementation");
    return 1; // Return a dummy context ID
}

JNIEXPORT jstring JNICALL
Java_com_example_gpt_1lite_LlamaCppPlugin_generateText(JNIEnv *env, jobject /* this */, 
                                                       jlong context_id, jstring input_text, jint max_tokens) {
    LOGE("generateText called - stub implementation");
    return env->NewStringUTF("This is a stub response from the native layer.");
}

JNIEXPORT void JNICALL
Java_com_example_gpt_1lite_LlamaCppPlugin_freeContext(JNIEnv *env, jobject /* this */, jlong context_id) {
    LOGE("freeContext called - stub implementation");
}

JNIEXPORT void JNICALL
Java_com_example_gpt_1lite_LlamaCppPlugin_freeModel(JNIEnv *env, jobject /* this */, jlong model_id) {
    LOGE("freeModel called - stub implementation");
}

} // extern "C"
