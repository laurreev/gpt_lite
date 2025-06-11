package com.example.gpt_lite

import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.MethodChannel.MethodCallHandler
import io.flutter.plugin.common.MethodChannel.Result

class LlamaCppPlugin : FlutterPlugin, MethodCallHandler {
    private lateinit var channel: MethodChannel

    companion object {
        init {
            System.loadLibrary("llama_cpp_flutter")
        }
    }

    override fun onAttachedToEngine(flutterPluginBinding: FlutterPlugin.FlutterPluginBinding) {
        channel = MethodChannel(flutterPluginBinding.binaryMessenger, "llama_cpp_plugin")
        channel.setMethodCallHandler(this)
    }    override fun onMethodCall(call: MethodCall, result: Result) {
        when (call.method) {
            "initBackend" -> {
                initBackend()
                result.success(null)
            }
            "loadModel" -> {
                val modelPath = call.argument<String>("modelPath")
                if (modelPath != null) {
                    val modelId = loadModel(modelPath)
                    result.success(modelId)
                } else {
                    result.error("INVALID_ARGUMENT", "Model path is required", null)
                }
            }            "createContext" -> {
                val modelId = call.argument<Any>("modelId")?.let {
                    when (it) {
                        is Int -> it.toLong()
                        is Long -> it
                        else -> null
                    }
                }
                if (modelId != null) {
                    val contextId = createContext(modelId)
                    result.success(contextId)
                } else {
                    result.error("INVALID_ARGUMENT", "Model ID is required", null)
                }
            }            "generateText" -> {
                val contextId = call.argument<Any>("contextId")?.let {
                    when (it) {
                        is Int -> it.toLong()
                        is Long -> it
                        else -> null
                    }
                }
                val inputText = call.argument<String>("inputText")
                val maxTokens = call.argument<Int>("maxTokens") ?: 100
                
                if (contextId != null && inputText != null) {
                    val response = generateText(contextId, inputText, maxTokens)
                    result.success(response)
                } else {
                    result.error("INVALID_ARGUMENT", "Context ID and input text are required", null)
                }
            }            "freeContext" -> {
                val contextId = call.argument<Any>("contextId")?.let {
                    when (it) {
                        is Int -> it.toLong()
                        is Long -> it
                        else -> null
                    }
                }
                if (contextId != null) {
                    freeContext(contextId)
                    result.success(null)
                } else {
                    result.error("INVALID_ARGUMENT", "Context ID is required", null)
                }
            }
            "freeModel" -> {
                val modelId = call.argument<Any>("modelId")?.let {
                    when (it) {
                        is Int -> it.toLong()
                        is Long -> it
                        else -> null
                    }
                }
                if (modelId != null) {
                    freeModel(modelId)
                    result.success(null)
                } else {
                    result.error("INVALID_ARGUMENT", "Model ID is required", null)
                }
            }
            else -> {
                result.notImplemented()
            }
        }
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
    }    // Native method declarations
    external fun initBackend()
    external fun loadModel(modelPath: String): Long
    external fun createContext(modelId: Long): Long
    external fun generateText(contextId: Long, inputText: String, maxTokens: Int): String
    external fun freeContext(contextId: Long)
    external fun freeModel(modelId: Long)
}
