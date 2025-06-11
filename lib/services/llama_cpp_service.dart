import 'package:flutter/services.dart';

class LlamaCppService {
  static const MethodChannel _channel = MethodChannel('llama_cpp_plugin');
  
  int? _modelId;
  int? _contextId;
  bool _backendInitialized = false;
  
  bool get isModelLoaded => _modelId != null;
  bool get isContextReady => _contextId != null;
  
  Future<void> initBackend() async {
    if (_backendInitialized) return;
    
    try {
      await _channel.invokeMethod('initBackend');
      _backendInitialized = true;
      print('LlamaCpp backend initialized successfully');
    } catch (e) {
      print('Error initializing backend: $e');
      throw e;
    }
  }
  
  Future<bool> loadModel(String modelPath) async {
    await initBackend(); // Ensure backend is initialized
    
    try {
      final result = await _channel.invokeMethod('loadModel', {
        'modelPath': modelPath,
      });
      
      if (result != null && result is int && result > 0) {
        _modelId = result;
        return true;
      }
      return false;
    } catch (e) {
      print('Error loading model: $e');
      return false;
    }
  }
  
  Future<bool> createContext() async {
    if (_modelId == null) return false;
    
    try {
      final result = await _channel.invokeMethod('createContext', {
        'modelId': _modelId,
      });
      
      if (result != null && result is int && result > 0) {
        _contextId = result;
        return true;
      }
      return false;
    } catch (e) {
      print('Error creating context: $e');
      return false;
    }
  }
  
  Future<String> generateText(String prompt, {int maxTokens = 100}) async {
    if (_contextId == null) return '';
    
    try {
      final result = await _channel.invokeMethod('generateText', {
        'contextId': _contextId,
        'inputText': prompt,
        'maxTokens': maxTokens,
      });
      
      return result?.toString() ?? '';
    } catch (e) {
      print('Error generating text: $e');
      return '';
    }
  }
  
  Future<void> dispose() async {
    if (_contextId != null) {
      try {
        await _channel.invokeMethod('freeContext', {
          'contextId': _contextId,
        });
      } catch (e) {
        print('Error freeing context: $e');
      }
      _contextId = null;
    }
    
    if (_modelId != null) {
      try {
        await _channel.invokeMethod('freeModel', {
          'modelId': _modelId,
        });
      } catch (e) {
        print('Error freeing model: $e');
      }
      _modelId = null;
    }
  }
}
