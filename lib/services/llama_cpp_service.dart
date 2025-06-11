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
      print('Error loading model: $e');      // Check for specific error types
      if (e.toString().contains('file too large') || e.toString().contains('Model file too large')) {
        throw Exception('Model file is too large (max 1GB). Please check if the file is corrupted.');
      } else if (e.toString().contains('Cannot open model file')) {
        throw Exception('Cannot access model file. Please check the file path and permissions.');
      } else if (e.toString().contains('Invalid format')) {
        throw Exception('Invalid model format. Please use a valid GGUF file.');
      }
      throw Exception('Failed to load model: ${e.toString()}');
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
  
  Future<bool> startStreaming(String prompt, {int maxTokens = 20}) async {
    if (_contextId == null) return false;
    
    try {
      final result = await _channel.invokeMethod('startStreaming', {
        'contextId': _contextId,
        'inputText': prompt,
        'maxTokens': maxTokens,
      });
      
      return result == true;
    } catch (e) {
      print('Error starting streaming: $e');
      return false;
    }
  }
  
  Future<String> getNextStreamingToken() async {
    if (_contextId == null) return '';
    
    try {
      final result = await _channel.invokeMethod('getNextStreamingToken', {
        'contextId': _contextId,
      });
      
      return result?.toString() ?? '';
    } catch (e) {
      print('Error getting next streaming token: $e');
      return '';
    }
  }
  
  Future<bool> isStreamingComplete() async {
    if (_contextId == null) return true;
    
    try {
      final result = await _channel.invokeMethod('isStreamingComplete', {
        'contextId': _contextId,
      });
      
      return result == true;
    } catch (e) {
      print('Error checking streaming completion: $e');
      return true;
    }
  }
  
  Future<void> stopStreaming() async {
    if (_contextId == null) return;
    
    try {
      await _channel.invokeMethod('stopStreaming', {
        'contextId': _contextId,
      });
    } catch (e) {
      print('Error stopping streaming: $e');
    }
  }
  
  Stream<String> generateTextStream(String prompt, {int maxTokens = 20}) async* {
    if (!await startStreaming(prompt, maxTokens: maxTokens)) {
      yield 'Error: Failed to start streaming';
      return;
    }
    
    while (!(await isStreamingComplete())) {
      final token = await getNextStreamingToken();
      if (token.isNotEmpty && token != '<unk>') {
        yield token;
      }
      
      // Small delay to prevent overwhelming the UI
      await Future.delayed(const Duration(milliseconds: 50));
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
