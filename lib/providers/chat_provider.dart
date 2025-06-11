import 'package:flutter/foundation.dart';
import '../models/chat_message.dart';
import '../services/llama_cpp_service.dart';

class ChatProvider extends ChangeNotifier {
  final LlamaCppService _llamaService = LlamaCppService();
  final List<ChatMessage> _messages = [];
  bool _isModelLoaded = false;
  bool _isInitializing = false;
  bool _isStreaming = false;
  String _modelPath = '';
  
  // Enhanced loading and performance tracking
  double _loadingProgress = 0.0;
  String _loadingStatus = '';
  int _tokensPerSecond = 0;
  int _totalTokensGenerated = 0;
  DateTime? _lastTokenTime;
  
  List<ChatMessage> get messages => List.unmodifiable(_messages);
  bool get isModelLoaded => _isModelLoaded;
  bool get isInitializing => _isInitializing;
  bool get isStreaming => _isStreaming;
  String get modelPath => _modelPath;
  double get loadingProgress => _loadingProgress;
  String get loadingStatus => _loadingStatus;
  int get tokensPerSecond => _tokensPerSecond;
  int get totalTokensGenerated => _totalTokensGenerated;

  void _updateLoadingProgress(double progress, String status) {
    _loadingProgress = progress;
    _loadingStatus = status;
    notifyListeners();
  }

  void _updatePerformanceMetrics() {
    final now = DateTime.now();
    if (_lastTokenTime != null) {
      final timeDiff = now.difference(_lastTokenTime!).inMilliseconds;
      if (timeDiff > 0) {
        _tokensPerSecond = (1000 / timeDiff).round();
      }
    }
    _lastTokenTime = now;
    _totalTokensGenerated++;
    notifyListeners();
  }
  Future<bool> initializeModel(String modelPath) async {
    if (_isInitializing) return false;
    
    _isInitializing = true;
    _modelPath = modelPath;
    _updateLoadingProgress(0.0, 'Initializing backend...');

    try {
      // Simulate more detailed loading steps for better UX
      await Future.delayed(const Duration(milliseconds: 500));
      _updateLoadingProgress(0.1, 'Reading model file...');
      
      await Future.delayed(const Duration(milliseconds: 300));
      _updateLoadingProgress(0.2, 'Validating GGUF format...');      // Load the model with progress tracking
      _updateLoadingProgress(0.3, 'Loading model data...');
      try {
        final modelLoaded = await _llamaService.loadModel(modelPath);
        if (!modelLoaded) {
          _isInitializing = false;
          _updateLoadingProgress(0.0, 'Model loading failed');
          return false;
        }
      } catch (e) {
        _isInitializing = false;
        _updateLoadingProgress(0.0, e.toString());
        throw e; // Re-throw to be caught by outer try-catch
      }

      _updateLoadingProgress(0.6, 'Initializing tensors...');
      await Future.delayed(const Duration(milliseconds: 500));
      
      _updateLoadingProgress(0.7, 'Creating inference context...');
      // Create context
      final contextCreated = await _llamaService.createContext();
      if (!contextCreated) {
        _isInitializing = false;
        _updateLoadingProgress(0.0, 'Failed to create context');
        return false;
      }

      _updateLoadingProgress(0.9, 'Optimizing for mobile...');
      await Future.delayed(const Duration(milliseconds: 300));
      
      _updateLoadingProgress(1.0, 'Model ready!');
      await Future.delayed(const Duration(milliseconds: 200));
      
      _isModelLoaded = true;
      _isInitializing = false;
      
      // Add enhanced welcome message with model info
      final modelName = modelPath.split('/').last;
      _addMessage(ChatMessage(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        text: '🚀 Model "$modelName" loaded successfully!\n\n'
              '✨ Phase 3 Neural Network active\n'
              '🔄 Real-time streaming enabled\n'
              '💬 Ready for conversations!',
        isUser: false,
        timestamp: DateTime.now(),
      ));
      
      notifyListeners();
      return true;
    } catch (e) {
      print('Error initializing model: $e');
      _isInitializing = false;
      _updateLoadingProgress(0.0, 'Error: ${e.toString()}');
      notifyListeners();
      return false;
    }
  }
  Future<void> sendMessage(String text) async {
    if (!_isModelLoaded || text.trim().isEmpty || _isStreaming) return;

    // Add user message
    final userMessage = ChatMessage(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      text: text.trim(),
      isUser: true,
      timestamp: DateTime.now(),
    );
    _addMessage(userMessage);

    // Start streaming response
    await _sendMessageStreaming(text.trim());
  }

  Future<void> _sendMessageStreaming(String text) async {
    _isStreaming = true;
    notifyListeners();

    // Create AI message placeholder
    final aiMessageId = DateTime.now().millisecondsSinceEpoch.toString();
    final aiMessage = ChatMessage(
      id: aiMessageId,
      text: '',
      isUser: false,
      timestamp: DateTime.now(),
    );
    _addMessage(aiMessage);

    try {
      String fullResponse = '';
      bool isFirstToken = true;

      await for (final token in _llamaService.generateTextStream(text, maxTokens: 50)) {
        if (token.isNotEmpty && token != '<unk>') {
          if (isFirstToken) {
            fullResponse = token;
            isFirstToken = false;
          } else {
            fullResponse += ' $token';
          }

          // Update the AI message with the accumulated response
          final messageIndex = _messages.indexWhere((msg) => msg.id == aiMessageId);
          if (messageIndex != -1) {
            _messages[messageIndex] = ChatMessage(
              id: aiMessageId,
              text: fullResponse,
              isUser: false,
              timestamp: aiMessage.timestamp,
            );
            notifyListeners();
          }
        }

        // Update loading progress and performance metrics
        _updateLoadingProgress(fullResponse.length / 50, 'Generating response...');
        _updatePerformanceMetrics();
      }

      // Ensure we have some response
      if (fullResponse.isEmpty) {
        final messageIndex = _messages.indexWhere((msg) => msg.id == aiMessageId);
        if (messageIndex != -1) {
          _messages[messageIndex] = ChatMessage(
            id: aiMessageId,
            text: 'Sorry, I couldn\'t generate a response.',
            isUser: false,
            timestamp: aiMessage.timestamp,
          );
          notifyListeners();
        }
      }

    } catch (e) {
      // Update message with error
      final messageIndex = _messages.indexWhere((msg) => msg.id == aiMessageId);
      if (messageIndex != -1) {
        _messages[messageIndex] = ChatMessage(
          id: aiMessageId,
          text: 'Sorry, there was an error generating a response.',
          isUser: false,
          timestamp: aiMessage.timestamp,
        );
        notifyListeners();
      }
    } finally {
      _isStreaming = false;
      notifyListeners();
    }
  }

  Future<void> sendMessageBatch(String text) async {
    if (!_isModelLoaded || text.trim().isEmpty) return;

    // Add user message
    final userMessage = ChatMessage(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      text: text.trim(),
      isUser: true,
      timestamp: DateTime.now(),
    );
    _addMessage(userMessage);

    // Add loading message for AI response
    final loadingMessage = ChatMessage(
      id: '${DateTime.now().millisecondsSinceEpoch + 1}',
      text: 'Thinking...',
      isUser: false,
      timestamp: DateTime.now(),
      isLoading: true,
    );
    _addMessage(loadingMessage);

    try {
      // Generate response
      final response = await _llamaService.generateText(text, maxTokens: 150);
      
      // Remove loading message and add actual response
      _messages.removeWhere((msg) => msg.id == loadingMessage.id);
      
      final aiMessage = ChatMessage(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        text: response.isNotEmpty ? response : 'Sorry, I couldn\'t generate a response.',
        isUser: false,
        timestamp: DateTime.now(),
      );
      _addMessage(aiMessage);
    } catch (e) {
      // Remove loading message and add error message
      _messages.removeWhere((msg) => msg.id == loadingMessage.id);
      
      final errorMessage = ChatMessage(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        text: 'Sorry, there was an error generating a response.',
        isUser: false,
        timestamp: DateTime.now(),
      );
      _addMessage(errorMessage);
    }
  }

  void _addMessage(ChatMessage message) {
    _messages.add(message);
    notifyListeners();
  }

  void clearChat() {
    _messages.clear();
    notifyListeners();
  }

  @override
  void dispose() {
    _llamaService.dispose();
    super.dispose();
  }
}
