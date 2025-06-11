import 'package:flutter/foundation.dart';
import '../models/chat_message.dart';
import '../services/llama_cpp_service.dart';

class ChatProvider extends ChangeNotifier {
  final LlamaCppService _llamaService = LlamaCppService();
  final List<ChatMessage> _messages = [];
  bool _isModelLoaded = false;
  bool _isInitializing = false;
  String _modelPath = '';

  List<ChatMessage> get messages => List.unmodifiable(_messages);
  bool get isModelLoaded => _isModelLoaded;
  bool get isInitializing => _isInitializing;
  String get modelPath => _modelPath;

  Future<bool> initializeModel(String modelPath) async {
    if (_isInitializing) return false;
    
    _isInitializing = true;
    _modelPath = modelPath;
    notifyListeners();

    try {
      // Load the model
      final modelLoaded = await _llamaService.loadModel(modelPath);
      if (!modelLoaded) {
        _isInitializing = false;
        notifyListeners();
        return false;
      }

      // Create context
      final contextCreated = await _llamaService.createContext();
      if (!contextCreated) {
        _isInitializing = false;
        notifyListeners();
        return false;
      }

      _isModelLoaded = true;
      _isInitializing = false;
      
      // Add welcome message
      _addMessage(ChatMessage(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        text: 'Model loaded successfully! You can now start chatting.',
        isUser: false,
        timestamp: DateTime.now(),
      ));
      
      notifyListeners();
      return true;
    } catch (e) {
      print('Error initializing model: $e');
      _isInitializing = false;
      notifyListeners();
      return false;
    }
  }

  Future<void> sendMessage(String text) async {
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
