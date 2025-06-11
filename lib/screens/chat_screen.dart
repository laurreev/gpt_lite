import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/chat_provider.dart';
import '../widgets/message_bubble.dart';
import 'model_selection_screen.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key});

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final TextEditingController _messageController = TextEditingController();
  final ScrollController _scrollController = ScrollController();

  @override
  void dispose() {
    _messageController.dispose();
    _scrollController.dispose();
    super.dispose();
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  void _sendMessage(ChatProvider chatProvider) {
    final text = _messageController.text.trim();
    if (text.isNotEmpty) {
      chatProvider.sendMessage(text);
      _messageController.clear();
      _scrollToBottom();
    }
  }

  void _goBackToModelSelection() {
    Navigator.of(context).pushReplacement(
      MaterialPageRoute(
        builder: (context) => const ModelSelectionScreen(),
      ),
    );
  }
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Consumer<ChatProvider>(
          builder: (context, chatProvider, child) {
            String title = 'GPT Lite';
            if (chatProvider.isStreaming) {
              title += ' • ${chatProvider.tokensPerSecond} tok/s';
            } else if (chatProvider.isModelLoaded) {
              title += ' • Ready (${chatProvider.totalTokensGenerated} tokens)';
            }
            return Text(title);
          },
        ),
        backgroundColor: Theme.of(context).primaryColor,
        foregroundColor: Colors.white,
        actions: [
          Consumer<ChatProvider>(
            builder: (context, chatProvider, child) {
              return PopupMenuButton<String>(
                onSelected: (value) {
                  switch (value) {
                    case 'change_model':
                      _goBackToModelSelection();
                      break;
                    case 'clear_chat':
                      chatProvider.clearChat();
                      break;
                  }
                },
                itemBuilder: (context) => [
                  const PopupMenuItem(
                    value: 'change_model',
                    child: Row(
                      children: [
                        Icon(Icons.swap_horiz),
                        SizedBox(width: 8),
                        Text('Change Model'),
                      ],
                    ),
                  ),
                  if (chatProvider.isModelLoaded)
                    const PopupMenuItem(
                      value: 'clear_chat',
                      child: Row(
                        children: [
                          Icon(Icons.clear_all),
                          SizedBox(width: 8),
                          Text('Clear Chat'),
                        ],
                      ),
                    ),
                ],
              );
            },
          ),
        ],
      ),
      body: Consumer<ChatProvider>(
        builder: (context, chatProvider, child) {
          // Chat screen only shows when model is loaded
          if (!chatProvider.isModelLoaded) {
            return const Center(
              child: CircularProgressIndicator(),
            );
          }

          return Column(
            children: [
              _buildModelInfo(chatProvider),
              Expanded(
                child: _buildMessagesList(chatProvider),
              ),
              _buildMessageInput(chatProvider),
            ],
          );
        },
      ),
    );
  }
  Widget _buildModelInfo(ChatProvider chatProvider) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(8),
      color: Colors.green.shade50,
      child: Column(
        children: [
          Text(
            'Model: ${chatProvider.modelPath.split('/').last}',
            style: const TextStyle(
              fontSize: 12,
              color: Colors.green,
              fontWeight: FontWeight.bold,
            ),
            textAlign: TextAlign.center,
          ),
          if (chatProvider.isStreaming || chatProvider.totalTokensGenerated > 0)
            Text(
              '⚡ ${chatProvider.tokensPerSecond} tok/s • 📊 ${chatProvider.totalTokensGenerated} total',
              style: const TextStyle(
                fontSize: 10,
                color: Colors.green,
              ),
              textAlign: TextAlign.center,
            ),
        ],
      ),
    );
  }

  Widget _buildMessagesList(ChatProvider chatProvider) {
    if (chatProvider.messages.isEmpty) {
      return const Center(
        child: Text(
          'Start a conversation!',
          style: TextStyle(
            fontSize: 16,
            color: Colors.grey,
          ),
        ),
      );
    }

    return ListView.builder(
      controller: _scrollController,
      itemCount: chatProvider.messages.length,
      itemBuilder: (context, index) {
        final message = chatProvider.messages[index];
        return MessageBubble(message: message);
      },
    );
  }

  Widget _buildMessageInput(ChatProvider chatProvider) {
    return Container(
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.grey.shade300,
            blurRadius: 4,
            offset: const Offset(0, -2),
          ),
        ],
      ),
      child: Row(
        children: [
          Expanded(
            child: TextField(
              controller: _messageController,
              decoration: const InputDecoration(
                hintText: 'Type your message...',
                border: OutlineInputBorder(),
                contentPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              ),
              maxLines: null,
              onSubmitted: (_) => _sendMessage(chatProvider),
            ),
          ),
          const SizedBox(width: 8),
          IconButton(
            onPressed: (chatProvider.isModelLoaded && !chatProvider.isStreaming)
                ? () => _sendMessage(chatProvider) 
                : null,
            icon: chatProvider.isStreaming 
                ? const SizedBox(
                    width: 16,
                    height: 16,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                    ),
                  )
                : const Icon(Icons.send),
            style: IconButton.styleFrom(
              backgroundColor: (chatProvider.isModelLoaded && !chatProvider.isStreaming)
                  ? Theme.of(context).primaryColor
                  : Colors.grey,
              foregroundColor: Colors.white,
            ),
          ),
        ],
      ),
    );
  }
}
