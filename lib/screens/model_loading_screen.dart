import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/chat_provider.dart';
import 'chat_screen.dart';

class ModelLoadingScreen extends StatefulWidget {
  final String modelPath;

  const ModelLoadingScreen({
    super.key,
    required this.modelPath,
  });

  @override
  State<ModelLoadingScreen> createState() => _ModelLoadingScreenState();
}

class _ModelLoadingScreenState extends State<ModelLoadingScreen>
    with TickerProviderStateMixin {
  late AnimationController _pulseController;
  late AnimationController _rotationController;
  late Animation<double> _pulseAnimation;
  late Animation<double> _rotationAnimation;

  @override
  void initState() {
    super.initState();
    
    // Initialize animations
    _pulseController = AnimationController(
      duration: const Duration(seconds: 2),
      vsync: this,
    )..repeat(reverse: true);
    
    _rotationController = AnimationController(
      duration: const Duration(seconds: 4),
      vsync: this,
    )..repeat();
    
    _pulseAnimation = Tween<double>(
      begin: 0.8,
      end: 1.2,
    ).animate(CurvedAnimation(
      parent: _pulseController,
      curve: Curves.easeInOut,
    ));
    
    _rotationAnimation = Tween<double>(
      begin: 0,
      end: 1,
    ).animate(_rotationController);
    
    // Start loading the model
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _loadModel();
    });
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _rotationController.dispose();
    super.dispose();
  }

  Future<void> _loadModel() async {
    final chatProvider = Provider.of<ChatProvider>(context, listen: false);
    
    try {
      final success = await chatProvider.initializeModel(widget.modelPath);
      
      if (mounted) {
        if (success) {
          // Navigate to chat screen after successful loading
          Navigator.of(context).pushReplacement(
            MaterialPageRoute(
              builder: (context) => const ChatScreen(),
            ),
          );        } else {
          // Show error and go back
          final errorMessage = chatProvider.loadingStatus.isNotEmpty 
              ? chatProvider.loadingStatus 
              : 'Failed to load model. Please try again.';
          
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(errorMessage),
              backgroundColor: Colors.red,
              duration: const Duration(seconds: 5),
              action: SnackBarAction(
                label: 'OK',
                textColor: Colors.white,
                onPressed: () {
                  ScaffoldMessenger.of(context).hideCurrentSnackBar();
                },
              ),
            ),
          );
          Navigator.of(context).pop();
        }
      }    } catch (e) {
      if (mounted) {
        String errorMessage = 'Error loading model: ${e.toString()}';
          // Clean up the error message for better user experience
        if (e.toString().contains('Model file is too large')) {
          errorMessage = 'Model file is too large (max 1GB).\n\nThis is unusual - most GGUF models should work.\nPlease check if the file is corrupted.';
        } else if (e.toString().contains('Cannot access model file')) {
          errorMessage = 'Cannot access the selected file.\n\nPlease check file permissions and try again.';
        } else if (e.toString().contains('Invalid model format')) {
          errorMessage = 'Invalid model format.\n\nPlease select a valid GGUF model file.';
        }
        
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(errorMessage),
            backgroundColor: Colors.red,
            duration: const Duration(seconds: 8),
            action: SnackBarAction(
              label: 'OK',
              textColor: Colors.white,
              onPressed: () {
                ScaffoldMessenger.of(context).hideCurrentSnackBar();
              },
            ),
          ),
        );
        Navigator.of(context).pop();
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Theme.of(context).primaryColor.withOpacity(0.1),
      body: Consumer<ChatProvider>(
        builder: (context, chatProvider, child) {
          return SafeArea(
            child: Padding(
              padding: const EdgeInsets.all(24.0),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  // Animated loading indicator
                  AnimatedBuilder(
                    animation: _pulseAnimation,
                    builder: (context, child) {
                      return Transform.scale(
                        scale: _pulseAnimation.value,
                        child: AnimatedBuilder(
                          animation: _rotationAnimation,
                          builder: (context, child) {
                            return Transform.rotate(
                              angle: _rotationAnimation.value * 2 * 3.14159,
                              child: Container(
                                width: 120,
                                height: 120,
                                decoration: BoxDecoration(
                                  shape: BoxShape.circle,
                                  gradient: LinearGradient(
                                    colors: [
                                      Theme.of(context).primaryColor,
                                      Theme.of(context).primaryColor.withOpacity(0.3),
                                    ],
                                    begin: Alignment.topLeft,
                                    end: Alignment.bottomRight,
                                  ),
                                  boxShadow: [
                                    BoxShadow(
                                      color: Theme.of(context).primaryColor.withOpacity(0.3),
                                      blurRadius: 20,
                                      spreadRadius: 5,
                                    ),
                                  ],
                                ),
                                child: const Icon(
                                  Icons.smart_toy,
                                  size: 60,
                                  color: Colors.white,
                                ),
                              ),
                            );
                          },
                        ),
                      );
                    },
                  ),
                  
                  const SizedBox(height: 40),
                  
                  // Progress indicator
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(16),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.1),
                          blurRadius: 20,
                          offset: const Offset(0, 10),
                        ),
                      ],
                    ),
                    child: Column(
                      children: [
                        Text(
                          'Loading AI Model',
                          style: TextStyle(
                            fontSize: 24,
                            fontWeight: FontWeight.bold,
                            color: Theme.of(context).primaryColor,
                          ),
                        ),
                        
                        const SizedBox(height: 16),
                        
                        // Model name
                        Container(
                          padding: const EdgeInsets.symmetric(
                            horizontal: 12,
                            vertical: 6,
                          ),
                          decoration: BoxDecoration(
                            color: Theme.of(context).primaryColor.withOpacity(0.1),
                            borderRadius: BorderRadius.circular(20),
                          ),
                          child: Text(
                            widget.modelPath.split('/').last,
                            style: TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w500,
                              color: Theme.of(context).primaryColor,
                            ),
                          ),
                        ),
                        
                        const SizedBox(height: 24),
                        
                        // Progress bar
                        Column(
                          children: [
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceBetween,
                              children: [
                                Text(
                                  chatProvider.loadingStatus.isNotEmpty 
                                    ? chatProvider.loadingStatus 
                                    : 'Initializing...',
                                  style: const TextStyle(
                                    fontSize: 16,
                                    fontWeight: FontWeight.w500,
                                  ),
                                ),
                                Text(
                                  '${(chatProvider.loadingProgress * 100).toInt()}%',
                                  style: TextStyle(
                                    fontSize: 16,
                                    fontWeight: FontWeight.bold,
                                    color: Theme.of(context).primaryColor,
                                  ),
                                ),
                              ],
                            ),
                            
                            const SizedBox(height: 12),
                            
                            ClipRRect(
                              borderRadius: BorderRadius.circular(10),
                              child: LinearProgressIndicator(
                                value: chatProvider.loadingProgress,
                                minHeight: 8,
                                backgroundColor: Colors.grey[300],
                                valueColor: AlwaysStoppedAnimation<Color>(
                                  Theme.of(context).primaryColor,
                                ),
                              ),
                            ),
                          ],
                        ),
                        
                        const SizedBox(height: 20),
                        
                        // Loading steps
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                          children: [
                            _buildLoadingStep(
                              icon: Icons.folder_open,
                              label: 'Reading',
                              isActive: chatProvider.loadingProgress >= 0.1,
                              isCompleted: chatProvider.loadingProgress >= 0.3,
                            ),
                            _buildLoadingStep(
                              icon: Icons.memory,
                              label: 'Loading',
                              isActive: chatProvider.loadingProgress >= 0.3,
                              isCompleted: chatProvider.loadingProgress >= 0.7,
                            ),
                            _buildLoadingStep(
                              icon: Icons.settings,
                              label: 'Setup',
                              isActive: chatProvider.loadingProgress >= 0.7,
                              isCompleted: chatProvider.loadingProgress >= 0.9,
                            ),
                            _buildLoadingStep(
                              icon: Icons.check_circle,
                              label: 'Ready',
                              isActive: chatProvider.loadingProgress >= 0.9,
                              isCompleted: chatProvider.loadingProgress >= 1.0,
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                  
                  const SizedBox(height: 40),
                  
                  // Technical info
                  Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: Colors.black.withOpacity(0.05),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Column(
                      children: [
                        Row(
                          children: [
                            Icon(
                              Icons.info_outline,
                              size: 16,
                              color: Colors.grey[600],
                            ),
                            const SizedBox(width: 8),
                            Text(
                              'Phase 3 Neural Network Engine',
                              style: TextStyle(
                                fontSize: 14,
                                fontWeight: FontWeight.w500,
                                color: Colors.grey[600],
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'Loading real tensor data and quantization layers...',
                          style: TextStyle(
                            fontSize: 12,
                            color: Colors.grey[500],
                          ),
                          textAlign: TextAlign.center,
                        ),
                      ],
                    ),
                  ),
                  
                  const SizedBox(height: 20),
                  
                  // Cancel button
                  TextButton(
                    onPressed: () {
                      Navigator.of(context).pop();
                    },
                    child: const Text('Cancel'),
                  ),
                ],
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildLoadingStep({
    required IconData icon,
    required String label,
    required bool isActive,
    required bool isCompleted,
  }) {
    Color color;
    if (isCompleted) {
      color = Colors.green;
    } else if (isActive) {
      color = Theme.of(context).primaryColor;
    } else {
      color = Colors.grey;
    }

    return Column(
      children: [
        AnimatedContainer(
          duration: const Duration(milliseconds: 300),
          width: 40,
          height: 40,
          decoration: BoxDecoration(
            color: isCompleted ? Colors.green : (isActive ? Theme.of(context).primaryColor.withOpacity(0.1) : Colors.grey.withOpacity(0.1)),
            shape: BoxShape.circle,
            border: Border.all(
              color: color,
              width: 2,
            ),
          ),
          child: Icon(
            isCompleted ? Icons.check : icon,
            color: color,
            size: 20,
          ),
        ),
        const SizedBox(height: 8),
        Text(
          label,
          style: TextStyle(
            fontSize: 12,
            color: color,
            fontWeight: isActive ? FontWeight.bold : FontWeight.normal,
          ),
        ),
      ],
    );
  }
}
