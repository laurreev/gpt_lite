# GPT Lite - Offline AI Chat Application

An offline ChatGPT-like Android application built with Flutter and llama.cpp for running large language models locally on your device.

## 🚀 Features

- 💬 **Offline AI Chat**: Run AI models completely offline without internet connection
- 📱 **Native Android Performance**: Built with Flutter for smooth user experience  
- 🔧 **llama.cpp Integration**: Framework ready for llama.cpp engine integration
- 📁 **Local Model Loading**: Load GGUF model files directly from your device
- 🎨 **Modern Chat Interface**: Clean and intuitive Material Design chat UI
- 🔒 **Privacy First**: All conversations happen locally on your device
- 🛠️ **Easy Setup**: Automated scripts for llama.cpp integration

## 📱 Screenshots

The app features a modern chat interface with:
- Welcome screen with model loading
- Chat bubbles for user and AI messages
- Loading indicators for AI responses
- File picker for model selection
- Menu options for chat management

## 🏗️ Current Status

**✅ Completed:**
- Complete Flutter UI implementation
- Native Android JNI bridge setup
- Chat interface with message handling
- File picker for model loading
- Provider-based state management
- Stub implementations for testing

**🔄 In Progress:**
- Real llama.cpp integration (setup scripts provided)
- Performance optimizations
- Memory management improvements

**📋 Planned:**
- GPU acceleration support
- More model format support
- iOS version
- Advanced chat features

## 🚀 Quick Start

### Prerequisites
- Flutter SDK (3.7.2 or higher)
- Android Studio with NDK
- Android device/emulator (API 21+)
- Git for cloning repositories

### Installation

1. **Clone and setup the project:**
   ```bash
   git clone <repository-url>
   cd gpt_lite
   flutter pub get
   ```

2. **Test the demo version:**
   ```bash
   flutter run
   ```
   This runs with stub implementations for testing the UI.

3. **Integrate real llama.cpp (optional):**
   
   **Windows:**
   ```cmd
   setup_llama.bat
   ```
   
   **Linux/Mac:**
   ```bash
   chmod +x setup_llama.sh
   ./setup_llama.sh
   ```

4. **Build for Android:**
   ```bash
   flutter build apk --release
   ```

## 📖 Usage

### Demo Mode
1. Launch the app
2. Tap the menu → "Enable Demo Mode"
3. Start chatting with sample responses

### With Real Models (after llama.cpp integration)
1. Download a GGUF model file
2. Launch the app
3. Use "Load Model File" to select your model
4. Wait for model initialization
5. Start chatting!

## 🎯 Recommended Models

For mobile devices, use quantized models:

**Small Models (Testing):**
- `phi-2.Q4_K_M.gguf` (~1.6GB)
- `TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf` (~0.6GB)

**Medium Models (Better Quality):**
- `llama-2-7b-chat.Q4_K_M.gguf` (~4GB)
- `mistral-7b-instruct-v0.1.Q4_K_M.gguf` (~4GB)

**Download from:**
- [Hugging Face](https://huggingface.co/models?filter=gguf)
- [TheBloke's Collections](https://huggingface.co/TheBloke)

## 🏗️ Architecture

### Flutter Layer
```
lib/
├── main.dart                    # App entry point
├── models/
│   └── chat_message.dart       # Message data model
├── providers/
│   └── chat_provider.dart      # State management
├── screens/
│   ├── chat_screen.dart        # Full-featured chat (with provider)
│   └── simple_chat_screen.dart # Demo version
├── services/
│   └── llama_cpp_service.dart  # Native bridge service
└── widgets/
    └── message_bubble.dart     # Chat message components
```

### Native Layer
```
android/app/src/main/cpp/
├── llama.h                     # C++ header definitions
├── llama_bridge.cpp           # JNI bridge implementation
├── llama_stub.cpp             # Stub implementations (demo)
└── CMakeLists.txt             # Native build configuration
```

### Integration Layer
```
android/app/src/main/kotlin/com/example/gpt_lite/
├── MainActivity.kt            # Flutter activity
└── LlamaCppPlugin.kt         # JNI plugin bridge
```

## 🔧 Development

### Running in Development
```bash
# Run with hot reload
flutter run

# Run on specific device
flutter devices
flutter run -d <device-id>

# Build debug APK
flutter build apk --debug
```

### Testing the Interface
The app includes a demo mode that works without real models:
1. Use `SimpleChatScreen` for basic testing
2. Enable demo mode for sample responses
3. Test all UI components and interactions

### Integration Testing
After llama.cpp integration:
1. Test with small models first
2. Monitor memory usage
3. Test on different Android versions
4. Verify model loading and inference

## 📚 Documentation

- **[Integration Guide](INTEGRATION_GUIDE.md)** - Detailed llama.cpp setup
- **[Flutter Documentation](https://flutter.dev/docs)** - Flutter development
- **[llama.cpp Repository](https://github.com/ggerganov/llama.cpp)** - Core AI engine

## 🤝 Contributing

Contributions welcome! Areas needing help:
- Real llama.cpp integration
- Performance optimizations
- UI/UX improvements
- Documentation
- Testing on different devices

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source under the [MIT License](LICENSE).

## ⚠️ Important Notes

- **Demo Version**: Current version includes stub implementations
- **Memory Requirements**: Real models need significant RAM (4-8GB+)
- **Performance**: CPU-only inference is slower than GPU
- **Model Compatibility**: Only GGUF format currently supported
- **Android Only**: iOS version not yet implemented

## 🆘 Support

**For GPT Lite specific issues:**
- Create an issue in this repository
- Include device info and logs
- Describe steps to reproduce

**For llama.cpp related issues:**
- Check the [llama.cpp repository](https://github.com/ggerganov/llama.cpp)
- Review their documentation and issues

**For Flutter development:**
- Check [Flutter documentation](https://flutter.dev/docs)
- Visit [Flutter community](https://flutter.dev/community)

---

**Built with ❤️ using Flutter and llama.cpp**

