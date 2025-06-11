import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'providers/chat_provider.dart';
import 'screens/model_selection_screen.dart';

void main() {
  runApp(const MainApp());
}

class MainApp extends StatelessWidget {
  const MainApp({super.key});
  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (context) => ChatProvider(),
      child: MaterialApp(
        title: 'GPT Lite',
        theme: ThemeData(
          primarySwatch: Colors.blue,
          useMaterial3: true,
        ),
        home: const ModelSelectionScreen(),
        debugShowCheckedModeBanner: false,
      ),
    );
  }
}
