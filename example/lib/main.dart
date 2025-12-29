import 'package:flutter/material.dart';

import 'package:flutter_onnxruntime_genai/flutter_onnxruntime_genai.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  String _libraryVersion = 'Unknown';
  String _status = 'Not tested';
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _loadLibraryInfo();
  }

  Future<void> _loadLibraryInfo() async {
    try {
      final onnx = OnnxGenAI();
      setState(() {
        _libraryVersion = onnx.libraryVersion;
        _status = 'Library loaded successfully';
      });
    } on OnnxGenAIException catch (e) {
      setState(() {
        _libraryVersion = 'Error';
        _status = e.message;
      });
    }
  }

  Future<void> _testHealthCheck() async {
    setState(() {
      _isLoading = true;
      _status = 'Testing model health...';
    });

    try {
      final onnx = OnnxGenAI();
      // Replace with actual model path for testing
      const modelPath = '/path/to/your/model';
      final healthStatus = await onnx.checkNativeHealthAsync(modelPath);
      setState(() {
        _status = HealthStatus.getMessage(healthStatus);
      });
    } on OnnxGenAIException catch (e) {
      setState(() {
        _status = 'Error: ${e.message}';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    const textStyle = TextStyle(fontSize: 16);
    const spacerSmall = SizedBox(height: 16);

    return MaterialApp(
      theme: ThemeData.dark(),
      home: Scaffold(
        appBar: AppBar(title: const Text('ONNX Runtime GenAI')),
        body: SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Flutter ONNX Runtime GenAI Plugin',
                style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
              ),
              spacerSmall,
              const Text(
                'This plugin wraps the Microsoft ONNX Runtime GenAI C-API '
                'for on-device multimodal inference.',
                style: textStyle,
              ),
              spacerSmall,
              spacerSmall,
              Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'Library Info',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 8),
                      Text('Version: $_libraryVersion', style: textStyle),
                      const SizedBox(height: 4),
                      Text('Status: $_status', style: textStyle),
                    ],
                  ),
                ),
              ),
              spacerSmall,
              Center(
                child: ElevatedButton(
                  onPressed: _isLoading ? null : _testHealthCheck,
                  child: _isLoading
                      ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(strokeWidth: 2),
                        )
                      : const Text('Test Model Health'),
                ),
              ),
              spacerSmall,
              spacerSmall,
              const Text(
                'Usage Example',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              spacerSmall,
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.grey[850],
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Text('''final onnx = OnnxGenAI();

// Run inference
final result = await onnx.runInferenceAsync(
  modelPath: '/path/to/model',
  prompt: 'Describe this image.',
  imagePath: '/path/to/image.jpg',
);''', style: TextStyle(fontFamily: 'monospace', fontSize: 13)),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
