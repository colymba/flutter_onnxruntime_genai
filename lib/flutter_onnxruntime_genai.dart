/// Flutter ONNX Runtime GenAI Plugin
///
/// A Flutter FFI plugin that wraps the Microsoft ONNX Runtime GenAI C-API
/// for on-device multimodal inference, designed for vision-language models
/// like Phi-3.5 Vision.
///
/// ## Getting Started
///
/// ```dart
/// import 'package:flutter_onnxruntime_genai/flutter_onnxruntime_genai.dart';
///
/// // Create an instance
/// final onnx = OnnxGenAI();
///
/// // Check library health
/// final status = await onnx.checkNativeHealthAsync('/path/to/model');
/// print('Health status: ${HealthStatus.getMessage(status)}');
///
/// // Run inference (safe for main isolate - uses background processing)
/// final result = await onnx.runInferenceAsync(
///   modelPath: '/path/to/model',
///   prompt: 'Describe this image in detail.',
///   imagePath: '/path/to/image.jpg',
/// );
/// print('Generated: $result');
/// ```
///
/// ## Streaming Inference
///
/// For streaming token-by-token output:
///
/// ```dart
/// final streamer = OnnxGenAIStreamer();
///
/// await for (final token in streamer.streamInference(
///   modelPath: '/path/to/model',
///   prompt: 'Tell me a story.',
/// )) {
///   stdout.write(token); // Print each token as generated
/// }
/// ```
///
/// ## Important Notes
///
/// - All synchronous inference methods (`runInference`, `runTextInference`)
///   are BLOCKING and should only be called from background isolates.
/// - Use the async variants (`runInferenceAsync`, `runTextInferenceAsync`)
///   when calling from the main UI isolate.
/// - Model files must be present on device before inference can run.
library;

export 'src/ffi_bindings.dart'
    show
        OnnxGenAI,
        OnnxGenAIStreamer,
        OnnxGenAIException,
        HealthStatus,
        OnnxGenAIConfig;
