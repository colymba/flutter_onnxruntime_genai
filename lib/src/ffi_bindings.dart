/// Flutter ONNX Runtime GenAI - FFI Bindings
///
/// This file contains the Dart FFI bindings for the ONNX Runtime GenAI
/// native library. It provides type-safe Dart wrappers around the C functions.
///
/// IMPORTANT: All inference functions are long-running operations.
/// They MUST be called from a background Isolate, NOT from the main UI isolate.
/// Use [OnnxGenAIIsolate] or the provided async methods.
library;

import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';
import 'dart:async';

import 'package:ffi/ffi.dart';

// =============================================================================
// Native Function Type Definitions
// =============================================================================

/// Native function: int32_t check_native_health(const char* model_path)
typedef CheckNativeHealthNative = Int32 Function(Pointer<Utf8> modelPath);
typedef CheckNativeHealthDart = int Function(Pointer<Utf8> modelPath);

/// Native function: const char* run_inference(const char* model_path, const char* prompt, const char* image_path)
typedef RunInferenceNative =
    Pointer<Utf8> Function(
      Pointer<Utf8> modelPath,
      Pointer<Utf8> prompt,
      Pointer<Utf8> imagePath,
    );
typedef RunInferenceDart =
    Pointer<Utf8> Function(
      Pointer<Utf8> modelPath,
      Pointer<Utf8> prompt,
      Pointer<Utf8> imagePath,
    );

/// Native function: const char* run_text_inference(const char* model_path, const char* prompt, int32_t max_length)
typedef RunTextInferenceNative =
    Pointer<Utf8> Function(
      Pointer<Utf8> modelPath,
      Pointer<Utf8> prompt,
      Int32 maxLength,
    );
typedef RunTextInferenceDart =
    Pointer<Utf8> Function(
      Pointer<Utf8> modelPath,
      Pointer<Utf8> prompt,
      int maxLength,
    );

/// Native function: const char* run_inference_multi(const char* model_path, const char* prompt, const char** image_paths, int32_t image_count)
typedef RunInferenceMultiNative =
    Pointer<Utf8> Function(
      Pointer<Utf8> modelPath,
      Pointer<Utf8> prompt,
      Pointer<Pointer<Utf8>> imagePaths,
      Int32 imageCount,
    );
typedef RunInferenceMultiDart =
    Pointer<Utf8> Function(
      Pointer<Utf8> modelPath,
      Pointer<Utf8> prompt,
      Pointer<Pointer<Utf8>> imagePaths,
      int imageCount,
    );

/// Native function: const char* get_library_version()
typedef GetLibraryVersionNative = Pointer<Utf8> Function();
typedef GetLibraryVersionDart = Pointer<Utf8> Function();

/// Native function: void shutdown_onnx_genai()
typedef ShutdownOnnxGenAINative = Void Function();
typedef ShutdownOnnxGenAIDart = void Function();

// =============================================================================
// Health Check Status Codes
// =============================================================================

/// Status codes returned by [checkNativeHealth].
class HealthStatus {
  /// Model loaded and verified successfully.
  static const int success = 1;

  /// NULL or empty path provided.
  static const int invalidPath = -1;

  /// Model creation failed.
  static const int modelCreationFailed = -2;

  /// Tokenizer creation failed.
  static const int tokenizerCreationFailed = -3;

  /// Returns a human-readable message for a status code.
  static String getMessage(int status) {
    switch (status) {
      case success:
        return 'Model loaded and verified successfully';
      case invalidPath:
        return 'Invalid model path provided';
      case modelCreationFailed:
        return 'Failed to create model - check if model files exist and are valid';
      case tokenizerCreationFailed:
        return 'Failed to create tokenizer - model may be corrupted';
      default:
        return 'Unknown status code: $status';
    }
  }
}

// =============================================================================
// OnnxGenAI - Main FFI Binding Class
// =============================================================================

/// Main class for interacting with ONNX Runtime GenAI via FFI.
///
/// This class loads the native library and provides Dart wrappers for the
/// C FFI functions. For inference operations, use the async methods which
/// automatically handle isolate management.
///
/// Example:
/// ```dart
/// final onnx = OnnxGenAI();
///
/// // Check if library is loaded
/// print('Library version: ${onnx.libraryVersion}');
///
/// // Run inference (automatically uses background isolate)
/// final result = await onnx.runInferenceAsync(
///   modelPath: '/path/to/model',
///   prompt: 'Describe this image.',
///   imagePath: '/path/to/image.jpg',
/// );
/// ```
class OnnxGenAI {
  /// Creates an instance of [OnnxGenAI] and loads the native library.
  ///
  /// Throws [OnnxGenAIException] if the library cannot be loaded.
  factory OnnxGenAI() => _instance ??= OnnxGenAI._();

  OnnxGenAI._() {
    _loadLibrary();
    _bindFunctions();
  }

  static OnnxGenAI? _instance;

  /// The loaded dynamic library.
  late final DynamicLibrary _dylib;

  // Bound native functions
  late final CheckNativeHealthDart _checkNativeHealth;
  late final RunInferenceDart _runInference;
  late final RunInferenceMultiDart _runInferenceMulti;
  late final RunTextInferenceDart _runTextInference;
  late final GetLibraryVersionDart _getLibraryVersion;
  late final ShutdownOnnxGenAIDart _shutdownOnnxGenAI;

  // Track worker isolate for cleanup
  Isolate? _workerIsolate;

  /// Library name without platform-specific prefix/suffix.
  static const String _libName = 'flutter_onnxruntime_genai';

  /// Loads the platform-specific dynamic library.
  void _loadLibrary() {
    try {
      if (Platform.isMacOS || Platform.isIOS) {
        _dylib = DynamicLibrary.open('$_libName.framework/$_libName');
      } else if (Platform.isAndroid || Platform.isLinux) {
        _dylib = DynamicLibrary.open('lib$_libName.so');
      } else if (Platform.isWindows) {
        _dylib = DynamicLibrary.open('$_libName.dll');
      } else {
        throw OnnxGenAIException(
          'Unsupported platform: ${Platform.operatingSystem}',
        );
      }
    } catch (e) {
      throw OnnxGenAIException('Failed to load native library: $e');
    }
  }

  /// Binds Dart functions to native symbols.
  void _bindFunctions() {
    _checkNativeHealth = _dylib
        .lookup<NativeFunction<CheckNativeHealthNative>>('check_native_health')
        .asFunction<CheckNativeHealthDart>();

    _runInference = _dylib
        .lookup<NativeFunction<RunInferenceNative>>('run_inference')
        .asFunction<RunInferenceDart>();

    _runInferenceMulti = _dylib
        .lookup<NativeFunction<RunInferenceMultiNative>>('run_inference_multi')
        .asFunction<RunInferenceMultiDart>();

    _runTextInference = _dylib
        .lookup<NativeFunction<RunTextInferenceNative>>('run_text_inference')
        .asFunction<RunTextInferenceDart>();

    _getLibraryVersion = _dylib
        .lookup<NativeFunction<GetLibraryVersionNative>>('get_library_version')
        .asFunction<GetLibraryVersionDart>();

    _shutdownOnnxGenAI = _dylib
        .lookup<NativeFunction<ShutdownOnnxGenAINative>>('shutdown_onnx_genai')
        .asFunction<ShutdownOnnxGenAIDart>();
  }

  // ===========================================================================
  // Public API - Synchronous (for use in Isolates)
  // ===========================================================================

  /// Gets the native library version.
  String get libraryVersion {
    final ptr = _getLibraryVersion();
    return ptr.toDartString();
  }

  /// Checks if the native library and model can be loaded.
  ///
  /// WARNING: This function loads the model and may take several seconds.
  /// Consider calling from a background isolate for large models.
  ///
  /// Returns a status code. Use [HealthStatus] to interpret the result.
  int checkNativeHealth(String modelPath) {
    final modelPathPtr = modelPath.toNativeUtf8();
    try {
      return _checkNativeHealth(modelPathPtr);
    } finally {
      calloc.free(modelPathPtr);
    }
  }

  /// Runs multimodal inference with text and optional image.
  ///
  /// WARNING: This is a LONG-RUNNING, BLOCKING operation!
  /// DO NOT call from the main UI isolate. Use [runInferenceAsync] instead.
  ///
  /// Parameters:
  /// - [modelPath]: Path to the ONNX GenAI model directory.
  /// - [prompt]: Text prompt for generation.
  /// - [imagePath]: Optional path to image file (null for text-only).
  ///
  /// Returns the generated text, or throws [OnnxGenAIException] on error.
  String runInference({
    required String modelPath,
    required String prompt,
    String? imagePath,
  }) {
    final modelPathPtr = modelPath.toNativeUtf8();
    final promptPtr = prompt.toNativeUtf8();
    final imagePathPtr = imagePath != null ? imagePath.toNativeUtf8() : nullptr;

    try {
      final resultPtr = _runInference(modelPathPtr, promptPtr, imagePathPtr);
      final result = resultPtr.toDartString();

      // Check for error prefix
      if (result.startsWith('ERROR:')) {
        throw OnnxGenAIException(result.substring(6).trim());
      }

      return result;
    } finally {
      calloc.free(modelPathPtr);
      calloc.free(promptPtr);
      if (imagePath != null) {
        calloc.free(imagePathPtr);
      }
    }
  }

  /// Runs multimodal inference with text and multiple images.
  ///
  /// WARNING: This is a LONG-RUNNING, BLOCKING operation!
  /// DO NOT call from the main UI isolate. Use [runInferenceMultiAsync] instead.
  ///
  /// Parameters:
  /// - [modelPath]: Path to the ONNX GenAI model directory.
  /// - [prompt]: Text prompt for generation (should contain <|image_N|> placeholders).
  /// - [imagePaths]: List of paths to image files.
  ///
  /// Returns the generated text, or throws [OnnxGenAIException] on error.
  String runInferenceMulti({
    required String modelPath,
    required String prompt,
    required List<String> imagePaths,
  }) {
    final modelPathPtr = modelPath.toNativeUtf8();
    final promptPtr = prompt.toNativeUtf8();

    // Allocate array of pointers for image paths
    final imagePathPtrs = calloc<Pointer<Utf8>>(imagePaths.length);
    for (int i = 0; i < imagePaths.length; i++) {
      imagePathPtrs[i] = imagePaths[i].toNativeUtf8();
    }

    try {
      final resultPtr = _runInferenceMulti(
        modelPathPtr,
        promptPtr,
        imagePathPtrs,
        imagePaths.length,
      );
      final result = resultPtr.toDartString();

      // Check for error prefix
      if (result.startsWith('ERROR:')) {
        throw OnnxGenAIException(result.substring(6).trim());
      }

      return result;
    } finally {
      calloc.free(modelPathPtr);
      calloc.free(promptPtr);
      // Free each image path string
      for (int i = 0; i < imagePaths.length; i++) {
        calloc.free(imagePathPtrs[i]);
      }
      // Free the array of pointers
      calloc.free(imagePathPtrs);
    }
  }

  /// Runs text-only inference.
  ///
  /// WARNING: This is a LONG-RUNNING, BLOCKING operation!
  /// DO NOT call from the main UI isolate.
  ///
  /// Parameters:
  /// - [modelPath]: Path to the ONNX GenAI model directory.
  /// - [prompt]: Text prompt for generation.
  /// - [maxLength]: Maximum tokens to generate (0 for model default).
  ///
  /// Returns the generated text, or throws [OnnxGenAIException] on error.
  String runTextInference({
    required String modelPath,
    required String prompt,
    int maxLength = 0,
  }) {
    final modelPathPtr = modelPath.toNativeUtf8();
    final promptPtr = prompt.toNativeUtf8();

    try {
      final resultPtr = _runTextInference(modelPathPtr, promptPtr, maxLength);
      final result = resultPtr.toDartString();

      // Check for error prefix
      if (result.startsWith('ERROR:')) {
        throw OnnxGenAIException(result.substring(6).trim());
      }

      return result;
    } finally {
      calloc.free(modelPathPtr);
      calloc.free(promptPtr);
    }
  }

  /// Shuts down the ONNX GenAI library and releases resources.
  void shutdown() {
    _shutdownOnnxGenAI();
    _workerIsolate?.kill();
    _workerIsolate = null;
  }

  // ===========================================================================
  // Public API - Asynchronous (safe for main isolate)
  // ===========================================================================

  /// Runs multimodal inference asynchronously in a background isolate.
  ///
  /// This is the recommended method for calling from the main UI isolate.
  ///
  /// Parameters:
  /// - [modelPath]: Path to the ONNX GenAI model directory.
  /// - [prompt]: Text prompt for generation.
  /// - [imagePath]: Optional path to image file (null for text-only).
  ///
  /// Returns a [Future] that completes with the generated text.
  Future<String> runInferenceAsync({
    required String modelPath,
    required String prompt,
    String? imagePath,
  }) async {
    return Isolate.run(() {
      final onnx = OnnxGenAI();
      return onnx.runInference(
        modelPath: modelPath,
        prompt: prompt,
        imagePath: imagePath,
      );
    });
  }

  /// Runs multimodal inference with multiple images asynchronously.
  ///
  /// This is the recommended method for calling from the main UI isolate.
  ///
  /// Parameters:
  /// - [modelPath]: Path to the ONNX GenAI model directory.
  /// - [prompt]: Text prompt for generation (should contain <|image_N|> placeholders).
  /// - [imagePaths]: List of paths to image files.
  ///
  /// Returns a [Future] that completes with the generated text.
  Future<String> runInferenceMultiAsync({
    required String modelPath,
    required String prompt,
    required List<String> imagePaths,
  }) async {
    return Isolate.run(() {
      final onnx = OnnxGenAI();
      return onnx.runInferenceMulti(
        modelPath: modelPath,
        prompt: prompt,
        imagePaths: imagePaths,
      );
    });
  }

  /// Runs text-only inference asynchronously in a background isolate.
  ///
  /// This is the recommended method for calling from the main UI isolate.
  Future<String> runTextInferenceAsync({
    required String modelPath,
    required String prompt,
    int maxLength = 0,
  }) async {
    return Isolate.run(() {
      final onnx = OnnxGenAI();
      return onnx.runTextInference(
        modelPath: modelPath,
        prompt: prompt,
        maxLength: maxLength,
      );
    });
  }

  /// Checks model health asynchronously.
  Future<int> checkNativeHealthAsync(String modelPath) async {
    return Isolate.run(() {
      final onnx = OnnxGenAI();
      return onnx.checkNativeHealth(modelPath);
    });
  }
}

// =============================================================================
// Streaming Inference Support
// =============================================================================

/// Provides streaming inference capabilities.
///
/// This class allows receiving generated tokens one-by-one as they are
/// produced, rather than waiting for the complete response.
///
/// Example:
/// ```dart
/// final streamer = OnnxGenAIStreamer();
///
/// await for (final token in streamer.streamInference(
///   modelPath: '/path/to/model',
///   prompt: 'Tell me a story.',
/// )) {
///   print(token); // Print each token as it's generated
/// }
/// ```
class OnnxGenAIStreamer {
  /// Streams inference results token-by-token.
  ///
  /// NOTE: The current native implementation does not support true streaming.
  /// This method runs inference in a background isolate and streams the
  /// complete result in chunks to simulate streaming behavior.
  ///
  /// For true token-by-token streaming, a callback-based native implementation
  /// would be needed.
  Stream<String> streamInference({
    required String modelPath,
    required String prompt,
    String? imagePath,
    int chunkSize = 1,
  }) async* {
    // Run inference in background
    final result = await Isolate.run(() {
      final onnx = OnnxGenAI();
      return onnx.runInference(
        modelPath: modelPath,
        prompt: prompt,
        imagePath: imagePath,
      );
    });

    // Stream the result character by character (simulated streaming)
    final chars = result.split('');
    final buffer = StringBuffer();

    for (var i = 0; i < chars.length; i++) {
      buffer.write(chars[i]);

      if (buffer.length >= chunkSize) {
        yield buffer.toString();
        buffer.clear();
      }
    }

    // Yield any remaining characters
    if (buffer.isNotEmpty) {
      yield buffer.toString();
    }
  }

  /// Streams text inference results.
  Stream<String> streamTextInference({
    required String modelPath,
    required String prompt,
    int maxLength = 0,
    int chunkSize = 1,
  }) async* {
    final result = await Isolate.run(() {
      final onnx = OnnxGenAI();
      return onnx.runTextInference(
        modelPath: modelPath,
        prompt: prompt,
        maxLength: maxLength,
      );
    });

    // Stream the result
    final chars = result.split('');
    final buffer = StringBuffer();

    for (var i = 0; i < chars.length; i++) {
      buffer.write(chars[i]);

      if (buffer.length >= chunkSize) {
        yield buffer.toString();
        buffer.clear();
      }
    }

    if (buffer.isNotEmpty) {
      yield buffer.toString();
    }
  }
}

// =============================================================================
// Exceptions
// =============================================================================

/// Exception thrown by ONNX GenAI operations.
class OnnxGenAIException implements Exception {
  /// The error message.
  final String message;

  /// Creates an [OnnxGenAIException] with the given [message].
  const OnnxGenAIException(this.message);

  @override
  String toString() => 'OnnxGenAIException: $message';
}
