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
import 'dart:convert';

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
// Configuration API Native Function Types
// =============================================================================

/// Native function: int64_t create_config(const char* model_path)
typedef CreateConfigNative = Int64 Function(Pointer<Utf8> modelPath);
typedef CreateConfigDart = int Function(Pointer<Utf8> modelPath);

/// Native function: void destroy_config(int64_t config_handle)
typedef DestroyConfigNative = Void Function(Int64 configHandle);
typedef DestroyConfigDart = void Function(int configHandle);

/// Native function: int32_t config_clear_providers(int64_t config_handle)
typedef ConfigClearProvidersNative = Int32 Function(Int64 configHandle);
typedef ConfigClearProvidersDart = int Function(int configHandle);

/// Native function: int32_t config_append_provider(int64_t config_handle, const char* provider_name)
typedef ConfigAppendProviderNative =
    Int32 Function(Int64 configHandle, Pointer<Utf8> providerName);
typedef ConfigAppendProviderDart =
    int Function(int configHandle, Pointer<Utf8> providerName);

/// Native function: int32_t config_set_provider_option(int64_t config_handle, const char* provider_name, const char* key, const char* value)
typedef ConfigSetProviderOptionNative =
    Int32 Function(
      Int64 configHandle,
      Pointer<Utf8> providerName,
      Pointer<Utf8> key,
      Pointer<Utf8> value,
    );
typedef ConfigSetProviderOptionDart =
    int Function(
      int configHandle,
      Pointer<Utf8> providerName,
      Pointer<Utf8> key,
      Pointer<Utf8> value,
    );

/// Native function: const char* run_inference_with_config(int64_t config_handle, const char* prompt, const char* image_path)
typedef RunInferenceWithConfigNative =
    Pointer<Utf8> Function(
      Int64 configHandle,
      Pointer<Utf8> prompt,
      Pointer<Utf8> imagePath,
    );
typedef RunInferenceWithConfigDart =
    Pointer<Utf8> Function(
      int configHandle,
      Pointer<Utf8> prompt,
      Pointer<Utf8> imagePath,
    );

/// Native function: const char* run_inference_multi_with_config(int64_t config_handle, const char* prompt, const char** image_paths, int32_t image_count)
typedef RunInferenceMultiWithConfigNative =
    Pointer<Utf8> Function(
      Int64 configHandle,
      Pointer<Utf8> prompt,
      Pointer<Pointer<Utf8>> imagePaths,
      Int32 imageCount,
    );
typedef RunInferenceMultiWithConfigDart =
    Pointer<Utf8> Function(
      int configHandle,
      Pointer<Utf8> prompt,
      Pointer<Pointer<Utf8>> imagePaths,
      int imageCount,
    );

/// Native function: const char* get_last_error()
typedef GetLastErrorNative = Pointer<Utf8> Function();
typedef GetLastErrorDart = Pointer<Utf8> Function();

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
// Debug Timing Helper
// =============================================================================

/// Helper class for timing inference steps when debug mode is enabled.
///
/// Tracks elapsed time for each step and provides a formatted summary.
///
/// Example output:
/// ```
/// [OnnxGenAI] Inference Timing:
///   Step 1: Create config ................ 12.3 ms
///   Step 2: Clear providers .............. 0.1 ms
///   Step 3: Add provider XNNPACK ......... 0.2 ms
///   Step 4: Run inference ................ 8542.7 ms
///   Step 5: Destroy config ............... 0.1 ms
///   ─────────────────────────────────────────────
///   Total: 8555.4 ms (8.56 seconds)
/// ```
class InferenceTimer {
  final bool enabled;
  final List<_TimingEntry> _entries = [];
  final Stopwatch _totalStopwatch = Stopwatch();
  Stopwatch? _stepStopwatch;
  int _stepCount = 0;

  /// Creates a timer. Pass [enabled] = false to create a no-op timer.
  InferenceTimer({this.enabled = true}) {
    if (enabled) {
      _totalStopwatch.start();
    }
  }

  /// Starts timing a new step with the given description.
  void startStep(String description) {
    if (!enabled) return;
    _stepStopwatch = Stopwatch()..start();
    _stepCount++;
  }

  /// Ends the current step and records its duration.
  void endStep(String description) {
    if (!enabled || _stepStopwatch == null) return;
    _stepStopwatch!.stop();
    _entries.add(
      _TimingEntry(
        step: _stepCount,
        description: description,
        duration: _stepStopwatch!.elapsed,
      ),
    );
    _stepStopwatch = null;
  }

  /// Times a synchronous operation and records it as a step.
  T time<T>(String description, T Function() operation) {
    if (!enabled) return operation();
    startStep(description);
    try {
      return operation();
    } finally {
      endStep(description);
    }
  }

  /// Stops the timer and prints the summary to console.
  void stop() {
    if (!enabled) return;
    _totalStopwatch.stop();
    _printSummary();
  }

  /// Gets the total elapsed time.
  Duration get totalElapsed => _totalStopwatch.elapsed;

  /// Gets all timing entries.
  List<({int step, String description, Duration duration})> get entries =>
      _entries
          .map(
            (e) => (
              step: e.step,
              description: e.description,
              duration: e.duration,
            ),
          )
          .toList();

  void _printSummary() {
    final buffer = StringBuffer();
    buffer.writeln('[OnnxGenAI] Inference Timing:');

    for (final entry in _entries) {
      final stepStr = 'Step ${entry.step}:';
      final descPadded = entry.description.padRight(30, '.');
      final ms = entry.duration.inMicroseconds / 1000;
      final msStr = ms.toStringAsFixed(1).padLeft(10);
      buffer.writeln('  $stepStr $descPadded $msStr ms');
    }

    buffer.writeln('  ${'─' * 45}');
    final totalMs = _totalStopwatch.elapsedMicroseconds / 1000;
    final totalSec = totalMs / 1000;
    buffer.writeln(
      '  Total: ${totalMs.toStringAsFixed(1)} ms (${totalSec.toStringAsFixed(2)} seconds)',
    );

    // ignore: avoid_print
    print(buffer.toString());
  }
}

class _TimingEntry {
  final int step;
  final String description;
  final Duration duration;

  _TimingEntry({
    required this.step,
    required this.description,
    required this.duration,
  });
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
/// // Enable debug timing
/// OnnxGenAI.debugTiming = true;
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
/// // When debug=true, timing info is printed to console
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

  /// Enable debug mode to print timing information for inference steps.
  ///
  /// When enabled, each async inference method will print detailed timing
  /// for each step (config creation, provider setup, inference, cleanup).
  ///
  /// Example:
  /// ```dart
  /// OnnxGenAI.debugTiming = true;
  /// final result = await onnx.runInferenceWithConfigAsync(...);
  /// // Prints:
  /// // [OnnxGenAI] Inference Timing:
  /// //   Step 1: Create config ................ 12.3 ms
  /// //   Step 2: Clear providers .............. 0.1 ms
  /// //   ...
  /// ```
  static bool debugTiming = false;

  /// The loaded dynamic library.
  late final DynamicLibrary _dylib;

  // Bound native functions
  late final CheckNativeHealthDart _checkNativeHealth;
  late final RunInferenceDart _runInference;
  late final RunInferenceMultiDart _runInferenceMulti;
  late final RunTextInferenceDart _runTextInference;
  late final GetLibraryVersionDart _getLibraryVersion;
  late final ShutdownOnnxGenAIDart _shutdownOnnxGenAI;

  // Configuration API functions
  late final CreateConfigDart _createConfig;
  late final DestroyConfigDart _destroyConfig;
  late final ConfigClearProvidersDart _configClearProviders;
  late final ConfigAppendProviderDart _configAppendProvider;
  late final ConfigSetProviderOptionDart _configSetProviderOption;
  late final RunInferenceWithConfigDart _runInferenceWithConfig;
  late final RunInferenceMultiWithConfigDart _runInferenceMultiWithConfig;
  late final GetLastErrorDart _getLastError;

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

    // Configuration API bindings
    _createConfig = _dylib
        .lookup<NativeFunction<CreateConfigNative>>('create_config')
        .asFunction<CreateConfigDart>();

    _destroyConfig = _dylib
        .lookup<NativeFunction<DestroyConfigNative>>('destroy_config')
        .asFunction<DestroyConfigDart>();

    _configClearProviders = _dylib
        .lookup<NativeFunction<ConfigClearProvidersNative>>(
          'config_clear_providers',
        )
        .asFunction<ConfigClearProvidersDart>();

    _configAppendProvider = _dylib
        .lookup<NativeFunction<ConfigAppendProviderNative>>(
          'config_append_provider',
        )
        .asFunction<ConfigAppendProviderDart>();

    _configSetProviderOption = _dylib
        .lookup<NativeFunction<ConfigSetProviderOptionNative>>(
          'config_set_provider_option',
        )
        .asFunction<ConfigSetProviderOptionDart>();

    _runInferenceWithConfig = _dylib
        .lookup<NativeFunction<RunInferenceWithConfigNative>>(
          'run_inference_with_config',
        )
        .asFunction<RunInferenceWithConfigDart>();

    _runInferenceMultiWithConfig = _dylib
        .lookup<NativeFunction<RunInferenceMultiWithConfigNative>>(
          'run_inference_multi_with_config',
        )
        .asFunction<RunInferenceMultiWithConfigDart>();

    _getLastError = _dylib
        .lookup<NativeFunction<GetLastErrorNative>>('get_last_error')
        .asFunction<GetLastErrorDart>();
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

  /// Gets the last error message from the native library.
  String getLastError() {
    final ptr = _getLastError();
    return ptr.toDartString();
  }

  // ===========================================================================
  // Configuration API - Runtime Session Options
  // ===========================================================================

  /// Creates a configuration object for customizing execution providers.
  ///
  /// Use this to configure execution providers and session options before
  /// loading the model. Returns a config handle that must be destroyed with
  /// [destroyConfig] when no longer needed.
  ///
  /// Example:
  /// ```dart
  /// final configHandle = onnx.createConfig('/path/to/model');
  /// onnx.configClearProviders(configHandle);
  /// onnx.configAppendProvider(configHandle, 'XNNPACK');
  /// onnx.configSetProviderOption(configHandle, 'cpu', 'intra_op_num_threads', '4');
  /// final result = onnx.runInferenceWithConfig(configHandle, 'Hello', null);
  /// onnx.destroyConfig(configHandle);
  /// ```
  ///
  /// Returns a config handle (> 0 on success, 0 on failure).
  int createConfig(String modelPath) {
    final modelPathPtr = modelPath.toNativeUtf8();
    try {
      return _createConfig(modelPathPtr);
    } finally {
      calloc.free(modelPathPtr);
    }
  }

  /// Destroys a configuration object and frees its resources.
  void destroyConfig(int configHandle) {
    _destroyConfig(configHandle);
  }

  /// Clears all execution providers from the config.
  ///
  /// Call this before adding custom providers to start fresh.
  /// Returns 1 on success, negative value on failure.
  int configClearProviders(int configHandle) {
    return _configClearProviders(configHandle);
  }

  /// Appends an execution provider to the config.
  ///
  /// Providers are tried in order of insertion. Common providers:
  /// - `"cpu"`: Default CPU execution
  /// - `"XNNPACK"`: Optimized ARM CPU kernels (recommended for mobile)
  /// - `"QNN"`: Qualcomm NPU (Snapdragon only)
  /// - `"NNAPI"`: Android Neural Networks API
  /// - `"CoreML"`: Apple Neural Engine (iOS only)
  ///
  /// Returns 1 on success, negative value on failure.
  int configAppendProvider(int configHandle, String providerName) {
    final providerPtr = providerName.toNativeUtf8();
    try {
      return _configAppendProvider(configHandle, providerPtr);
    } finally {
      calloc.free(providerPtr);
    }
  }

  /// Sets an option for a specific execution provider.
  ///
  /// Common options for CPU provider:
  /// - `"intra_op_num_threads"`: Threads within an op (e.g., "4")
  /// - `"inter_op_num_threads"`: Threads between ops (e.g., "1")
  ///
  /// Returns 1 on success, negative value on failure.
  int configSetProviderOption(
    int configHandle,
    String providerName,
    String key,
    String value,
  ) {
    final providerPtr = providerName.toNativeUtf8();
    final keyPtr = key.toNativeUtf8();
    final valuePtr = value.toNativeUtf8();
    try {
      return _configSetProviderOption(
        configHandle,
        providerPtr,
        keyPtr,
        valuePtr,
      );
    } finally {
      calloc.free(providerPtr);
      calloc.free(keyPtr);
      calloc.free(valuePtr);
    }
  }

  /// Runs inference using a pre-configured config.
  ///
  /// WARNING: This is a LONG-RUNNING, BLOCKING operation!
  /// DO NOT call from the main UI isolate. Use [runInferenceWithConfigAsync] instead.
  ///
  /// Parameters:
  /// - [configHandle]: Handle returned by [createConfig]
  /// - [prompt]: Text prompt for generation
  /// - [imagePath]: Optional path to image file (null for text-only)
  ///
  /// Returns the generated text, or throws [OnnxGenAIException] on error.
  String runInferenceWithConfig({
    required int configHandle,
    required String prompt,
    String? imagePath,
  }) {
    final promptPtr = prompt.toNativeUtf8();
    final imagePathPtr = imagePath != null ? imagePath.toNativeUtf8() : nullptr;

    try {
      final resultPtr = _runInferenceWithConfig(
        configHandle,
        promptPtr,
        imagePathPtr,
      );
      final result = resultPtr.toDartString();

      if (result.startsWith('ERROR:')) {
        throw OnnxGenAIException(result.substring(6).trim());
      }

      return result;
    } finally {
      calloc.free(promptPtr);
      if (imagePath != null) {
        calloc.free(imagePathPtr);
      }
    }
  }

  /// Runs multi-image inference using a pre-configured config.
  ///
  /// WARNING: This is a LONG-RUNNING, BLOCKING operation!
  /// DO NOT call from the main UI isolate. Use [runInferenceMultiWithConfigAsync] instead.
  ///
  /// Parameters:
  /// - [configHandle]: Handle returned by [createConfig]
  /// - [prompt]: Text prompt for generation (with <|image_N|> placeholders)
  /// - [imagePaths]: List of paths to image files
  ///
  /// Returns the generated text, or throws [OnnxGenAIException] on error.
  String runInferenceMultiWithConfig({
    required int configHandle,
    required String prompt,
    required List<String> imagePaths,
  }) {
    final promptPtr = prompt.toNativeUtf8();

    // Allocate array of pointers for image paths
    final imagePathsPtr = calloc<Pointer<Utf8>>(imagePaths.length);
    for (var i = 0; i < imagePaths.length; i++) {
      imagePathsPtr[i] = imagePaths[i].toNativeUtf8();
    }

    try {
      final resultPtr = _runInferenceMultiWithConfig(
        configHandle,
        promptPtr,
        imagePathsPtr,
        imagePaths.length,
      );
      final result = resultPtr.toDartString();

      if (result.startsWith('ERROR:')) {
        throw OnnxGenAIException(result.substring(6).trim());
      }

      return result;
    } finally {
      calloc.free(promptPtr);
      for (var i = 0; i < imagePaths.length; i++) {
        calloc.free(imagePathsPtr[i]);
      }
      calloc.free(imagePathsPtr);
    }
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

  /// Runs inference with custom execution provider configuration asynchronously.
  ///
  /// This is the recommended way to use custom configs from the main UI isolate.
  /// Creates the config, applies settings, runs inference, and cleans up - all
  /// in a background isolate.
  ///
  /// Parameters:
  /// - [modelPath]: Path to the ONNX GenAI model directory.
  /// - [prompt]: Text prompt for generation.
  /// - [imagePath]: Optional path to image file (null for text-only).
  /// - [providers]: List of execution providers in priority order.
  ///   Common values: `['XNNPACK', 'cpu']`, `['cpu']`, `['CoreML', 'cpu']`
  /// - [providerOptions]: Map of provider-specific options.
  ///   Example: `{'cpu': {'intra_op_num_threads': '4'}}`
  ///
  /// Returns a [Future] that completes with the generated text.
  ///
  /// Example:
  /// ```dart
  /// final result = await onnx.runInferenceWithConfigAsync(
  ///   modelPath: '/path/to/model',
  ///   prompt: 'Hello, how are you?',
  ///   providers: ['XNNPACK', 'cpu'],
  ///   providerOptions: {
  ///     'cpu': {
  ///       'intra_op_num_threads': '4',
  ///       'inter_op_num_threads': '1',
  ///     },
  ///   },
  /// );
  /// ```
  Future<String> runInferenceWithConfigAsync({
    required String modelPath,
    required String prompt,
    String? imagePath,
    List<String>? providers,
    Map<String, Map<String, String>>? providerOptions,
  }) async {
    // Capture debug flag before entering isolate (static vars aren't shared)
    final debugEnabled = OnnxGenAI.debugTiming;

    return Isolate.run(() {
      final onnx = OnnxGenAI();
      final timer = InferenceTimer(enabled: debugEnabled);

      // Create config
      // Note: configHandle is a pointer cast to int64, which may appear negative
      // if the MSB is set. Only check for 0 (null pointer).
      final configHandle = timer.time('Create config', () {
        return onnx.createConfig(modelPath);
      });
      if (configHandle == 0) {
        timer.stop();
        throw OnnxGenAIException(
          'Failed to create config: ${onnx.getLastError()}',
        );
      }

      try {
        // Configure providers
        if (providers != null && providers.isNotEmpty) {
          timer.time('Clear providers', () {
            onnx.configClearProviders(configHandle);
          });
          for (final provider in providers) {
            final result = timer.time('Add provider $provider', () {
              return onnx.configAppendProvider(configHandle, provider);
            });
            if (result < 0) {
              timer.stop();
              throw OnnxGenAIException(
                'Failed to add provider "$provider": ${onnx.getLastError()}',
              );
            }
          }
        }

        // Configure provider options
        if (providerOptions != null) {
          for (final entry in providerOptions.entries) {
            final providerName = entry.key;
            final options = entry.value;
            for (final option in options.entries) {
              final result = timer.time('Set $providerName.${option.key}', () {
                return onnx.configSetProviderOption(
                  configHandle,
                  providerName,
                  option.key,
                  option.value,
                );
              });
              if (result < 0) {
                timer.stop();
                throw OnnxGenAIException(
                  'Failed to set option "${option.key}" for "$providerName": ${onnx.getLastError()}',
                );
              }
            }
          }
        }

        // Run inference
        final result = timer.time('Run inference', () {
          return onnx.runInferenceWithConfig(
            configHandle: configHandle,
            prompt: prompt,
            imagePath: imagePath,
          );
        });

        return result;
      } finally {
        timer.time('Destroy config', () {
          onnx.destroyConfig(configHandle);
        });
        timer.stop();
      }
    });
  }

  /// Runs multi-image inference with custom execution provider configuration asynchronously.
  ///
  /// This is the recommended way to use custom configs from the main UI isolate
  /// when working with multiple images.
  ///
  /// Parameters:
  /// - [modelPath]: Path to the ONNX GenAI model directory.
  /// - [prompt]: Text prompt for generation (with <|image_N|> placeholders).
  /// - [imagePaths]: List of paths to image files.
  /// - [providers]: List of execution providers in priority order.
  /// - [providerOptions]: Map of provider-specific options.
  ///
  /// Returns a [Future] that completes with the generated text.
  ///
  /// Example:
  /// ```dart
  /// final result = await onnx.runInferenceMultiWithConfigAsync(
  ///   modelPath: '/path/to/model',
  ///   prompt: '<|image_1|><|image_2|>\nCompare these two images.',
  ///   imagePaths: ['/path/to/image1.jpg', '/path/to/image2.jpg'],
  ///   providers: ['XNNPACK', 'cpu'],
  ///   providerOptions: {
  ///     'cpu': {
  ///       'intra_op_num_threads': '4',
  ///       'inter_op_num_threads': '1',
  ///     },
  ///   },
  /// );
  /// ```
  Future<String> runInferenceMultiWithConfigAsync({
    required String modelPath,
    required String prompt,
    required List<String> imagePaths,
    List<String>? providers,
    Map<String, Map<String, String>>? providerOptions,
  }) async {
    // Capture debug flag before entering isolate (static vars aren't shared)
    final debugEnabled = OnnxGenAI.debugTiming;

    return Isolate.run(() {
      final onnx = OnnxGenAI();
      final timer = InferenceTimer(enabled: debugEnabled);

      // Create config
      // Note: configHandle is a pointer cast to int64, which may appear negative
      // if the MSB is set. Only check for 0 (null pointer).
      final configHandle = timer.time('Create config', () {
        return onnx.createConfig(modelPath);
      });
      if (configHandle == 0) {
        timer.stop();
        throw OnnxGenAIException(
          'Failed to create config: ${onnx.getLastError()}',
        );
      }

      try {
        // Configure providers
        if (providers != null && providers.isNotEmpty) {
          timer.time('Clear providers', () {
            onnx.configClearProviders(configHandle);
          });
          for (final provider in providers) {
            final result = timer.time('Add provider $provider', () {
              return onnx.configAppendProvider(configHandle, provider);
            });
            if (result < 0) {
              timer.stop();
              throw OnnxGenAIException(
                'Failed to add provider "$provider": ${onnx.getLastError()}',
              );
            }
          }
        }

        // Configure provider options
        if (providerOptions != null) {
          for (final entry in providerOptions.entries) {
            final providerName = entry.key;
            final options = entry.value;
            for (final option in options.entries) {
              final result = timer.time('Set $providerName.${option.key}', () {
                return onnx.configSetProviderOption(
                  configHandle,
                  providerName,
                  option.key,
                  option.value,
                );
              });
              if (result < 0) {
                timer.stop();
                throw OnnxGenAIException(
                  'Failed to set option "${option.key}" for "$providerName": ${onnx.getLastError()}',
                );
              }
            }
          }
        }

        // Run inference with multiple images
        final result = timer.time('Run inference (multi)', () {
          return onnx.runInferenceMultiWithConfig(
            configHandle: configHandle,
            prompt: prompt,
            imagePaths: imagePaths,
          );
        });

        return result;
      } finally {
        timer.time('Destroy config', () {
          onnx.destroyConfig(configHandle);
        });
        timer.stop();
      }
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

// =============================================================================
// Model Configuration Utility
// =============================================================================

/// Utility class for reading and modifying genai_config.json files.
///
/// Since ONNX Runtime GenAI's runtime API only supports execution provider
/// configuration, session options like thread counts and memory settings
/// must be configured by modifying the genai_config.json file directly.
///
/// Example:
/// ```dart
/// final configUtil = OnnxGenAIConfig(modelPath);
///
/// // Optimize for mobile
/// await configUtil.optimizeForMobile(
///   maxLength: 2048,
///   contextLength: 4096,
///   intraOpThreads: 4,
/// );
///
/// // Or update individual settings
/// await configUtil.update({
///   'search.max_length': 2048,
///   'model.decoder.session_options.intra_op_num_threads': 4,
/// });
/// ```
class OnnxGenAIConfig {
  /// Path to the model directory containing genai_config.json.
  final String modelPath;

  /// Creates a config utility for the given model path.
  OnnxGenAIConfig(this.modelPath);

  /// Path to the genai_config.json file.
  String get configFilePath => '$modelPath/genai_config.json';

  /// Path to the backup config file (factory settings).
  String get backupFilePath => '$modelPath/genai_config.json.back';

  /// Checks if a factory backup exists.
  Future<bool> hasFactoryBackup() async {
    return File(backupFilePath).exists();
  }

  /// Creates a backup of the current config as factory settings.
  ///
  /// Only creates the backup if it doesn't already exist, preserving
  /// the original "factory" configuration.
  ///
  /// Returns true if backup was created, false if it already existed.
  Future<bool> backupFactoryConfig() async {
    final backupFile = File(backupFilePath);
    if (await backupFile.exists()) {
      return false; // Backup already exists, don't overwrite
    }

    final configFile = File(configFilePath);
    if (!await configFile.exists()) {
      throw OnnxGenAIException('Config file not found: $configFilePath');
    }

    await configFile.copy(backupFilePath);
    return true;
  }

  /// Restores the factory configuration from backup.
  ///
  /// Throws [OnnxGenAIException] if no backup exists.
  ///
  /// Returns the restored configuration.
  Future<Map<String, dynamic>> restoreFactoryConfig() async {
    final backupFile = File(backupFilePath);
    if (!await backupFile.exists()) {
      throw OnnxGenAIException('No factory backup found at: $backupFilePath');
    }

    // Copy backup back to main config
    await backupFile.copy(configFilePath);

    // Return the restored config
    return read();
  }

  /// Reads and returns the current configuration as a Map.
  Future<Map<String, dynamic>> read() async {
    final file = File(configFilePath);
    if (!await file.exists()) {
      throw OnnxGenAIException('Config file not found: $configFilePath');
    }
    final content = await file.readAsString();
    return Map<String, dynamic>.from(
      const JsonDecoder().convert(content) as Map,
    );
  }

  /// Writes the configuration map to the file.
  ///
  /// Automatically creates a factory backup before the first write
  /// if one doesn't already exist.
  Future<void> write(Map<String, dynamic> config) async {
    // Backup factory config before first modification
    await backupFactoryConfig();

    final file = File(configFilePath);
    final encoder = const JsonEncoder.withIndent('    ');
    await file.writeAsString(encoder.convert(config));
  }

  /// Updates specific configuration values using dot-notation keys.
  ///
  /// Example keys:
  /// - `'search.max_length'` → sets `config['search']['max_length']`
  /// - `'model.context_length'` → sets `config['model']['context_length']`
  /// - `'model.decoder.session_options.intra_op_num_threads'`
  ///
  /// Returns the updated configuration.
  Future<Map<String, dynamic>> update(Map<String, dynamic> updates) async {
    final config = await read();

    for (final entry in updates.entries) {
      _setNestedValue(config, entry.key.split('.'), entry.value);
    }

    await write(config);
    return config;
  }

  /// Gets a specific value using dot-notation key.
  Future<dynamic> get(String key) async {
    final config = await read();
    return _getNestedValue(config, key.split('.'));
  }

  /// Applies mobile-optimized settings to reduce memory usage and improve speed.
  ///
  /// Parameters:
  /// - [maxLength]: Maximum tokens to generate (default: 2048)
  /// - [contextLength]: Maximum context window (default: 4096)
  /// - [intraOpThreads]: Threads within an operation (default: 4)
  /// - [interOpThreads]: Threads between operations (default: 1)
  /// - [pastPresentShareBuffer]: Reuse KV-cache buffers (default: true)
  /// - [doSample]: Use sampling vs greedy decoding (default: false for speed)
  ///
  /// Returns the updated configuration.
  Future<Map<String, dynamic>> optimizeForMobile({
    int maxLength = 2048,
    int contextLength = 4096,
    int intraOpThreads = 4,
    int interOpThreads = 1,
    bool pastPresentShareBuffer = true,
    bool doSample = false,
  }) async {
    return update({
      'model.context_length': contextLength,
      // Thread counts are numbers, other session_options are strings
      'model.decoder.session_options.intra_op_num_threads': intraOpThreads,
      'model.decoder.session_options.inter_op_num_threads': interOpThreads,
      'search.max_length': maxLength,
      'search.past_present_share_buffer': pastPresentShareBuffer,
      'search.do_sample': doSample,
    });
  }

  /// Applies aggressive performance optimizations for maximum speed.
  ///
  /// This configuration pushes harder than [optimizeForMobile] by:
  /// - Using more threads (Pixel 8a has 9 cores)
  /// - Enabling all graph optimizations (ORT_ENABLE_ALL)
  /// - Enabling memory pattern and CPU memory arena optimizations
  /// - Using greedy decoding (no sampling overhead)
  /// - Smaller context window to reduce memory pressure
  /// - Enabling KV-cache buffer sharing
  ///
  /// **Warning:** These settings prioritize speed over quality. The model
  /// may produce slightly different outputs compared to default settings.
  ///
  /// Recommended for devices:
  /// - Pixel 8/8a/8 Pro (Tensor G3, 9 cores) → intraOpThreads: 8
  /// - Pixel 7/7a/7 Pro (Tensor G2, 8 cores) → intraOpThreads: 7
  /// - Samsung Galaxy S24 (Snapdragon 8 Gen 3, 8 cores) → intraOpThreads: 7
  ///
  /// Parameters:
  /// - [maxLength]: Maximum tokens to generate (default: 1024, shorter = faster)
  /// - [contextLength]: Maximum context window (default: 2048)
  /// - [intraOpThreads]: Threads within an operation (default: 8 for big.LITTLE)
  /// - [interOpThreads]: Threads between operations (default: 1)
  ///
  /// Returns the updated configuration.
  Future<Map<String, dynamic>> optimizeAggressive({
    int maxLength = 1024,
    int contextLength = 2048,
    int intraOpThreads = 8,
    int interOpThreads = 1,
  }) async {
    return update({
      // Model/context settings (integer)
      'model.context_length': contextLength,
      // Session options - thread counts are integers
      'model.decoder.session_options.intra_op_num_threads': intraOpThreads,
      'model.decoder.session_options.inter_op_num_threads': interOpThreads,
      // Graph optimization: must be one of: ORT_DISABLE_ALL, ORT_ENABLE_BASIC,
      // ORT_ENABLE_EXTENDED, ORT_ENABLE_ALL (string!)
      'model.decoder.session_options.graph_optimization_level':
          'ORT_ENABLE_ALL',
      // Memory optimizations (booleans, NOT strings!)
      'model.decoder.session_options.enable_mem_pattern': true,
      'model.decoder.session_options.enable_cpu_mem_arena': true,
      // Search/generation options - greedy for speed
      // Integers:
      'search.max_length': maxLength,
      'search.min_length': 1,
      'search.num_beams': 1,
      'search.top_k': 1,
      // Floats:
      'search.temperature': 1.0,
      'search.repetition_penalty': 1.0,
      // Booleans:
      'search.past_present_share_buffer': true,
      'search.do_sample': false,
      'search.early_stopping': true,
    });
  }

  /// Prints the current configuration to the console for debugging.
  ///
  /// Useful for verifying which settings are applied.
  Future<void> printConfig() async {
    final config = await read();
    final encoder = const JsonEncoder.withIndent('  ');
    // ignore: avoid_print
    print(
      '[OnnxGenAIConfig] Current configuration:\n${encoder.convert(config)}',
    );
  }

  /// Resets configuration to hardcoded default values.
  ///
  /// Use [restoreFactoryConfig] instead to restore the original model settings.
  ///
  /// @deprecated Prefer [restoreFactoryConfig] which restores the actual
  /// original configuration from backup.
  Future<Map<String, dynamic>> resetToDefaults({
    int maxLength = 131072,
    int contextLength = 131072,
  }) async {
    return update({
      'model.context_length': contextLength,
      'search.max_length': maxLength,
      'search.past_present_share_buffer': false,
      'search.do_sample': true,
    });
  }

  /// Sets a nested value in a map using a path of keys.
  void _setNestedValue(
    Map<String, dynamic> map,
    List<String> path,
    dynamic value,
  ) {
    if (path.isEmpty) return;

    if (path.length == 1) {
      map[path.first] = value;
      return;
    }

    final key = path.first;
    if (!map.containsKey(key) || map[key] is! Map) {
      map[key] = <String, dynamic>{};
    }

    _setNestedValue(map[key] as Map<String, dynamic>, path.sublist(1), value);
  }

  /// Gets a nested value from a map using a path of keys.
  dynamic _getNestedValue(Map<String, dynamic> map, List<String> path) {
    if (path.isEmpty) return null;

    final key = path.first;
    if (!map.containsKey(key)) return null;

    if (path.length == 1) {
      return map[key];
    }

    if (map[key] is! Map) return null;

    return _getNestedValue(map[key] as Map<String, dynamic>, path.sublist(1));
  }
}
