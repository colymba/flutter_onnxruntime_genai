# Flutter ONNX Runtime GenAI

A Flutter FFI plugin that wraps the Microsoft [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai) C-API for on-device multimodal inference.

> [!CAUTION]
> **⚠️ UNSTABLE & EXPERIMENTAL ⚠️**
> 
> This plugin is in an **early experimental stage** and is **NOT production-ready**. Use at your own risk.
> 
> - APIs may change significantly without notice
> - Memory management issues may cause crashes on resource-constrained devices
> - The KV-cache memory allocation can exceed device RAM with large `max_length` values
> - Not all ONNX GenAI features are exposed or tested
> - Limited testing has been done on real devices
>
> **Known Limitations:**
> - Models with large context windows (e.g., 128K tokens) require configuration changes to avoid OOM crashes
> - The `max_length` parameter in `genai_config.json` must be reduced for mobile devices (recommended: 2048 or less)
> - Text-only inference with vision models requires processing through the multimodal pipeline

## ✅ Tested Models

| Model | Device | Status | Notes |
|-------|--------|--------|-------|
| [Gemma 3 4B Instruct](https://huggingface.co/onnxruntime/Gemma-3-ONNX/tree/main/gemma-3-4b-it/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4) | Pixel 8a | ✅ Working | INT4 quantized, ~1240 tokens generated successfully |

> If you've tested other models, please open a PR to add them to this list!

## 🤝 Contributing & Help Wanted

We are looking for help to develop and test this package! Whether you have experience with Dart FFI, ONNX Runtime, or native mobile development (Swift/Kotlin/C++), your contributions are highly welcome.

If you'd like to help, please:
- Open an issue to discuss proposed changes.
- Submit pull requests for bug fixes or new features.
- Help with testing on different devices and documenting the results.

## Features

- **Vision-Language Models**: Run models like Phi-3.5 Vision on-device
- **Cross-Platform**: Android and iOS support
- **FFI Performance**: Direct native library access via Dart FFI
- **Async Inference**: Background isolate support for non-blocking UI
- **Streaming**: Token-by-token output streaming

## Requirements

- Flutter 3.3.0+
- Dart SDK 3.10.4+
- Android: API 24+ (NDK for building), **arm64 device only** (x86_64 emulator not included in pub.dev package)
- iOS: 13.0+ (Xcode for building), **device only** (simulator not included in pub.dev package)

> **Note**: The pub.dev package only includes device libraries (Android arm64, iOS arm64) to stay under size limits. For simulator/emulator testing, build from source using the instructions below.

## Installation

### 1. Add Dependency

```yaml
dependencies:
  flutter_onnxruntime_genai: ^0.1.6
```

Or from Git for the latest development version:

```yaml
dependencies:
  flutter_onnxruntime_genai:
    git:
      url: https://github.com/colymba/flutter_onnxruntime_genai.git
```

### 2. Set Up ONNX Runtime GenAI Submodule (Optional - for building from source)

```bash
cd flutter_onnxruntime_genai
git submodule add https://github.com/microsoft/onnxruntime-genai.git native_src/onnxruntime-genai
git submodule update --init --recursive
```

### 3. Build Native Libraries (Optional)

Prebuilt binaries for Android (arm64, x86_64) and iOS (Device & Simulator) are included in the package. You only need to build from source if you want to customize the build or use a different version:

```bash
./scripts/build_onnx_libs.sh all
```

## Usage

### Basic Inference

```dart
import 'package:flutter_onnxruntime_genai/flutter_onnxruntime_genai.dart';

// Create instance
final onnx = OnnxGenAI();

// Check library version
print('Library version: ${onnx.libraryVersion}');

// Verify model health
final status = await onnx.checkNativeHealthAsync('/path/to/model');
if (status == HealthStatus.success) {
  print('Model is ready!');
} else {
  print('Error: ${HealthStatus.getMessage(status)}');
}

// Run multimodal inference (text + image)
try {
  final result = await onnx.runInferenceAsync(
    modelPath: '/path/to/phi-3.5-vision',
    prompt: 'Describe this image in detail.',
    imagePath: '/path/to/image.jpg',
  );
  print('Generated: $result');
} on OnnxGenAIException catch (e) {
  print('Inference error: $e');
}
```

### Text-Only Inference

```dart
final result = await onnx.runTextInferenceAsync(
  modelPath: '/path/to/model',
  prompt: 'What is the capital of France?',
  maxLength: 256,
);
```

### Streaming Output

```dart
final streamer = OnnxGenAIStreamer();

await for (final token in streamer.streamInference(
  modelPath: '/path/to/model',
  prompt: 'Tell me a story about a brave knight.',
)) {
  stdout.write(token); // Print tokens as they're generated
}
```

## Architecture

```
flutter_onnxruntime_genai/
├── native_src/
│   └── onnxruntime-genai/      # Git submodule
├── src/
│   ├── include/
│   │   └── ort_genai_c.h       # C-API header (stub)
│   ├── flutter_onnxruntime_genai.cpp
│   ├── flutter_onnxruntime_genai.h
│   └── CMakeLists.txt
├── android/
│   ├── build.gradle
│   └── src/main/jniLibs/       # Prebuilt .so files
├── ios/
│   ├── flutter_onnxruntime_genai.podspec
│   └── Frameworks/             # XCFramework
├── lib/
│   ├── flutter_onnxruntime_genai.dart
│   └── src/
│       └── ffi_bindings.dart   # Dart FFI layer
└── scripts/
    └── build_onnx_libs.sh      # Cross-compilation script
```

## API Reference

### OnnxGenAI

Main class for ONNX Runtime GenAI operations.

| Method | Description |
|--------|-------------|
| `libraryVersion` | Get the native library version |
| `checkNativeHealth(path)` | Verify model can be loaded |
| `runInference(...)` | Multimodal inference (blocking) |
| `runTextInference(...)` | Text-only inference (blocking) |
| `runInferenceAsync(...)` | Multimodal inference (background isolate) |
| `runTextInferenceAsync(...)` | Text-only inference (background isolate) |
| `shutdown()` | Release native resources |

### HealthStatus

Status codes from `checkNativeHealth`:

| Code | Constant | Meaning |
|------|----------|---------|
| 1 | `success` | Model loaded successfully |
| -1 | `invalidPath` | NULL or empty path |
| -2 | `modelCreationFailed` | Model creation failed |
| -3 | `tokenizerCreationFailed` | Tokenizer creation failed |

## Important Notes

### Threading

All synchronous inference methods (`runInference`, `runTextInference`) are **blocking** operations that can take seconds to minutes depending on the model and prompt.

**DO NOT** call these from the main UI isolate. Use the async variants which automatically run in a background isolate.

### Android 15 Compatibility

The build script and CMake configuration include the critical 16KB page alignment flag (`-Wl,-z,max-page-size=16384`) for Android 15+ compatibility.

### Debug Logging

The native C++ code includes comprehensive step-by-step debug logging for troubleshooting. Logs are output to:
- **Android**: Logcat with tag `OnnxGenAI` (use `adb logcat -s OnnxGenAI:*`)
- **iOS/Desktop**: stderr

To **disable** debug logging in production builds, edit `src/flutter_onnxruntime_genai.cpp` and set:

```cpp
#define ONNX_DEBUG_LOG 0
```

### Model Files

ONNX GenAI models must be present on the device before inference. The model directory should contain:
- `config.json`
- Model weights (`.onnx` files)
- Tokenizer files

## Building from Source

### Prerequisites

- Android NDK (set `ANDROID_NDK_HOME`)
- Xcode command line tools
- CMake 3.20+
- Python 3.8+

### Build Commands

```bash
# Build for Android only
./scripts/build_onnx_libs.sh android

# Build for iOS only
./scripts/build_onnx_libs.sh ios

# Build for both platforms
./scripts/build_onnx_libs.sh all

# Clean build artifacts
./scripts/build_onnx_libs.sh clean
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

ONNX Runtime GenAI is licensed under the MIT License by Microsoft.
