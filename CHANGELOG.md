## 0.4.1

* **Enhanced `optimizeForMobile()`** - Now includes high-priority ONNX Runtime optimizations:
  * ARM64 bfloat16 GEMM fast math (`mlas.enable_gemm_fastmath_arm64_bfloat16`)
  * Denormal-as-zero for faster float ops (`session.set_denormal_as_zero`)
  * Thread spinning disabled for power efficiency (`session.intra_op.allow_spinning`)
* Performance improvement: **5.22 tok/s** on Pixel 8a (+113% vs factory, +13% vs v0.4.0)

## 0.4.0

* **New: Performance Benchmarking** - Added comprehensive benchmarks for Gemma 3 4B on Pixel 8a.
* **New: `OnnxGenAIConfig.optimizeForMobile()`** - Convenience method to configure thread counts for optimal mobile performance.
* Added `hasFactoryBackup()` method to check if factory config backup exists.
* Added `restoreFactoryConfig()` method to restore original `genai_config.json`.
* Added `BENCHMARKING.md` documenting performance results across different configurations.

### Performance Recommendations (Pixel 8a / Tensor G3)

| Config | tok/s | Improvement |
|--------|-------|-------------|
| XNNPACK + 4 threads | 4.59 | **+87%** 🏆 |
| Default + 4 threads | 4.53 | +84% |
| Factory defaults | 2.46 | baseline |

> **Key insight**: Use 4 intra-op threads on big.LITTLE SoCs. 8+ threads causes contention on efficiency cores.

## 0.3.0

* **New: Runtime Configuration API** - Configure execution providers from Dart at runtime!
* Added `createConfig()` / `destroyConfig()` for managing configuration objects.
* Added `configClearProviders()` / `configAppendProvider()` for customizing execution providers.
* Added `configSetProviderOption()` for setting provider-specific options.
* Added `runInferenceWithConfig()` for synchronous inference with custom config.
* Added `runInferenceWithConfigAsync()` - **recommended** async method with full configuration support.
* Added `runInferenceMultiWithConfig()` for synchronous multi-image inference with custom config.
* Added `runInferenceMultiWithConfigAsync()` - **recommended** async method for multi-image with config.
* Added `getLastError()` for detailed error messages from native layer.
* Added `OnnxGenAIConfig` utility class for modifying `genai_config.json` (session options, max_length, etc.)

### Example Usage

```dart
// Use XNNPACK for optimized ARM inference on mobile
final result = await onnx.runInferenceWithConfigAsync(
  modelPath: '/path/to/model',
  prompt: 'Hello!',
  providers: ['XNNPACK'],  // Use XNNPACK execution provider
);
```

### Supported Execution Providers

| Provider | Platform | Description |
|----------|----------|-------------|
| `XNNPACK` | Android/iOS | Optimized ARM NEON kernels |
| `QNN` | Android | Qualcomm Hexagon NPU (Snapdragon) |
| `CoreML` | iOS/macOS | Apple Neural Engine |
| `SNPE` | Android | Snapdragon Neural Processing Engine |
| `OpenVINO` | Desktop | Intel optimized inference |

> **Note**: `cpu` is NOT a valid provider. CPU execution is the default fallback.

## 0.2.0

* **First confirmed working release!** Successfully tested on-device inference on Google Pixel 8a.
* Tested with [Gemma 3 4B Instruct](https://huggingface.co/onnxruntime/Gemma-3-ONNX/tree/main/gemma-3-4b-it/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4) (INT4 quantized) - generated 1240 tokens successfully.
* Added "Tested Models" section to README documenting verified model/device combinations.

## 0.1.6

* Fixed multimodal text-only inference: Always process through `OgaProcessorProcessImages` even without images (required for vision models like Phi-3.5).
* Added KV-cache memory management: Set `max_length` to 2048 tokens to prevent OOM crashes on mobile devices.
* Added ONNX GenAI internal logging callback for better debugging.
* Added signal handlers (SIGSEGV, SIGABRT, etc.) for crash debugging.
* Enhanced debug logging with granular step tracking around critical API calls.
* Fixed crash during `OgaGenerator_SetInputs` caused by prompt size exceeding `max_length`.

## 0.1.5

* Internal testing release.

## 0.1.4

* Added comprehensive debug logging to trace native C++ execution step-by-step.
* Debug logs use Android Logcat (`__android_log_print`) on Android and `stderr` on other platforms.
* Added multi-image inference support via `run_inference_multi` and `runInferenceMultiAsync`.
* Debug logging can be disabled by setting `ONNX_DEBUG_LOG` to `0` in `flutter_onnxruntime_genai.cpp`.

## 0.1.3

* Fixed Android runtime crash: Added missing `libonnxruntime.so` dependency to jniLibs.
* Updated build script to automatically copy ONNX Runtime library alongside GenAI library.

## 0.1.2

* Fixed C++ API compatibility with ONNX Runtime GenAI C header.
* Updated `OgaTokenizerEncode` to use pre-created sequences.
* Replaced deprecated `OgaGeneratorParamsSetInputSequences` with `OgaGenerator_AppendTokenSequences`.
* Replaced non-existent `OgaGenerator_ComputeLogits` - using `OgaGenerator_GenerateNextToken` directly.
* Replaced `OgaGenerator_GetLastToken` with `OgaGenerator_GetNextTokens`.
* Fixed `OgaProcessorProcessImages` function name.
* Fixed `OgaGenerator_SetInputs` to be called on generator instead of params.

## 0.1.1

* Include prebuilt stripped native libraries for Android and iOS.
* Reduced package size for pub.dev compatibility.
* Updated documentation.

## 0.1.0

* Initial experimental release.
* Support for ONNX Runtime GenAI C-API.
* Multimodal inference support (Text + Image) for models like Phi-3.5 Vision.
* Support for Android (with 16KB page alignment) and iOS.
* Async induction via background isolates.
* Token-by-token streaming output.
