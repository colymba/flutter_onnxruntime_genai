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
