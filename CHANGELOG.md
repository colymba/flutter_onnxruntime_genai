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
