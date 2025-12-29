# ONNX Runtime GenAI Git Submodule

This directory is reserved for the ONNX Runtime GenAI git submodule.

## Setup

To add the submodule, run:

```bash
git submodule add https://github.com/microsoft/onnxruntime-genai.git native_src/onnxruntime-genai
git submodule update --init --recursive
```

## Build

After setting up the submodule, use the build script:

```bash
./scripts/build_onnx_libs.sh
```

This will cross-compile the ONNX Runtime GenAI libraries for:
- Android: arm64-v8a, x86_64 (with 16KB page alignment for Android 15+)
- iOS: Device (arm64) and Simulator (arm64, x86_64) as XCFramework
