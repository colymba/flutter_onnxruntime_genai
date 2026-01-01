# Performance Benchmarking

This document tracks inference performance benchmarks for various models, devices, and configurations.

## Test Methodology

- **Prompt**: Simplified test prompt (~100 tokens) for consistent comparison
- **Metric**: Tokens per second (tok/s) during generation
- **Timing**: Measured using `OnnxGenAI.debugTiming = true`

## Device Specifications

| Device | SoC | CPU | RAM | Notes |
|--------|-----|-----|-----|-------|
| Pixel 8a | Google Tensor G3 | 9 cores (1x Cortex-X3, 4x Cortex-A715, 4x Cortex-A510) | 8GB | |

---

## Gemma 3 4B Instruct (INT4)

**Model**: [onnxruntime/Gemma-3-ONNX](https://huggingface.co/onnxruntime/Gemma-3-ONNX/tree/main/gemma-3-4b-it/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4)  
**Quantization**: INT4 RTN Block-32  
**Size**: ~2.5GB

### Pixel 8a Results

| Test | Provider | Config | Tokens | Time (s) | tok/s | Notes |
|------|----------|--------|--------|----------|-------|-------|
| A | Default | Factory (128K ctx) | 1971 | 802.3 | 2.46 | Baseline |
| B | XNNPACK | Factory (128K ctx) | 1971 | 919.8 | 2.14 | ⚠️ 13% slower! |
| C | Default | +4 threads | 1971 | 435.0 | 4.53 | ✅ +84% faster! |
| D | XNNPACK | +4 threads | 1971 | 429.2 | 4.59 | 🏆 **+87% BEST** |
| E | Default | +8 threads | 1971 | 741.5 | 2.66 | ⚠️ Thread contention |
| F | XNNPACK | +8 threads | 1971 | 484.2 | 4.07 | +65% (XNNPACK helps!) |

### Key Findings

1. **Optimal config: XNNPACK + 4 threads** = 4.59 tok/s (+87% vs baseline)
2. **Threading matters more than provider**: 4 threads gives +84% alone
3. **8 threads causes contention** on Tensor G3's big.LITTLE architecture
4. **XNNPACK alone hurts performance** (-13%) but helps when combined with threading
5. **XNNPACK mitigates 8-thread contention**: 4.07 vs 2.66 tok/s

### Configuration Details

**Factory Config** (`genai_config.json` defaults):
```json
{
  "model": {
    "context_length": 131072,
    "decoder": {
      "session_options": {
        "log_id": "onnxruntime-genai",
        "provider_options": []
      }
    }
  }
}
```

**+4 threads** (via `optimizeForMobile()`):
```json
{
  "model.decoder.session_options.intra_op_num_threads": 4,
  "model.decoder.session_options.inter_op_num_threads": 1
}
```

**+8 threads** (via `optimizeAggressive()`):
```json
{
  "model.decoder.session_options.intra_op_num_threads": 8,
  "model.decoder.session_options.inter_op_num_threads": 1
}
```

---

## Execution Providers

| Provider | Platform | Description |
|----------|----------|-------------|
| Default | All | ONNX Runtime default CPU execution |
| XNNPACK | Android/iOS | Optimized ARM NEON kernels |
| QNN | Android | Qualcomm Hexagon NPU (Snapdragon only) |
| CoreML | iOS | Apple Neural Engine |

---

## Key Findings

*(To be updated after testing)*

- [ ] Does XNNPACK improve performance for large models?
- [ ] What's the optimal thread count for Tensor G3?
- [ ] Do session_options like `graph_optimization_level` help or hurt?

---

## How to Run Benchmarks

1. Set `OnnxGenAI.debugTiming = true` in your code
2. Run inference and check the timing output:
   ```
   [OnnxGenAI] Inference Timing:
     Step 1: Create config ................ 1.2 ms
     Step 2: Run inference ................ 45000.0 ms
     ...
     Total: 45005.3 ms (45.01 seconds)
   ```
3. Calculate: `tokens / (inference_time_ms / 1000) = tok/s`

---

## Contributing

Tested on a different device or model? Please submit a PR with your results!
