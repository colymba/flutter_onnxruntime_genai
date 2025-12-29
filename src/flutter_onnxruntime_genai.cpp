/**
 * @file flutter_onnxruntime_genai.cpp
 * @brief Flutter FFI Bridge for ONNX Runtime GenAI
 *
 * This file implements the C FFI layer that bridges Dart code with the
 * ONNX Runtime GenAI C-API for multimodal inference (Phi-3.5 Vision).
 *
 * IMPORTANT: All inference functions in this file are long-running operations.
 * They MUST be called from a background Dart Isolate, NOT the main UI isolate.
 * Calling these from the main isolate will block the UI and cause dropped
 * frames.
 */

#include "flutter_onnxruntime_genai.h"
#include "include/ort_genai_c.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

// =============================================================================
// Thread-safe result buffer management
// =============================================================================

namespace {
// Thread-local storage for result strings
// This avoids memory management issues when returning strings to Dart
thread_local std::string g_result_buffer;
thread_local std::string g_error_buffer;

// Mutex for thread-safe operations
std::mutex g_init_mutex;

// Track initialization state
bool g_initialized = false;
} // namespace

/**
 * @brief Safely copy a string to the result buffer and return a pointer.
 *
 * The returned pointer is valid until the next call to this function
 * from the same thread. Dart must copy the string before the next FFI call.
 */
static const char *set_result(const std::string &result) {
  g_result_buffer = result;
  return g_result_buffer.c_str();
}

/**
 * @brief Set an error message and return it.
 */
static const char *set_error(const std::string &error) {
  g_error_buffer = "ERROR: " + error;
  return g_error_buffer.c_str();
}

/**
 * @brief Handle OgaResult and return error message if present.
 * @return true if there was an error, false otherwise
 */
static bool check_oga_result(OgaResult *result, const char *context) {
  if (result != nullptr) {
    const char *error_msg = OgaResultGetError(result);
    if (error_msg != nullptr) {
      g_error_buffer = std::string(context) + ": " + error_msg;
      OgaDestroyResult(result);
      return true;
    }
    OgaDestroyResult(result);
  }
  return false;
}

// =============================================================================
// FFI Exported Functions
// =============================================================================

extern "C" {

/**
 * @brief Check if the native library and model can be loaded.
 *
 * Use this function to verify that:
 * 1. The native library is properly linked
 * 2. The model path is accessible
 * 3. The model can be loaded successfully
 *
 * @param model_path Path to the ONNX GenAI model directory
 * @return 0 on success, negative error codes on failure:
 *         -1: NULL path provided
 *         -2: Model creation failed
 *         -3: Tokenizer creation failed
 *         1: Model loaded and verified successfully
 */
FFI_PLUGIN_EXPORT int32_t check_native_health(const char *model_path) {
  if (model_path == nullptr || strlen(model_path) == 0) {
    return -1; // NULL or empty path
  }

  OgaModel *model = nullptr;
  OgaResult *result = OgaCreateModel(model_path, &model);

  if (check_oga_result(result, "Model creation failed") || model == nullptr) {
    return -2; // Model creation failed
  }

  // Try to create a tokenizer to verify model integrity
  OgaTokenizer *tokenizer = nullptr;
  result = OgaCreateTokenizer(model, &tokenizer);

  if (check_oga_result(result, "Tokenizer creation failed") ||
      tokenizer == nullptr) {
    OgaDestroyModel(model);
    return -3; // Tokenizer creation failed
  }

  // Cleanup
  OgaDestroyTokenizer(tokenizer);
  OgaDestroyModel(model);

  return 1; // Success
}

/**
 * @brief Run text-only inference with the model.
 *
 * WARNING: This is a long-running operation! Call from a background Isolate
 * only.
 *
 * @param model_path Path to the ONNX GenAI model directory
 * @param prompt The text prompt for generation
 * @param max_length Maximum number of tokens to generate (0 for default)
 * @return Generated text on success, error message starting with "ERROR:" on
 * failure
 */
FFI_PLUGIN_EXPORT const char *run_text_inference(const char *model_path,
                                                 const char *prompt,
                                                 int32_t max_length) {
  if (model_path == nullptr || prompt == nullptr) {
    return set_error("NULL model_path or prompt provided");
  }

  // Create model
  OgaModel *model = nullptr;
  OgaResult *result = OgaCreateModel(model_path, &model);
  if (check_oga_result(result, "Model creation failed") || model == nullptr) {
    return set_error(g_error_buffer);
  }

  // Create tokenizer
  OgaTokenizer *tokenizer = nullptr;
  result = OgaCreateTokenizer(model, &tokenizer);
  if (check_oga_result(result, "Tokenizer creation failed") ||
      tokenizer == nullptr) {
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }

  // Encode the prompt
  OgaSequences *input_sequences = nullptr;
  result = OgaTokenizerEncode(tokenizer, prompt, &input_sequences);
  if (check_oga_result(result, "Tokenization failed") ||
      input_sequences == nullptr) {
    OgaDestroyTokenizer(tokenizer);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }

  // Create generator parameters
  OgaGeneratorParams *params = nullptr;
  result = OgaCreateGeneratorParams(model, &params);
  if (check_oga_result(result, "Generator params creation failed") ||
      params == nullptr) {
    OgaDestroySequences(input_sequences);
    OgaDestroyTokenizer(tokenizer);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }

  // Set max length if specified
  if (max_length > 0) {
    OgaGeneratorParamsSetSearchNumber(params, "max_length",
                                      static_cast<double>(max_length));
  }

  // Set input sequences
  result = OgaGeneratorParamsSetInputSequences(params, input_sequences);
  if (check_oga_result(result, "Setting input sequences failed")) {
    OgaDestroyGeneratorParams(params);
    OgaDestroySequences(input_sequences);
    OgaDestroyTokenizer(tokenizer);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }

  // Create generator
  OgaGenerator *generator = nullptr;
  result = OgaCreateGenerator(model, params, &generator);
  if (check_oga_result(result, "Generator creation failed") ||
      generator == nullptr) {
    OgaDestroyGeneratorParams(params);
    OgaDestroySequences(input_sequences);
    OgaDestroyTokenizer(tokenizer);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }

  // Generate tokens
  std::string generated_text;
  OgaTokenizerStream *stream = nullptr;
  result = OgaCreateTokenizerStream(tokenizer, &stream);
  if (check_oga_result(result, "Tokenizer stream creation failed") ||
      stream == nullptr) {
    OgaDestroyGenerator(generator);
    OgaDestroyGeneratorParams(params);
    OgaDestroySequences(input_sequences);
    OgaDestroyTokenizer(tokenizer);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }

  while (!OgaGenerator_IsDone(generator)) {
    result = OgaGenerator_ComputeLogits(generator);
    if (check_oga_result(result, "Compute logits failed")) {
      break;
    }

    result = OgaGenerator_GenerateNextToken(generator);
    if (check_oga_result(result, "Generate next token failed")) {
      break;
    }

    // Get the last generated token
    int32_t token = OgaGenerator_GetLastToken(generator, 0);

    // Decode token to text
    const char *token_text = nullptr;
    result = OgaTokenizerStreamDecode(stream, token, &token_text);
    if (!check_oga_result(result, "Token decode failed") &&
        token_text != nullptr) {
      generated_text += token_text;
    }
  }

  // Cleanup
  OgaDestroyTokenizerStream(stream);
  OgaDestroyGenerator(generator);
  OgaDestroyGeneratorParams(params);
  OgaDestroySequences(input_sequences);
  OgaDestroyTokenizer(tokenizer);
  OgaDestroyModel(model);

  return set_result(generated_text);
}

/**
 * @brief Run multimodal inference with text and image.
 *
 * This function is specifically designed for vision-language models like
 * Phi-3.5 Vision. It processes both the text prompt and image together for
 * generation.
 *
 * WARNING: This is a long-running operation! Call from a background Isolate
 * only.
 *
 * @param model_path Path to the ONNX GenAI model directory
 * @param prompt The text prompt for generation
 * @param image_path Path to the image file (JPEG, PNG, etc.)
 * @return Generated text on success, error message starting with "ERROR:" on
 * failure
 */
FFI_PLUGIN_EXPORT const char *run_inference(const char *model_path,
                                            const char *prompt,
                                            const char *image_path) {
  if (model_path == nullptr || prompt == nullptr) {
    return set_error("NULL model_path or prompt provided");
  }

  // Create model
  OgaModel *model = nullptr;
  OgaResult *result = OgaCreateModel(model_path, &model);
  if (check_oga_result(result, "Model creation failed") || model == nullptr) {
    return set_error(g_error_buffer);
  }

  // Create multimodal processor
  OgaMultiModalProcessor *processor = nullptr;
  result = OgaCreateMultiModalProcessor(model, &processor);
  if (check_oga_result(result, "MultiModal processor creation failed") ||
      processor == nullptr) {
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }

  // Create tokenizer from processor
  OgaTokenizer *tokenizer = nullptr;
  result = OgaMultiModalProcessorCreateTokenizer(processor, &tokenizer);
  if (check_oga_result(result, "Tokenizer creation failed") ||
      tokenizer == nullptr) {
    OgaDestroyMultiModalProcessor(processor);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }

  // Load image if provided
  OgaImages *images = nullptr;
  OgaNamedTensors *named_tensors = nullptr;

  if (image_path != nullptr && strlen(image_path) > 0) {
    result = OgaLoadImage(image_path, &images);
    if (check_oga_result(result, "Image loading failed") || images == nullptr) {
      OgaDestroyTokenizer(tokenizer);
      OgaDestroyMultiModalProcessor(processor);
      OgaDestroyModel(model);
      return set_error(g_error_buffer);
    }

    // Process images with prompt
    result = OgaMultiModalProcessorProcessImages(processor, prompt, images,
                                                 &named_tensors);
    if (check_oga_result(result, "Image processing failed") ||
        named_tensors == nullptr) {
      OgaDestroyImages(images);
      OgaDestroyTokenizer(tokenizer);
      OgaDestroyMultiModalProcessor(processor);
      OgaDestroyModel(model);
      return set_error(g_error_buffer);
    }
  }

  // Create generator parameters
  OgaGeneratorParams *params = nullptr;
  result = OgaCreateGeneratorParams(model, &params);
  if (check_oga_result(result, "Generator params creation failed") ||
      params == nullptr) {
    if (named_tensors)
      OgaDestroyNamedTensors(named_tensors);
    if (images)
      OgaDestroyImages(images);
    OgaDestroyTokenizer(tokenizer);
    OgaDestroyMultiModalProcessor(processor);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }

  // Set input tensors if we have image data
  if (named_tensors != nullptr) {
    result = OgaGeneratorParamsSetInputs(params, named_tensors);
    if (check_oga_result(result, "Setting input tensors failed")) {
      OgaDestroyGeneratorParams(params);
      OgaDestroyNamedTensors(named_tensors);
      OgaDestroyImages(images);
      OgaDestroyTokenizer(tokenizer);
      OgaDestroyMultiModalProcessor(processor);
      OgaDestroyModel(model);
      return set_error(g_error_buffer);
    }
  } else {
    // Text-only mode: encode the prompt
    OgaSequences *input_sequences = nullptr;
    result = OgaTokenizerEncode(tokenizer, prompt, &input_sequences);
    if (check_oga_result(result, "Tokenization failed") ||
        input_sequences == nullptr) {
      OgaDestroyGeneratorParams(params);
      OgaDestroyTokenizer(tokenizer);
      OgaDestroyMultiModalProcessor(processor);
      OgaDestroyModel(model);
      return set_error(g_error_buffer);
    }

    result = OgaGeneratorParamsSetInputSequences(params, input_sequences);
    OgaDestroySequences(input_sequences);
    if (check_oga_result(result, "Setting input sequences failed")) {
      OgaDestroyGeneratorParams(params);
      OgaDestroyTokenizer(tokenizer);
      OgaDestroyMultiModalProcessor(processor);
      OgaDestroyModel(model);
      return set_error(g_error_buffer);
    }
  }

  // Create generator
  OgaGenerator *generator = nullptr;
  result = OgaCreateGenerator(model, params, &generator);
  if (check_oga_result(result, "Generator creation failed") ||
      generator == nullptr) {
    OgaDestroyGeneratorParams(params);
    if (named_tensors)
      OgaDestroyNamedTensors(named_tensors);
    if (images)
      OgaDestroyImages(images);
    OgaDestroyTokenizer(tokenizer);
    OgaDestroyMultiModalProcessor(processor);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }

  // Create tokenizer stream for incremental decoding
  OgaTokenizerStream *stream = nullptr;
  result = OgaCreateTokenizerStream(tokenizer, &stream);
  if (check_oga_result(result, "Tokenizer stream creation failed") ||
      stream == nullptr) {
    OgaDestroyGenerator(generator);
    OgaDestroyGeneratorParams(params);
    if (named_tensors)
      OgaDestroyNamedTensors(named_tensors);
    if (images)
      OgaDestroyImages(images);
    OgaDestroyTokenizer(tokenizer);
    OgaDestroyMultiModalProcessor(processor);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }

  // Generate tokens
  std::string generated_text;
  while (!OgaGenerator_IsDone(generator)) {
    result = OgaGenerator_ComputeLogits(generator);
    if (check_oga_result(result, "Compute logits failed")) {
      break;
    }

    result = OgaGenerator_GenerateNextToken(generator);
    if (check_oga_result(result, "Generate next token failed")) {
      break;
    }

    // Get the last generated token
    int32_t token = OgaGenerator_GetLastToken(generator, 0);

    // Decode token to text
    const char *token_text = nullptr;
    result = OgaTokenizerStreamDecode(stream, token, &token_text);
    if (!check_oga_result(result, "Token decode failed") &&
        token_text != nullptr) {
      generated_text += token_text;
    }
  }

  // Cleanup
  OgaDestroyTokenizerStream(stream);
  OgaDestroyGenerator(generator);
  OgaDestroyGeneratorParams(params);
  if (named_tensors)
    OgaDestroyNamedTensors(named_tensors);
  if (images)
    OgaDestroyImages(images);
  OgaDestroyTokenizer(tokenizer);
  OgaDestroyMultiModalProcessor(processor);
  OgaDestroyModel(model);

  return set_result(generated_text);
}

/**
 * @brief Free any global resources held by the library.
 *
 * Call this when the application is shutting down or when the plugin
 * is being unloaded to ensure proper cleanup.
 */
FFI_PLUGIN_EXPORT void shutdown_onnx_genai() {
  std::lock_guard<std::mutex> lock(g_init_mutex);
  if (g_initialized) {
    OgaShutdown();
    g_initialized = false;
  }
}

/**
 * @brief Get the library version string.
 *
 * @return Version string in format "major.minor.patch"
 */
FFI_PLUGIN_EXPORT const char *get_library_version() { return "0.1.0"; }

} // extern "C"
