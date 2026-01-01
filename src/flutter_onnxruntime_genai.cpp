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
#include <cstdio>
#include <signal.h>

#ifdef __ANDROID__
#include <unistd.h>
#endif

// =============================================================================
// Debug logging macros
// =============================================================================

// Set to 1 to enable debug logging, 0 to disable
#define ONNX_DEBUG_LOG 1

#if ONNX_DEBUG_LOG
  #ifdef __ANDROID__
    #include <android/log.h>
    #define DEBUG_TAG "OnnxGenAI"
    #define DEBUG_LOG(fmt, ...) __android_log_print(ANDROID_LOG_DEBUG, DEBUG_TAG, "[DEBUG] " fmt, ##__VA_ARGS__)
    #define DEBUG_ERROR(fmt, ...) __android_log_print(ANDROID_LOG_ERROR, DEBUG_TAG, "[ERROR] " fmt, ##__VA_ARGS__)
  #else
    #define DEBUG_LOG(fmt, ...) fprintf(stderr, "[OnnxGenAI DEBUG] " fmt "\n", ##__VA_ARGS__)
    #define DEBUG_ERROR(fmt, ...) fprintf(stderr, "[OnnxGenAI ERROR] " fmt "\n", ##__VA_ARGS__)
  #endif
#else
  #define DEBUG_LOG(fmt, ...) ((void)0)
  #define DEBUG_ERROR(fmt, ...) ((void)0)
#endif

// =============================================================================
// ONNX GenAI internal logging callback
// =============================================================================

#if ONNX_DEBUG_LOG
static void oga_log_callback(const char* message, size_t length) {
  (void)length;
  #ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_INFO, "OnnxGenAI-Internal", "%s", message);
  #else
    fprintf(stderr, "[OnnxGenAI-Internal] %s\n", message);
  #endif
}
#endif

// =============================================================================
// Signal handler for crash debugging
// =============================================================================

#if ONNX_DEBUG_LOG
static void crash_signal_handler(int sig) {
  const char* sig_name = "UNKNOWN";
  switch (sig) {
    case SIGSEGV: sig_name = "SIGSEGV"; break;
    case SIGABRT: sig_name = "SIGABRT"; break;
    case SIGFPE: sig_name = "SIGFPE"; break;
    case SIGILL: sig_name = "SIGILL"; break;
    case SIGBUS: sig_name = "SIGBUS"; break;
  }
  #ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_ERROR, "OnnxGenAI", 
                        "[CRASH] Caught signal %d (%s)", sig, sig_name);
  #else
    fprintf(stderr, "[OnnxGenAI CRASH] Caught signal %d (%s)\n", sig, sig_name);
  #endif
  // Re-raise signal for default handling (to generate crash report)
  signal(sig, SIG_DFL);
  raise(sig);
}

static void install_signal_handlers() {
  signal(SIGSEGV, crash_signal_handler);
  signal(SIGABRT, crash_signal_handler);
  signal(SIGFPE, crash_signal_handler);
  signal(SIGILL, crash_signal_handler);
  signal(SIGBUS, crash_signal_handler);
  DEBUG_LOG("Signal handlers installed for crash debugging");
}
#endif

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

// Track if logging/signal handlers are set up
bool g_debug_initialized = false;
} // namespace

// Initialize debug features (logging + signal handlers)
static void init_debug_features() {
#if ONNX_DEBUG_LOG
  if (!g_debug_initialized) {
    g_debug_initialized = true;
    install_signal_handlers();
    // Enable ONNX GenAI internal logging
    OgaResult* result = OgaSetLogBool("enabled", true);
    if (result != nullptr) {
      DEBUG_ERROR("Failed to enable OGA logging: %s", OgaResultGetError(result));
      OgaDestroyResult(result);
    } else {
      DEBUG_LOG("OGA internal logging enabled");
    }
    result = OgaSetLogCallback(oga_log_callback);
    if (result != nullptr) {
      DEBUG_ERROR("Failed to set OGA log callback: %s", OgaResultGetError(result));
      OgaDestroyResult(result);
    } else {
      DEBUG_LOG("OGA log callback set");
    }
  }
#endif
}

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
  init_debug_features();
  DEBUG_LOG("=== check_native_health START ===");
  DEBUG_LOG("model_path: %s", model_path ? model_path : "NULL");

  if (model_path == nullptr || strlen(model_path) == 0) {
    DEBUG_ERROR("NULL or empty model_path provided");
    return -1; // NULL or empty path
  }

  DEBUG_LOG("Step 1: Creating model...");
  OgaModel *model = nullptr;
  OgaResult *result = OgaCreateModel(model_path, &model);

  if (check_oga_result(result, "Model creation failed") || model == nullptr) {
    DEBUG_ERROR("Model creation failed");
    return -2; // Model creation failed
  }
  DEBUG_LOG("Step 1: Model created successfully");

  // Try to create a tokenizer to verify model integrity
  DEBUG_LOG("Step 2: Creating tokenizer...");
  OgaTokenizer *tokenizer = nullptr;
  result = OgaCreateTokenizer(model, &tokenizer);

  if (check_oga_result(result, "Tokenizer creation failed") ||
      tokenizer == nullptr) {
    DEBUG_ERROR("Tokenizer creation failed");
    OgaDestroyModel(model);
    return -3; // Tokenizer creation failed
  }
  DEBUG_LOG("Step 2: Tokenizer created successfully");

  // Cleanup
  DEBUG_LOG("Step 3: Cleaning up...");
  OgaDestroyTokenizer(tokenizer);
  OgaDestroyModel(model);
  DEBUG_LOG("Step 3: Cleanup complete");
  DEBUG_LOG("=== check_native_health END (success) ===");

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
  DEBUG_LOG("=== run_text_inference START ===");
  DEBUG_LOG("model_path: %s", model_path ? model_path : "NULL");
  DEBUG_LOG("prompt length: %zu", prompt ? strlen(prompt) : 0);
  DEBUG_LOG("max_length: %d", max_length);

  if (model_path == nullptr || prompt == nullptr) {
    DEBUG_ERROR("NULL model_path or prompt provided");
    return set_error("NULL model_path or prompt provided");
  }

  // Create model
  DEBUG_LOG("Step 1: Creating model...");
  OgaModel *model = nullptr;
  OgaResult *result = OgaCreateModel(model_path, &model);
  if (check_oga_result(result, "Model creation failed") || model == nullptr) {
    DEBUG_ERROR("Model creation failed");
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 1: Model created successfully");

  // Create tokenizer
  DEBUG_LOG("Step 2: Creating tokenizer...");
  OgaTokenizer *tokenizer = nullptr;
  result = OgaCreateTokenizer(model, &tokenizer);
  if (check_oga_result(result, "Tokenizer creation failed") ||
      tokenizer == nullptr) {
    DEBUG_ERROR("Tokenizer creation failed");
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 2: Tokenizer created successfully");

  // Create sequences container first, then encode the prompt
  DEBUG_LOG("Step 3: Creating sequences and encoding prompt...");
  OgaSequences *input_sequences = nullptr;
  result = OgaCreateSequences(&input_sequences);
  if (check_oga_result(result, "Sequences creation failed") ||
      input_sequences == nullptr) {
    DEBUG_ERROR("Sequences creation failed");
    OgaDestroyTokenizer(tokenizer);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }

  result = OgaTokenizerEncode(tokenizer, prompt, input_sequences);
  if (check_oga_result(result, "Tokenization failed")) {
    DEBUG_ERROR("Tokenization failed");
    OgaDestroySequences(input_sequences);
    OgaDestroyTokenizer(tokenizer);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 3: Prompt encoded successfully");

  // Create generator parameters
  DEBUG_LOG("Step 4: Creating generator params...");
  OgaGeneratorParams *params = nullptr;
  result = OgaCreateGeneratorParams(model, &params);
  if (check_oga_result(result, "Generator params creation failed") ||
      params == nullptr) {
    DEBUG_ERROR("Generator params creation failed");
    OgaDestroySequences(input_sequences);
    OgaDestroyTokenizer(tokenizer);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }

  // Set max length if specified
  if (max_length > 0) {
    DEBUG_LOG("Setting max_length to %d", max_length);
    OgaGeneratorParamsSetSearchNumber(params, "max_length",
                                      static_cast<double>(max_length));
  }
  DEBUG_LOG("Step 4: Generator params created successfully");

  // Create generator
  DEBUG_LOG("Step 5: Creating generator...");
  OgaGenerator *generator = nullptr;
  result = OgaCreateGenerator(model, params, &generator);
  if (check_oga_result(result, "Generator creation failed") ||
      generator == nullptr) {
    DEBUG_ERROR("Generator creation failed");
    OgaDestroyGeneratorParams(params);
    OgaDestroySequences(input_sequences);
    OgaDestroyTokenizer(tokenizer);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 5: Generator created successfully");

  // Append input sequences to generator
  DEBUG_LOG("Step 6: Appending input sequences...");
  result = OgaGenerator_AppendTokenSequences(generator, input_sequences);
  if (check_oga_result(result, "Setting input sequences failed")) {
    DEBUG_ERROR("Setting input sequences failed");
    OgaDestroyGenerator(generator);
    OgaDestroyGeneratorParams(params);
    OgaDestroySequences(input_sequences);
    OgaDestroyTokenizer(tokenizer);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 6: Input sequences appended successfully");

  // Generate tokens
  DEBUG_LOG("Step 7: Creating tokenizer stream...");
  std::string generated_text;
  OgaTokenizerStream *stream = nullptr;
  result = OgaCreateTokenizerStream(tokenizer, &stream);
  if (check_oga_result(result, "Tokenizer stream creation failed") ||
      stream == nullptr) {
    DEBUG_ERROR("Tokenizer stream creation failed");
    OgaDestroyGenerator(generator);
    OgaDestroyGeneratorParams(params);
    OgaDestroySequences(input_sequences);
    OgaDestroyTokenizer(tokenizer);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 7: Tokenizer stream created successfully");

  DEBUG_LOG("Step 8: Starting token generation loop...");
  int generated_count = 0;
  while (!OgaGenerator_IsDone(generator)) {
    result = OgaGenerator_GenerateNextToken(generator);
    if (check_oga_result(result, "Generate next token failed")) {
      DEBUG_ERROR("Generate next token failed at token %d", generated_count);
      break;
    }

    // Get the last generated tokens
    const int32_t *tokens = nullptr;
    size_t token_count = 0;
    result = OgaGenerator_GetNextTokens(generator, &tokens, &token_count);
    if (check_oga_result(result, "Get next tokens failed") || token_count == 0) {
      DEBUG_ERROR("Get next tokens failed at token %d", generated_count);
      break;
    }

    // Decode first token to text (batch size = 1)
    const char *token_text = nullptr;
    result = OgaTokenizerStreamDecode(stream, tokens[0], &token_text);
    if (!check_oga_result(result, "Token decode failed") &&
        token_text != nullptr) {
      generated_text += token_text;
    }
    generated_count++;
    
    // Log progress every 50 tokens
    if (generated_count % 50 == 0) {
      DEBUG_LOG("Generated %d tokens so far...", generated_count);
    }
  }
  DEBUG_LOG("Step 8: Generation complete. Total tokens: %d", generated_count);

  // Cleanup
  DEBUG_LOG("Step 9: Cleaning up resources...");
  OgaDestroyTokenizerStream(stream);
  OgaDestroyGenerator(generator);
  OgaDestroyGeneratorParams(params);
  OgaDestroySequences(input_sequences);
  OgaDestroyTokenizer(tokenizer);
  OgaDestroyModel(model);
  DEBUG_LOG("Step 9: Cleanup complete");
  DEBUG_LOG("=== run_text_inference END ===");

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
  init_debug_features();
  DEBUG_LOG("=== run_inference START ===");
  DEBUG_LOG("model_path: %s", model_path ? model_path : "NULL");
  DEBUG_LOG("prompt length: %zu", prompt ? strlen(prompt) : 0);
  DEBUG_LOG("image_path: %s", image_path ? image_path : "NULL");

  if (model_path == nullptr || prompt == nullptr) {
    DEBUG_ERROR("NULL model_path or prompt provided");
    return set_error("NULL model_path or prompt provided");
  }

  // Create model
  DEBUG_LOG("Step 1: Creating model...");
  OgaModel *model = nullptr;
  OgaResult *result = OgaCreateModel(model_path, &model);
  if (check_oga_result(result, "Model creation failed") || model == nullptr) {
    DEBUG_ERROR("Model creation failed");
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 1: Model created successfully");

  // Create multimodal processor
  DEBUG_LOG("Step 2: Creating multimodal processor...");
  OgaMultiModalProcessor *processor = nullptr;
  result = OgaCreateMultiModalProcessor(model, &processor);
  if (check_oga_result(result, "MultiModal processor creation failed") ||
      processor == nullptr) {
    DEBUG_ERROR("MultiModal processor creation failed");
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 2: MultiModal processor created successfully");

  // Create tokenizer from model
  DEBUG_LOG("Step 3: Creating tokenizer...");
  OgaTokenizer *tokenizer = nullptr;
  result = OgaCreateTokenizer(model, &tokenizer);
  if (check_oga_result(result, "Tokenizer creation failed") ||
      tokenizer == nullptr) {
    DEBUG_ERROR("Tokenizer creation failed");
    OgaDestroyMultiModalProcessor(processor);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 3: Tokenizer created successfully");

  // Load image if provided
  OgaImages *images = nullptr;
  OgaNamedTensors *named_tensors = nullptr;

  if (image_path != nullptr && strlen(image_path) > 0) {
    DEBUG_LOG("Step 4: Loading image from: %s", image_path);
    result = OgaLoadImage(image_path, &images);
    if (check_oga_result(result, "Image loading failed") || images == nullptr) {
      DEBUG_ERROR("Image loading failed");
      OgaDestroyTokenizer(tokenizer);
      OgaDestroyMultiModalProcessor(processor);
      OgaDestroyModel(model);
      return set_error(g_error_buffer);
    }
    DEBUG_LOG("Step 4: Image loaded successfully");
  } else {
    DEBUG_LOG("Step 4: No image provided, will process text-only through multimodal processor");
  }

  // Process through multimodal processor (required for vision models even for text-only)
  DEBUG_LOG("Step 5: Processing prompt through multimodal processor (images=%p)...",
            (void *)images);
  result = OgaProcessorProcessImages(processor, prompt, images, &named_tensors);
  if (check_oga_result(result, "Multimodal processing failed") ||
      named_tensors == nullptr) {
    DEBUG_ERROR("Multimodal processing failed");
    if (images)
      OgaDestroyImages(images);
    OgaDestroyTokenizer(tokenizer);
    OgaDestroyMultiModalProcessor(processor);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 5: Multimodal processing completed successfully");

  // Create generator parameters
  DEBUG_LOG("Step 6: Creating generator params...");
  OgaGeneratorParams *params = nullptr;
  result = OgaCreateGeneratorParams(model, &params);
  if (check_oga_result(result, "Generator params creation failed") ||
      params == nullptr) {
    DEBUG_ERROR("Generator params creation failed");
    if (named_tensors)
      OgaDestroyNamedTensors(named_tensors);
    if (images)
      OgaDestroyImages(images);
    OgaDestroyTokenizer(tokenizer);
    OgaDestroyMultiModalProcessor(processor);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }
  
  // Set max_length to limit memory usage for KV-cache
  // Use 2048 tokens - must be larger than input prompt size (~954 tokens)
  DEBUG_LOG("Step 6a: Setting max_length to 2048 to limit memory...");
  result = OgaGeneratorParamsSetSearchNumber(params, "max_length", 2048.0);
  if (result != nullptr) {
    DEBUG_ERROR("Failed to set max_length: %s", OgaResultGetError(result));
    OgaDestroyResult(result);
    // Non-fatal, continue anyway
  } else {
    DEBUG_LOG("Step 6a: max_length set successfully");
  }
  
  DEBUG_LOG("Step 6: Generator params created successfully");

  // Create generator - this is often where multimodal models crash
  // due to memory allocation or model graph initialization issues
  OgaGenerator *generator = nullptr;
  DEBUG_LOG("Step 7: Creating generator...");
  DEBUG_LOG("Step 7a: Generator pointer address: %p", (void *)&generator);
  DEBUG_LOG("Step 7b: Model pointer: %p", (void *)model);
  DEBUG_LOG("Step 7c: Params pointer: %p", (void *)params);
  DEBUG_LOG("Step 7d: named_tensors: %p, images: %p",
            (void *)named_tensors, (void *)images);
#ifdef __ANDROID__
  // Force log flush before potentially crashing call
  __android_log_print(ANDROID_LOG_DEBUG, "OnnxGenAI",
                      "[DEBUG] Step 7e: About to call OgaCreateGenerator...");
#else
  fprintf(stderr, "[OnnxGenAI] Step 7e: About to call OgaCreateGenerator...\n");
  fflush(stderr);
#endif

  result = OgaCreateGenerator(model, params, &generator);

#ifdef __ANDROID__
  __android_log_print(ANDROID_LOG_DEBUG, "OnnxGenAI",
                      "[DEBUG] Step 7f: OgaCreateGenerator returned, result=%p",
                      (void *)result);
#else
  fprintf(stderr, "[OnnxGenAI] Step 7f: OgaCreateGenerator returned\n");
  fflush(stderr);
#endif

  if (check_oga_result(result, "Generator creation failed") ||
      generator == nullptr) {
    DEBUG_ERROR("Generator creation failed - generator=%p", (void *)generator);
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
  DEBUG_LOG("Step 7g: Generator created successfully, generator=%p",
            (void *)generator);

  // Set input tensors from multimodal processor
  DEBUG_LOG("Step 8: Setting input tensors...");
  DEBUG_LOG("Step 8a: generator=%p, named_tensors=%p", (void *)generator, (void *)named_tensors);
#ifdef __ANDROID__
  // Force log flush before potentially crashing call
  __android_log_print(ANDROID_LOG_DEBUG, "OnnxGenAI",
                      "[DEBUG] Step 8b: About to call OgaGenerator_SetInputs...");
#else
  fprintf(stderr, "[OnnxGenAI] Step 8b: About to call OgaGenerator_SetInputs...\n");
  fflush(stderr);
#endif

  result = OgaGenerator_SetInputs(generator, named_tensors);

#ifdef __ANDROID__
  __android_log_print(ANDROID_LOG_DEBUG, "OnnxGenAI",
                      "[DEBUG] Step 8c: OgaGenerator_SetInputs returned, result=%p",
                      (void *)result);
#else
  fprintf(stderr, "[OnnxGenAI] Step 8c: OgaGenerator_SetInputs returned\n");
  fflush(stderr);
#endif

  if (check_oga_result(result, "Setting input tensors failed")) {
    DEBUG_ERROR("Setting input tensors failed");
    OgaDestroyGenerator(generator);
    OgaDestroyGeneratorParams(params);
    OgaDestroyNamedTensors(named_tensors);
    if (images)
      OgaDestroyImages(images);
    OgaDestroyTokenizer(tokenizer);
    OgaDestroyMultiModalProcessor(processor);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 8d: Input tensors set successfully");

  // Create tokenizer stream for incremental decoding
  DEBUG_LOG("Step 9: Creating tokenizer stream...");
  OgaTokenizerStream *stream = nullptr;
  result = OgaCreateTokenizerStream(tokenizer, &stream);
  if (check_oga_result(result, "Tokenizer stream creation failed") ||
      stream == nullptr) {
    DEBUG_ERROR("Tokenizer stream creation failed");
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
  DEBUG_LOG("Step 9: Tokenizer stream created successfully");

  // Generate tokens
  DEBUG_LOG("Step 10: Starting token generation loop...");
  std::string generated_text;
  int generated_count = 0;
  while (!OgaGenerator_IsDone(generator)) {
    result = OgaGenerator_GenerateNextToken(generator);
    if (check_oga_result(result, "Generate next token failed")) {
      DEBUG_ERROR("Generate next token failed at token %d", generated_count);
      break;
    }

    // Get the last generated tokens
    const int32_t *tokens = nullptr;
    size_t token_count = 0;
    result = OgaGenerator_GetNextTokens(generator, &tokens, &token_count);
    if (check_oga_result(result, "Get next tokens failed") || token_count == 0) {
      DEBUG_ERROR("Get next tokens failed at token %d", generated_count);
      break;
    }

    // Decode first token to text (batch size = 1)
    const char *token_text = nullptr;
    result = OgaTokenizerStreamDecode(stream, tokens[0], &token_text);
    if (!check_oga_result(result, "Token decode failed") &&
        token_text != nullptr) {
      generated_text += token_text;
    }
    generated_count++;
    
    // Log progress every 50 tokens
    if (generated_count % 50 == 0) {
      DEBUG_LOG("Generated %d tokens so far...", generated_count);
    }
  }
  DEBUG_LOG("Step 10: Generation complete. Total tokens: %d", generated_count);

  // Cleanup
  DEBUG_LOG("Step 11: Cleaning up resources...");
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
  DEBUG_LOG("Step 11: Cleanup complete");
  DEBUG_LOG("=== run_inference END ===");

  return set_result(generated_text);
}

/**
 * @brief Run multimodal inference with text and multiple images.
 *
 * This function is designed for vision-language models like Phi-3.5 Vision.
 * It processes the text prompt along with multiple images for generation.
 * The prompt should contain image placeholders like <|image_1|>, <|image_2|>,
 * etc. matching the number of images provided.
 *
 * WARNING: This is a long-running operation! Call from a background Isolate
 * only.
 *
 * @param model_path Path to the ONNX GenAI model directory
 * @param prompt The text prompt for generation (with image placeholders)
 * @param image_paths Array of paths to image files
 * @param image_count Number of images in the array
 * @return Generated text on success, error message starting with "ERROR:" on
 * failure
 */
FFI_PLUGIN_EXPORT const char *run_inference_multi(const char *model_path,
                                                  const char *prompt,
                                                  const char **image_paths,
                                                  int32_t image_count) {
  init_debug_features();
  DEBUG_LOG("=== run_inference_multi START ===");
  DEBUG_LOG("model_path: %s", model_path ? model_path : "NULL");
  DEBUG_LOG("prompt length: %zu", prompt ? strlen(prompt) : 0);
  DEBUG_LOG("image_count: %d", image_count);
  for (int i = 0; i < image_count && image_paths != nullptr; i++) {
    DEBUG_LOG("image_paths[%d]: %s", i, image_paths[i] ? image_paths[i] : "NULL");
  }

  if (model_path == nullptr || prompt == nullptr) {
    DEBUG_ERROR("NULL model_path or prompt provided");
    return set_error("NULL model_path or prompt provided");
  }

  if (image_count > 0 && image_paths == nullptr) {
    DEBUG_ERROR("image_paths is NULL but image_count > 0");
    return set_error("image_paths is NULL but image_count > 0");
  }

  // Create model
  DEBUG_LOG("Step 1: Creating model...");
  OgaModel *model = nullptr;
  OgaResult *result = OgaCreateModel(model_path, &model);
  if (check_oga_result(result, "Model creation failed") || model == nullptr) {
    DEBUG_ERROR("Model creation failed");
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 1: Model created successfully");

  // Create multimodal processor
  DEBUG_LOG("Step 2: Creating multimodal processor...");
  OgaMultiModalProcessor *processor = nullptr;
  result = OgaCreateMultiModalProcessor(model, &processor);
  if (check_oga_result(result, "MultiModal processor creation failed") ||
      processor == nullptr) {
    DEBUG_ERROR("MultiModal processor creation failed");
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 2: MultiModal processor created successfully");

  // Create tokenizer from model
  DEBUG_LOG("Step 3: Creating tokenizer...");
  OgaTokenizer *tokenizer = nullptr;
  result = OgaCreateTokenizer(model, &tokenizer);
  if (check_oga_result(result, "Tokenizer creation failed") ||
      tokenizer == nullptr) {
    DEBUG_ERROR("Tokenizer creation failed");
    OgaDestroyMultiModalProcessor(processor);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 3: Tokenizer created successfully");

  // Load images if provided
  OgaImages *images = nullptr;
  OgaNamedTensors *named_tensors = nullptr;
  OgaStringArray *image_path_array = nullptr;

  if (image_count > 0) {
    // Create string array from image paths
    DEBUG_LOG("Step 4: Creating string array from %d image paths...", image_count);
    result = OgaCreateStringArrayFromStrings(image_paths,
                                             static_cast<size_t>(image_count),
                                             &image_path_array);
    if (check_oga_result(result, "String array creation failed") ||
        image_path_array == nullptr) {
      DEBUG_ERROR("String array creation failed");
      OgaDestroyTokenizer(tokenizer);
      OgaDestroyMultiModalProcessor(processor);
      OgaDestroyModel(model);
      return set_error(g_error_buffer);
    }
    DEBUG_LOG("Step 4: String array created successfully");

    // Load all images
    DEBUG_LOG("Step 5: Loading %d images...", image_count);
    result = OgaLoadImages(image_path_array, &images);
    if (check_oga_result(result, "Image loading failed") || images == nullptr) {
      DEBUG_ERROR("Image loading failed");
      OgaDestroyStringArray(image_path_array);
      OgaDestroyTokenizer(tokenizer);
      OgaDestroyMultiModalProcessor(processor);
      OgaDestroyModel(model);
      return set_error(g_error_buffer);
    }
    DEBUG_LOG("Step 5: Images loaded successfully");
  } else {
    DEBUG_LOG("Step 4-5: No images provided, will process text-only through multimodal processor");
  }

  // Process through multimodal processor (required for vision models even for text-only)
  DEBUG_LOG("Step 6: Processing prompt through multimodal processor (images=%p)...",
            (void *)images);
  result = OgaProcessorProcessImages(processor, prompt, images, &named_tensors);
  if (check_oga_result(result, "Multimodal processing failed") ||
      named_tensors == nullptr) {
    DEBUG_ERROR("Multimodal processing failed");
    if (images)
      OgaDestroyImages(images);
    if (image_path_array)
      OgaDestroyStringArray(image_path_array);
    OgaDestroyTokenizer(tokenizer);
    OgaDestroyMultiModalProcessor(processor);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 6: Multimodal processing completed successfully");

  // Create generator parameters
  DEBUG_LOG("Step 7: Creating generator params...");
  OgaGeneratorParams *params = nullptr;
  result = OgaCreateGeneratorParams(model, &params);
  if (check_oga_result(result, "Generator params creation failed") ||
      params == nullptr) {
    DEBUG_ERROR("Generator params creation failed");
    if (named_tensors)
      OgaDestroyNamedTensors(named_tensors);
    if (images)
      OgaDestroyImages(images);
    if (image_path_array)
      OgaDestroyStringArray(image_path_array);
    OgaDestroyTokenizer(tokenizer);
    OgaDestroyMultiModalProcessor(processor);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }
  
  // Set max_length to limit memory usage for KV-cache
  // Use 2048 tokens - must be larger than input prompt size (~954 tokens)
  DEBUG_LOG("Step 7a: Setting max_length to 2048 to limit memory...");
  result = OgaGeneratorParamsSetSearchNumber(params, "max_length", 2048.0);
  if (result != nullptr) {
    DEBUG_ERROR("Failed to set max_length: %s", OgaResultGetError(result));
    OgaDestroyResult(result);
    // Non-fatal, continue anyway
  } else {
    DEBUG_LOG("Step 7a: max_length set successfully");
  }
  
  DEBUG_LOG("Step 7: Generator params created successfully");

  // Create generator - this is often where multimodal models crash
  // due to memory allocation or model graph initialization issues
  OgaGenerator *generator = nullptr;
  DEBUG_LOG("Step 8: Creating generator...");
  DEBUG_LOG("Step 8a: Generator pointer address: %p", (void *)&generator);
  DEBUG_LOG("Step 8b: Model pointer: %p", (void *)model);
  DEBUG_LOG("Step 8c: Params pointer: %p", (void *)params);
  DEBUG_LOG("Step 8d: named_tensors: %p, images: %p, image_count: %d",
            (void *)named_tensors, (void *)images, image_count);
#ifdef __ANDROID__
  // Force log flush before potentially crashing call
  __android_log_print(ANDROID_LOG_DEBUG, "OnnxGenAI",
                      "[DEBUG] Step 8e: About to call OgaCreateGenerator...");
#else
  fprintf(stderr, "[OnnxGenAI] Step 8e: About to call OgaCreateGenerator...\n");
  fflush(stderr);
#endif

  result = OgaCreateGenerator(model, params, &generator);

#ifdef __ANDROID__
  __android_log_print(ANDROID_LOG_DEBUG, "OnnxGenAI",
                      "[DEBUG] Step 8f: OgaCreateGenerator returned, result=%p",
                      (void *)result);
#else
  fprintf(stderr, "[OnnxGenAI] Step 8f: OgaCreateGenerator returned\n");
  fflush(stderr);
#endif

  if (check_oga_result(result, "Generator creation failed") ||
      generator == nullptr) {
    DEBUG_ERROR("Generator creation failed - generator=%p", (void *)generator);
    OgaDestroyGeneratorParams(params);
    if (named_tensors)
      OgaDestroyNamedTensors(named_tensors);
    if (images)
      OgaDestroyImages(images);
    if (image_path_array)
      OgaDestroyStringArray(image_path_array);
    OgaDestroyTokenizer(tokenizer);
    OgaDestroyMultiModalProcessor(processor);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 8g: Generator created successfully, generator=%p",
            (void *)generator);

  // Set input tensors from multimodal processor
  DEBUG_LOG("Step 9: Setting input tensors...");
  DEBUG_LOG("Step 9a: generator=%p, named_tensors=%p", (void *)generator, (void *)named_tensors);
#ifdef __ANDROID__
  // Force log flush before potentially crashing call
  __android_log_print(ANDROID_LOG_DEBUG, "OnnxGenAI",
                      "[DEBUG] Step 9b: About to call OgaGenerator_SetInputs...");
#else
  fprintf(stderr, "[OnnxGenAI] Step 9b: About to call OgaGenerator_SetInputs...\n");
  fflush(stderr);
#endif

  result = OgaGenerator_SetInputs(generator, named_tensors);

#ifdef __ANDROID__
  __android_log_print(ANDROID_LOG_DEBUG, "OnnxGenAI",
                      "[DEBUG] Step 9c: OgaGenerator_SetInputs returned, result=%p",
                      (void *)result);
#else
  fprintf(stderr, "[OnnxGenAI] Step 9c: OgaGenerator_SetInputs returned\n");
  fflush(stderr);
#endif

  if (check_oga_result(result, "Setting input tensors failed")) {
    DEBUG_ERROR("Setting input tensors failed");
    OgaDestroyGenerator(generator);
    OgaDestroyGeneratorParams(params);
    OgaDestroyNamedTensors(named_tensors);
    if (images)
      OgaDestroyImages(images);
    if (image_path_array)
      OgaDestroyStringArray(image_path_array);
    OgaDestroyTokenizer(tokenizer);
    OgaDestroyMultiModalProcessor(processor);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 9d: Input tensors set successfully");

  // Create tokenizer stream for incremental decoding
  DEBUG_LOG("Step 10: Creating tokenizer stream...");
  OgaTokenizerStream *stream = nullptr;
  result = OgaCreateTokenizerStream(tokenizer, &stream);
  if (check_oga_result(result, "Tokenizer stream creation failed") ||
      stream == nullptr) {
    DEBUG_ERROR("Tokenizer stream creation failed");
    OgaDestroyGenerator(generator);
    OgaDestroyGeneratorParams(params);
    if (named_tensors)
      OgaDestroyNamedTensors(named_tensors);
    if (images)
      OgaDestroyImages(images);
    if (image_path_array)
      OgaDestroyStringArray(image_path_array);
    OgaDestroyTokenizer(tokenizer);
    OgaDestroyMultiModalProcessor(processor);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 10: Tokenizer stream created successfully");

  // Generate tokens
  DEBUG_LOG("Step 11: Starting token generation loop...");
  std::string generated_text;
  int generated_count = 0;
  while (!OgaGenerator_IsDone(generator)) {
    result = OgaGenerator_GenerateNextToken(generator);
    if (check_oga_result(result, "Generate next token failed")) {
      DEBUG_ERROR("Generate next token failed at token %d", generated_count);
      break;
    }

    // Get the last generated tokens
    const int32_t *tokens = nullptr;
    size_t token_count = 0;
    result = OgaGenerator_GetNextTokens(generator, &tokens, &token_count);
    if (check_oga_result(result, "Get next tokens failed") ||
        token_count == 0) {
      DEBUG_ERROR("Get next tokens failed at token %d", generated_count);
      break;
    }

    // Decode first token to text (batch size = 1)
    const char *token_text = nullptr;
    result = OgaTokenizerStreamDecode(stream, tokens[0], &token_text);
    if (!check_oga_result(result, "Token decode failed") &&
        token_text != nullptr) {
      generated_text += token_text;
    }
    generated_count++;
    
    // Log progress every 50 tokens
    if (generated_count % 50 == 0) {
      DEBUG_LOG("Generated %d tokens so far...", generated_count);
    }
  }
  DEBUG_LOG("Step 11: Generation complete. Total tokens: %d", generated_count);

  // Cleanup
  DEBUG_LOG("Step 12: Cleaning up resources...");
  OgaDestroyTokenizerStream(stream);
  OgaDestroyGenerator(generator);
  OgaDestroyGeneratorParams(params);
  if (named_tensors)
    OgaDestroyNamedTensors(named_tensors);
  if (images)
    OgaDestroyImages(images);
  if (image_path_array)
    OgaDestroyStringArray(image_path_array);
  OgaDestroyTokenizer(tokenizer);
  OgaDestroyMultiModalProcessor(processor);
  OgaDestroyModel(model);
  DEBUG_LOG("Step 12: Cleanup complete");
  DEBUG_LOG("=== run_inference_multi END ===");

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
FFI_PLUGIN_EXPORT const char *get_library_version() { return "0.3.0"; }

// =============================================================================
// Configuration API Implementation
// =============================================================================

/**
 * @brief Create a configuration object from a model path.
 */
FFI_PLUGIN_EXPORT int64_t create_config(const char *model_path) {
  init_debug_features();
  DEBUG_LOG("=== create_config START ===");
  DEBUG_LOG("model_path: %s", model_path ? model_path : "NULL");

  if (model_path == nullptr || strlen(model_path) == 0) {
    DEBUG_ERROR("NULL or empty model_path provided");
    set_error("NULL or empty model_path provided");
    return 0;
  }

  OgaConfig *config = nullptr;
  OgaResult *result = OgaCreateConfig(model_path, &config);
  
  if (check_oga_result(result, "Config creation failed") || config == nullptr) {
    DEBUG_ERROR("Config creation failed");
    return 0;
  }
  
  DEBUG_LOG("Config created successfully: %p", (void*)config);
  DEBUG_LOG("=== create_config END ===");
  
  return reinterpret_cast<int64_t>(config);
}

/**
 * @brief Destroy a configuration object.
 */
FFI_PLUGIN_EXPORT void destroy_config(int64_t config_handle) {
  DEBUG_LOG("=== destroy_config ===");
  if (config_handle == 0) {
    DEBUG_ERROR("NULL config handle");
    return;
  }
  
  OgaConfig *config = reinterpret_cast<OgaConfig*>(config_handle);
  OgaDestroyConfig(config);
  DEBUG_LOG("Config destroyed");
}

/**
 * @brief Clear all execution providers from the config.
 */
FFI_PLUGIN_EXPORT int32_t config_clear_providers(int64_t config_handle) {
  DEBUG_LOG("=== config_clear_providers ===");
  if (config_handle == 0) {
    DEBUG_ERROR("NULL config handle");
    set_error("NULL config handle");
    return -1;
  }
  
  OgaConfig *config = reinterpret_cast<OgaConfig*>(config_handle);
  OgaResult *result = OgaConfigClearProviders(config);
  
  if (check_oga_result(result, "Clear providers failed")) {
    return -2;
  }
  
  DEBUG_LOG("Providers cleared successfully");
  return 1;
}

/**
 * @brief Append an execution provider to the config.
 */
FFI_PLUGIN_EXPORT int32_t config_append_provider(int64_t config_handle,
                                                  const char *provider_name) {
  DEBUG_LOG("=== config_append_provider ===");
  DEBUG_LOG("provider_name: %s", provider_name ? provider_name : "NULL");
  
  if (config_handle == 0) {
    DEBUG_ERROR("NULL config handle");
    set_error("NULL config handle");
    return -1;
  }
  
  if (provider_name == nullptr || strlen(provider_name) == 0) {
    DEBUG_ERROR("NULL or empty provider name");
    set_error("NULL or empty provider name");
    return -2;
  }
  
  OgaConfig *config = reinterpret_cast<OgaConfig*>(config_handle);
  OgaResult *result = OgaConfigAppendProvider(config, provider_name);
  
  if (check_oga_result(result, "Append provider failed")) {
    return -3;
  }
  
  DEBUG_LOG("Provider '%s' appended successfully", provider_name);
  return 1;
}

/**
 * @brief Set an option for a specific execution provider.
 */
FFI_PLUGIN_EXPORT int32_t config_set_provider_option(int64_t config_handle,
                                                      const char *provider_name,
                                                      const char *key,
                                                      const char *value) {
  DEBUG_LOG("=== config_set_provider_option ===");
  DEBUG_LOG("provider: %s, key: %s, value: %s",
            provider_name ? provider_name : "NULL",
            key ? key : "NULL",
            value ? value : "NULL");
  
  if (config_handle == 0) {
    DEBUG_ERROR("NULL config handle");
    set_error("NULL config handle");
    return -1;
  }
  
  if (provider_name == nullptr || key == nullptr || value == nullptr) {
    DEBUG_ERROR("NULL parameter");
    set_error("NULL parameter");
    return -2;
  }
  
  OgaConfig *config = reinterpret_cast<OgaConfig*>(config_handle);
  OgaResult *result = OgaConfigSetProviderOption(config, provider_name, key, value);
  
  if (check_oga_result(result, "Set provider option failed")) {
    return -3;
  }
  
  DEBUG_LOG("Option set successfully");
  return 1;
}

/**
 * @brief Run inference using a pre-configured config.
 */
FFI_PLUGIN_EXPORT const char *run_inference_with_config(int64_t config_handle,
                                                         const char *prompt,
                                                         const char *image_path) {
  init_debug_features();
  DEBUG_LOG("=== run_inference_with_config START ===");
  DEBUG_LOG("config_handle: %lld", (long long)config_handle);
  DEBUG_LOG("prompt length: %zu", prompt ? strlen(prompt) : 0);
  DEBUG_LOG("image_path: %s", image_path ? image_path : "NULL");

  if (config_handle == 0) {
    DEBUG_ERROR("NULL config handle");
    return set_error("NULL config handle");
  }
  
  if (prompt == nullptr) {
    DEBUG_ERROR("NULL prompt provided");
    return set_error("NULL prompt provided");
  }

  OgaConfig *config = reinterpret_cast<OgaConfig*>(config_handle);

  // Create model from config
  DEBUG_LOG("Step 1: Creating model from config...");
  OgaModel *model = nullptr;
  OgaResult *result = OgaCreateModelFromConfig(config, &model);
  if (check_oga_result(result, "Model creation from config failed") || model == nullptr) {
    DEBUG_ERROR("Model creation from config failed");
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 1: Model created successfully from config");

  // Create multimodal processor
  DEBUG_LOG("Step 2: Creating multimodal processor...");
  OgaMultiModalProcessor *processor = nullptr;
  result = OgaCreateMultiModalProcessor(model, &processor);
  if (check_oga_result(result, "MultiModal processor creation failed") ||
      processor == nullptr) {
    DEBUG_ERROR("MultiModal processor creation failed");
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 2: MultiModal processor created successfully");

  // Create tokenizer from model
  DEBUG_LOG("Step 3: Creating tokenizer...");
  OgaTokenizer *tokenizer = nullptr;
  result = OgaCreateTokenizer(model, &tokenizer);
  if (check_oga_result(result, "Tokenizer creation failed") ||
      tokenizer == nullptr) {
    DEBUG_ERROR("Tokenizer creation failed");
    OgaDestroyMultiModalProcessor(processor);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 3: Tokenizer created successfully");

  // Load image if provided
  OgaImages *images = nullptr;
  OgaNamedTensors *named_tensors = nullptr;

  if (image_path != nullptr && strlen(image_path) > 0) {
    DEBUG_LOG("Step 4: Loading image from: %s", image_path);
    result = OgaLoadImage(image_path, &images);
    if (check_oga_result(result, "Image loading failed") || images == nullptr) {
      DEBUG_ERROR("Image loading failed");
      OgaDestroyTokenizer(tokenizer);
      OgaDestroyMultiModalProcessor(processor);
      OgaDestroyModel(model);
      return set_error(g_error_buffer);
    }
    DEBUG_LOG("Step 4: Image loaded successfully");
  } else {
    DEBUG_LOG("Step 4: No image provided, will process text-only through multimodal processor");
  }

  // Process through multimodal processor
  DEBUG_LOG("Step 5: Processing prompt through multimodal processor...");
  result = OgaProcessorProcessImages(processor, prompt, images, &named_tensors);
  if (check_oga_result(result, "Multimodal processing failed") ||
      named_tensors == nullptr) {
    DEBUG_ERROR("Multimodal processing failed");
    if (images)
      OgaDestroyImages(images);
    OgaDestroyTokenizer(tokenizer);
    OgaDestroyMultiModalProcessor(processor);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 5: Multimodal processing completed successfully");

  // Create generator parameters
  DEBUG_LOG("Step 6: Creating generator params...");
  OgaGeneratorParams *params = nullptr;
  result = OgaCreateGeneratorParams(model, &params);
  if (check_oga_result(result, "Generator params creation failed") ||
      params == nullptr) {
    DEBUG_ERROR("Generator params creation failed");
    if (named_tensors)
      OgaDestroyNamedTensors(named_tensors);
    if (images)
      OgaDestroyImages(images);
    OgaDestroyTokenizer(tokenizer);
    OgaDestroyMultiModalProcessor(processor);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }
  
  // Set max_length to limit memory usage
  DEBUG_LOG("Step 6a: Setting max_length to 2048...");
  result = OgaGeneratorParamsSetSearchNumber(params, "max_length", 2048.0);
  if (result != nullptr) {
    DEBUG_ERROR("Failed to set max_length: %s", OgaResultGetError(result));
    OgaDestroyResult(result);
  } else {
    DEBUG_LOG("Step 6a: max_length set successfully");
  }
  
  DEBUG_LOG("Step 6: Generator params created successfully");

  // Create generator
  DEBUG_LOG("Step 7: Creating generator...");
  OgaGenerator *generator = nullptr;
  result = OgaCreateGenerator(model, params, &generator);
  if (check_oga_result(result, "Generator creation failed") ||
      generator == nullptr) {
    DEBUG_ERROR("Generator creation failed");
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
  DEBUG_LOG("Step 7: Generator created successfully");

  // Set input tensors
  DEBUG_LOG("Step 8: Setting input tensors...");
  result = OgaGenerator_SetInputs(generator, named_tensors);
  if (check_oga_result(result, "Setting input tensors failed")) {
    DEBUG_ERROR("Setting input tensors failed");
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
  DEBUG_LOG("Step 8: Input tensors set successfully");

  // Create tokenizer stream
  DEBUG_LOG("Step 9: Creating tokenizer stream...");
  OgaTokenizerStream *stream = nullptr;
  result = OgaCreateTokenizerStream(tokenizer, &stream);
  if (check_oga_result(result, "Tokenizer stream creation failed") ||
      stream == nullptr) {
    DEBUG_ERROR("Tokenizer stream creation failed");
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
  DEBUG_LOG("Step 9: Tokenizer stream created successfully");

  // Generation loop
  DEBUG_LOG("Step 10: Starting token generation loop...");
  std::string generated_text;
  int generated_count = 0;

  while (!OgaGenerator_IsDone(generator)) {
    result = OgaGenerator_GenerateNextToken(generator);
    if (check_oga_result(result, "Generate next token failed")) {
      DEBUG_ERROR("Generate next token failed at token %d", generated_count);
      break;
    }

    const int32_t *tokens = nullptr;
    size_t token_count = 0;
    result = OgaGenerator_GetNextTokens(generator, &tokens, &token_count);
    if (check_oga_result(result, "Get next tokens failed") ||
        token_count == 0) {
      DEBUG_ERROR("Get next tokens failed at token %d", generated_count);
      break;
    }

    const char *token_text = nullptr;
    result = OgaTokenizerStreamDecode(stream, tokens[0], &token_text);
    if (!check_oga_result(result, "Token decode failed") &&
        token_text != nullptr) {
      generated_text += token_text;
    }
    generated_count++;
    
    if (generated_count % 50 == 0) {
      DEBUG_LOG("Generated %d tokens so far...", generated_count);
    }
  }
  DEBUG_LOG("Step 10: Generation complete. Total tokens: %d", generated_count);

  // Cleanup
  DEBUG_LOG("Step 11: Cleaning up resources...");
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
  DEBUG_LOG("Step 11: Cleanup complete");
  DEBUG_LOG("=== run_inference_with_config END ===");

  return set_result(generated_text);
}

/**
 * @brief Run multi-image inference using a pre-configured config.
 */
FFI_PLUGIN_EXPORT const char *run_inference_multi_with_config(int64_t config_handle,
                                                               const char *prompt,
                                                               const char **image_paths,
                                                               int32_t image_count) {
  init_debug_features();
  DEBUG_LOG("=== run_inference_multi_with_config START ===");
  DEBUG_LOG("config_handle: %lld", (long long)config_handle);
  DEBUG_LOG("prompt length: %zu", prompt ? strlen(prompt) : 0);
  DEBUG_LOG("image_count: %d", image_count);

  if (config_handle == 0) {
    DEBUG_ERROR("NULL config handle");
    return set_error("NULL config handle");
  }
  
  if (prompt == nullptr) {
    DEBUG_ERROR("NULL prompt provided");
    return set_error("NULL prompt provided");
  }

  if (image_count > 0 && image_paths == nullptr) {
    DEBUG_ERROR("NULL image_paths with image_count > 0");
    return set_error("NULL image_paths with image_count > 0");
  }

  OgaConfig *config = reinterpret_cast<OgaConfig*>(config_handle);

  // Create model from config
  DEBUG_LOG("Step 1: Creating model from config...");
  OgaModel *model = nullptr;
  OgaResult *result = OgaCreateModelFromConfig(config, &model);
  if (check_oga_result(result, "Model creation from config failed") || model == nullptr) {
    DEBUG_ERROR("Model creation from config failed");
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 1: Model created successfully from config");

  // Create multimodal processor
  DEBUG_LOG("Step 2: Creating multimodal processor...");
  OgaMultiModalProcessor *processor = nullptr;
  result = OgaCreateMultiModalProcessor(model, &processor);
  if (check_oga_result(result, "MultiModal processor creation failed") ||
      processor == nullptr) {
    DEBUG_ERROR("MultiModal processor creation failed");
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 2: MultiModal processor created successfully");

  // Create tokenizer
  DEBUG_LOG("Step 3: Creating tokenizer...");
  OgaTokenizer *tokenizer = nullptr;
  result = OgaCreateTokenizer(model, &tokenizer);
  if (check_oga_result(result, "Tokenizer creation failed") ||
      tokenizer == nullptr) {
    DEBUG_ERROR("Tokenizer creation failed");
    OgaDestroyMultiModalProcessor(processor);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 3: Tokenizer created successfully");

  // Load images if provided
  OgaImages *images = nullptr;
  OgaStringArray *image_path_array = nullptr;
  
  if (image_count > 0) {
    DEBUG_LOG("Step 4: Loading %d images...", image_count);
    
    // Create string array for image paths
    result = OgaCreateStringArray(&image_path_array);
    if (check_oga_result(result, "String array creation failed") ||
        image_path_array == nullptr) {
      DEBUG_ERROR("String array creation failed");
      OgaDestroyTokenizer(tokenizer);
      OgaDestroyMultiModalProcessor(processor);
      OgaDestroyModel(model);
      return set_error(g_error_buffer);
    }

    // Add each image path to the array
    for (int i = 0; i < image_count; i++) {
      DEBUG_LOG("  Adding image %d: %s", i + 1, image_paths[i] ? image_paths[i] : "NULL");
      if (image_paths[i] == nullptr) {
        DEBUG_ERROR("NULL image path at index %d", i);
        OgaDestroyStringArray(image_path_array);
        OgaDestroyTokenizer(tokenizer);
        OgaDestroyMultiModalProcessor(processor);
        OgaDestroyModel(model);
        return set_error("NULL image path in array");
      }
      result = OgaStringArrayAddString(image_path_array, image_paths[i]);
      if (check_oga_result(result, "Add image path failed")) {
        DEBUG_ERROR("Failed to add image path at index %d", i);
        OgaDestroyStringArray(image_path_array);
        OgaDestroyTokenizer(tokenizer);
        OgaDestroyMultiModalProcessor(processor);
        OgaDestroyModel(model);
        return set_error(g_error_buffer);
      }
    }

    // Load all images
    result = OgaLoadImages(image_path_array, &images);
    if (check_oga_result(result, "Image loading failed") || images == nullptr) {
      DEBUG_ERROR("Failed to load images");
      OgaDestroyStringArray(image_path_array);
      OgaDestroyTokenizer(tokenizer);
      OgaDestroyMultiModalProcessor(processor);
      OgaDestroyModel(model);
      return set_error(g_error_buffer);
    }
    DEBUG_LOG("Step 4: All images loaded successfully");
  } else {
    DEBUG_LOG("Step 4: No images - text-only inference");
  }

  // Process prompt and images
  DEBUG_LOG("Step 5: Processing prompt and images...");
  OgaNamedTensors *named_tensors = nullptr;
  result = OgaProcessorProcessImages(processor, prompt, images, &named_tensors);
  if (check_oga_result(result, "Processing failed") ||
      named_tensors == nullptr) {
    DEBUG_ERROR("Processing failed");
    if (images)
      OgaDestroyImages(images);
    if (image_path_array)
      OgaDestroyStringArray(image_path_array);
    OgaDestroyTokenizer(tokenizer);
    OgaDestroyMultiModalProcessor(processor);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 5: Processing complete");

  // Create generator params
  DEBUG_LOG("Step 6: Creating generator params...");
  OgaGeneratorParams *params = nullptr;
  result = OgaCreateGeneratorParams(model, &params);
  if (check_oga_result(result, "Generator params creation failed") ||
      params == nullptr) {
    DEBUG_ERROR("Generator params creation failed");
    OgaDestroyNamedTensors(named_tensors);
    if (images)
      OgaDestroyImages(images);
    if (image_path_array)
      OgaDestroyStringArray(image_path_array);
    OgaDestroyTokenizer(tokenizer);
    OgaDestroyMultiModalProcessor(processor);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 6: Generator params created");

  // Create generator
  DEBUG_LOG("Step 7: Creating generator...");
  OgaGenerator *generator = nullptr;
  result = OgaCreateGenerator(model, params, &generator);
  if (check_oga_result(result, "Generator creation failed") ||
      generator == nullptr) {
    DEBUG_ERROR("Generator creation failed");
    OgaDestroyGeneratorParams(params);
    OgaDestroyNamedTensors(named_tensors);
    if (images)
      OgaDestroyImages(images);
    if (image_path_array)
      OgaDestroyStringArray(image_path_array);
    OgaDestroyTokenizer(tokenizer);
    OgaDestroyMultiModalProcessor(processor);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 7: Generator created");

  // Set input tensors
  DEBUG_LOG("Step 8: Setting input tensors...");
  result = OgaGenerator_SetInputs(generator, named_tensors);
  if (check_oga_result(result, "Setting input tensors failed")) {
    DEBUG_ERROR("Setting input tensors failed");
    OgaDestroyGenerator(generator);
    OgaDestroyGeneratorParams(params);
    OgaDestroyNamedTensors(named_tensors);
    if (images)
      OgaDestroyImages(images);
    if (image_path_array)
      OgaDestroyStringArray(image_path_array);
    OgaDestroyTokenizer(tokenizer);
    OgaDestroyMultiModalProcessor(processor);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 8: Input tensors set");

  // Create tokenizer stream
  DEBUG_LOG("Step 9: Creating tokenizer stream...");
  OgaTokenizerStream *stream = nullptr;
  result = OgaCreateTokenizerStream(tokenizer, &stream);
  if (check_oga_result(result, "Tokenizer stream creation failed") ||
      stream == nullptr) {
    DEBUG_ERROR("Tokenizer stream creation failed");
    OgaDestroyGenerator(generator);
    OgaDestroyGeneratorParams(params);
    OgaDestroyNamedTensors(named_tensors);
    if (images)
      OgaDestroyImages(images);
    if (image_path_array)
      OgaDestroyStringArray(image_path_array);
    OgaDestroyTokenizer(tokenizer);
    OgaDestroyMultiModalProcessor(processor);
    OgaDestroyModel(model);
    return set_error(g_error_buffer);
  }
  DEBUG_LOG("Step 9: Tokenizer stream created");

  // Generation loop
  DEBUG_LOG("Step 10: Starting token generation loop...");
  std::string generated_text;
  int generated_count = 0;

  while (!OgaGenerator_IsDone(generator)) {
    result = OgaGenerator_GenerateNextToken(generator);
    if (check_oga_result(result, "Generate next token failed")) {
      DEBUG_ERROR("Generate next token failed at token %d", generated_count);
      break;
    }

    const int32_t *tokens = nullptr;
    size_t token_count = 0;
    result = OgaGenerator_GetNextTokens(generator, &tokens, &token_count);
    if (check_oga_result(result, "Get next tokens failed") ||
        token_count == 0) {
      DEBUG_ERROR("Get next tokens failed at token %d", generated_count);
      break;
    }

    const char *token_text = nullptr;
    result = OgaTokenizerStreamDecode(stream, tokens[0], &token_text);
    if (!check_oga_result(result, "Token decode failed") &&
        token_text != nullptr) {
      generated_text += token_text;
    }
    generated_count++;
    
    if (generated_count % 50 == 0) {
      DEBUG_LOG("Generated %d tokens so far...", generated_count);
    }
  }
  DEBUG_LOG("Step 10: Generation complete. Total tokens: %d", generated_count);

  // Cleanup
  DEBUG_LOG("Step 11: Cleaning up resources...");
  OgaDestroyTokenizerStream(stream);
  OgaDestroyGenerator(generator);
  OgaDestroyGeneratorParams(params);
  OgaDestroyNamedTensors(named_tensors);
  if (images)
    OgaDestroyImages(images);
  if (image_path_array)
    OgaDestroyStringArray(image_path_array);
  OgaDestroyTokenizer(tokenizer);
  OgaDestroyMultiModalProcessor(processor);
  OgaDestroyModel(model);
  DEBUG_LOG("Step 11: Cleanup complete");
  DEBUG_LOG("=== run_inference_multi_with_config END ===");

  return set_result(generated_text);
}

/**
 * @brief Get the last error message.
 */
FFI_PLUGIN_EXPORT const char *get_last_error() {
  return g_error_buffer.c_str();
}

} // extern "C"
