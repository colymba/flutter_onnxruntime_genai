/**
 * @file flutter_onnxruntime_genai.h
 * @brief Header file for Flutter FFI Bridge to ONNX Runtime GenAI
 *
 * This header declares the C FFI functions that bridge Dart code with the
 * ONNX Runtime GenAI C-API for multimodal inference.
 */

#ifndef FLUTTER_ONNXRUNTIME_GENAI_H
#define FLUTTER_ONNXRUNTIME_GENAI_H

#include <stddef.h>
#include <stdint.h>

// Platform-specific export macro
#if defined(_WIN32)
#define FFI_PLUGIN_EXPORT __declspec(dllexport)
#elif defined(__GNUC__) && __GNUC__ >= 4
#define FFI_PLUGIN_EXPORT __attribute__((visibility("default")))
#else
#define FFI_PLUGIN_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Health Check & Utility Functions
// =============================================================================

/**
 * @brief Check if the native library and model can be loaded.
 *
 * Use this function to verify that:
 * 1. The native library is properly linked
 * 2. The model path is accessible
 * 3. The model can be loaded successfully
 *
 * @param model_path Path to the ONNX GenAI model directory
 * @return Status code:
 *         1: Success - model loaded and verified
 *        -1: NULL or empty path provided
 *        -2: Model creation failed
 *        -3: Tokenizer creation failed
 */
FFI_PLUGIN_EXPORT int32_t check_native_health(const char *model_path);

/**
 * @brief Get the library version string.
 * @return Version string in format "major.minor.patch"
 */
FFI_PLUGIN_EXPORT const char *get_library_version();

/**
 * @brief Shutdown the ONNX GenAI library and free global resources.
 * Call this when the application is shutting down.
 */
FFI_PLUGIN_EXPORT void shutdown_onnx_genai();

// =============================================================================
// Inference Functions
// =============================================================================

/**
 * @brief Run multimodal inference with text and optional image.
 *
 * This function is designed for vision-language models like Phi-3.5 Vision.
 * It processes both the text prompt and image together for generation.
 *
 * WARNING: This is a LONG-RUNNING operation!
 * MUST be called from a background Dart Isolate, NOT the main UI isolate.
 * Calling from the main isolate will block the UI and cause dropped frames.
 *
 * @param model_path Path to the ONNX GenAI model directory
 * @param prompt The text prompt for generation
 * @param image_path Path to the image file (JPEG, PNG, etc.), or NULL for
 * text-only
 * @return Generated text on success, or error message prefixed with "ERROR:" on
 * failure The returned string is valid until the next call from the same
 * thread.
 */
FFI_PLUGIN_EXPORT const char *run_inference(const char *model_path,
                                            const char *prompt,
                                            const char *image_path);

/**
 * @brief Run multimodal inference with text and multiple images.
 *
 * This function is designed for vision-language models like Phi-3.5 Vision.
 * It processes the text prompt along with multiple images for generation.
 * The prompt should contain image placeholders like <|image_1|>, <|image_2|>, etc.
 * matching the number of images provided.
 *
 * WARNING: This is a LONG-RUNNING operation!
 * MUST be called from a background Dart Isolate, NOT the main UI isolate.
 *
 * @param model_path Path to the ONNX GenAI model directory
 * @param prompt The text prompt for generation (with image placeholders)
 * @param image_paths Array of paths to image files (JPEG, PNG, etc.)
 * @param image_count Number of images in the array
 * @return Generated text on success, or error message prefixed with "ERROR:" on
 * failure
 */
FFI_PLUGIN_EXPORT const char *run_inference_multi(const char *model_path,
                                                  const char *prompt,
                                                  const char **image_paths,
                                                  int32_t image_count);

/**
 * @brief Run text-only inference with the model.
 *
 * A simplified version of run_inference for text-only generation.
 *
 * WARNING: This is a LONG-RUNNING operation!
 * MUST be called from a background Dart Isolate, NOT the main UI isolate.
 *
 * @param model_path Path to the ONNX GenAI model directory
 * @param prompt The text prompt for generation
 * @param max_length Maximum number of tokens to generate (0 for model default)
 * @return Generated text on success, or error message prefixed with "ERROR:" on
 * failure
 */
FFI_PLUGIN_EXPORT const char *run_text_inference(const char *model_path,
                                                 const char *prompt,
                                                 int32_t max_length);

// =============================================================================
// Configuration API - Runtime Session Options
// =============================================================================

/**
 * @brief Create a configuration object from a model path.
 *
 * The configuration can be modified before loading the model to customize
 * execution providers and session options.
 *
 * @param model_path Path to the ONNX GenAI model directory
 * @return Opaque config handle (int64_t pointer), or 0 on failure
 */
FFI_PLUGIN_EXPORT int64_t create_config(const char *model_path);

/**
 * @brief Destroy a configuration object.
 * @param config_handle Handle returned by create_config
 */
FFI_PLUGIN_EXPORT void destroy_config(int64_t config_handle);

/**
 * @brief Clear all execution providers from the config.
 *
 * Call this before adding custom providers to start fresh.
 *
 * @param config_handle Handle returned by create_config
 * @return 1 on success, negative on failure
 */
FFI_PLUGIN_EXPORT int32_t config_clear_providers(int64_t config_handle);

/**
 * @brief Append an execution provider to the config.
 *
 * Providers are tried in order of insertion. Common providers:
 * - "cpu": Default CPU execution
 * - "XNNPACK": Optimized ARM CPU kernels
 * - "QNN": Qualcomm NPU (Snapdragon only)
 * - "NNAPI": Android Neural Networks API
 * - "CoreML": Apple Neural Engine (iOS only)
 *
 * @param config_handle Handle returned by create_config
 * @param provider_name Name of the execution provider
 * @return 1 on success, negative on failure
 */
FFI_PLUGIN_EXPORT int32_t config_append_provider(int64_t config_handle,
                                                  const char *provider_name);

/**
 * @brief Set an option for a specific execution provider.
 *
 * Common options for CPU provider:
 * - "intra_op_num_threads": Threads within an op (e.g., "4")
 * - "inter_op_num_threads": Threads between ops (e.g., "1")
 *
 * @param config_handle Handle returned by create_config
 * @param provider_name Name of the execution provider
 * @param key Option key
 * @param value Option value (as string)
 * @return 1 on success, negative on failure
 */
FFI_PLUGIN_EXPORT int32_t config_set_provider_option(int64_t config_handle,
                                                      const char *provider_name,
                                                      const char *key,
                                                      const char *value);

/**
 * @brief Run inference using a pre-configured config.
 *
 * This allows using custom execution providers and session options.
 *
 * WARNING: This is a LONG-RUNNING operation!
 * MUST be called from a background Dart Isolate.
 *
 * @param config_handle Handle returned by create_config
 * @param prompt The text prompt for generation
 * @param image_path Path to image file, or NULL for text-only
 * @return Generated text on success, or error message prefixed with "ERROR:"
 */
FFI_PLUGIN_EXPORT const char *run_inference_with_config(int64_t config_handle,
                                                         const char *prompt,
                                                         const char *image_path);

/**
 * @brief Run multi-image inference using a pre-configured config.
 *
 * This allows using custom execution providers and session options with
 * multiple images. The prompt should contain image placeholders like
 * <|image_1|>, <|image_2|>, etc. matching the number of images provided.
 *
 * WARNING: This is a LONG-RUNNING operation!
 * MUST be called from a background Dart Isolate.
 *
 * @param config_handle Handle returned by create_config
 * @param prompt The text prompt for generation (with image placeholders)
 * @param image_paths Array of paths to image files
 * @param image_count Number of images in the array
 * @return Generated text on success, or error message prefixed with "ERROR:"
 */
FFI_PLUGIN_EXPORT const char *run_inference_multi_with_config(int64_t config_handle,
                                                               const char *prompt,
                                                               const char **image_paths,
                                                               int32_t image_count);

/**
 * @brief Get the last error message.
 * @return Error message string, or empty string if no error
 */
FFI_PLUGIN_EXPORT const char *get_last_error();

#ifdef __cplusplus
}
#endif

#endif // FLUTTER_ONNXRUNTIME_GENAI_H
