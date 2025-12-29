/**
 * @file ort_genai_c.h
 * @brief ONNX Runtime GenAI C-API stub header
 * 
 * This is a stub header file documenting the expected C-API from ONNX Runtime GenAI.
 * Replace this file with the official header from:
 * https://github.com/microsoft/onnxruntime-genai/blob/main/src/ort_genai_c.h
 * 
 * The actual header will be available after setting up the git submodule:
 *   native_src/onnxruntime-genai/src/ort_genai_c.h
 */

#ifndef ORT_GENAI_C_H
#define ORT_GENAI_C_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Error Handling
// =============================================================================

typedef struct OgaResult OgaResult;

/** Get the error message from a result. Returns NULL if no error. */
const char* OgaResultGetError(const OgaResult* result);

/** Destroy a result object. */
void OgaDestroyResult(OgaResult* result);

// =============================================================================
// String Handling
// =============================================================================

typedef struct OgaString OgaString;

/** Get the C string pointer from an OgaString. */
const char* OgaStringGetString(const OgaString* string);

/** Destroy a string object. */
void OgaDestroyString(OgaString* string);

// =============================================================================
// Sequences (Token IDs)
// =============================================================================

typedef struct OgaSequences OgaSequences;

/** Get the number of sequences. */
size_t OgaSequencesCount(const OgaSequences* sequences);

/** Get the token count for a specific sequence. */
size_t OgaSequenceGetTokenCount(const OgaSequences* sequences, size_t index);

/** Get the token IDs for a specific sequence. */
const int32_t* OgaSequenceGetTokenData(const OgaSequences* sequences, size_t index);

/** Destroy sequences object. */
void OgaDestroySequences(OgaSequences* sequences);

// =============================================================================
// Model
// =============================================================================

typedef struct OgaModel OgaModel;

/** Create a model from a path. */
OgaResult* OgaCreateModel(const char* config_path, OgaModel** model);

/** Destroy a model. */
void OgaDestroyModel(OgaModel* model);

// =============================================================================
// Tokenizer
// =============================================================================

typedef struct OgaTokenizer OgaTokenizer;

/** Create a tokenizer from a model. */
OgaResult* OgaCreateTokenizer(const OgaModel* model, OgaTokenizer** tokenizer);

/** Encode text to token IDs. */
OgaResult* OgaTokenizerEncode(const OgaTokenizer* tokenizer, const char* text, OgaSequences** sequences);

/** Decode token IDs to text. */
OgaResult* OgaTokenizerDecode(const OgaTokenizer* tokenizer, const int32_t* tokens, size_t token_count, OgaString** string);

/** Destroy a tokenizer. */
void OgaDestroyTokenizer(OgaTokenizer* tokenizer);

// =============================================================================
// Tokenizer Stream (for streaming decode)
// =============================================================================

typedef struct OgaTokenizerStream OgaTokenizerStream;

/** Create a tokenizer stream from a tokenizer. */
OgaResult* OgaCreateTokenizerStream(const OgaTokenizer* tokenizer, OgaTokenizerStream** tokenizer_stream);

/** Decode a single token, returning the incremental string piece. */
OgaResult* OgaTokenizerStreamDecode(OgaTokenizerStream* tokenizer_stream, int32_t token, const char** string);

/** Destroy a tokenizer stream. */
void OgaDestroyTokenizerStream(OgaTokenizerStream* tokenizer_stream);

// =============================================================================
// Generator Parameters
// =============================================================================

typedef struct OgaGeneratorParams OgaGeneratorParams;

/** Create generator parameters from a model. */
OgaResult* OgaCreateGeneratorParams(const OgaModel* model, OgaGeneratorParams** params);

/** Set search option (e.g., max_length, temperature). */
OgaResult* OgaGeneratorParamsSetSearchNumber(OgaGeneratorParams* params, const char* name, double value);

/** Set input IDs for generation. */
OgaResult* OgaGeneratorParamsSetInputIds(OgaGeneratorParams* params, const int32_t* input_ids, size_t input_ids_count, size_t batch_size);

/** Set input sequences for generation. */
OgaResult* OgaGeneratorParamsSetInputSequences(OgaGeneratorParams* params, const OgaSequences* sequences);

/** Destroy generator parameters. */
void OgaDestroyGeneratorParams(OgaGeneratorParams* params);

// =============================================================================
// Generator
// =============================================================================

typedef struct OgaGenerator OgaGenerator;

/** Create a generator from model and parameters. */
OgaResult* OgaCreateGenerator(const OgaModel* model, const OgaGeneratorParams* params, OgaGenerator** generator);

/** Check if generation is complete. */
int OgaGenerator_IsDone(const OgaGenerator* generator);

/** Compute logits for the next token. */
OgaResult* OgaGenerator_ComputeLogits(OgaGenerator* generator);

/** Generate the next token. */
OgaResult* OgaGenerator_GenerateNextToken(OgaGenerator* generator);

/** Get the sequence count. */
size_t OgaGenerator_GetSequenceCount(const OgaGenerator* generator);

/** Get the sequence length for a specific sequence. */
size_t OgaGenerator_GetSequenceLength(const OgaGenerator* generator, size_t index);

/** Get the sequence data. */
const int32_t* OgaGenerator_GetSequenceData(const OgaGenerator* generator, size_t index);

/** Get the last token generated. */
int32_t OgaGenerator_GetLastToken(const OgaGenerator* generator, size_t index);

/** Destroy a generator. */
void OgaDestroyGenerator(OgaGenerator* generator);

// =============================================================================
// MultiModal Processor (for Phi-3.5 Vision, etc.)
// =============================================================================

typedef struct OgaMultiModalProcessor OgaMultiModalProcessor;
typedef struct OgaImages OgaImages;
typedef struct OgaNamedTensors OgaNamedTensors;

/** Create a multimodal processor from a model. */
OgaResult* OgaCreateMultiModalProcessor(const OgaModel* model, OgaMultiModalProcessor** processor);

/** Load images from paths. */
OgaResult* OgaLoadImages(const char* const* image_paths, size_t image_count, OgaImages** images);

/** Load a single image from path. */
OgaResult* OgaLoadImage(const char* image_path, OgaImages** images);

/** Process text and images together. */
OgaResult* OgaMultiModalProcessorProcessImages(
    const OgaMultiModalProcessor* processor,
    const char* prompt,
    const OgaImages* images,
    OgaNamedTensors** named_tensors);

/** Create tokenizer from multimodal processor. */
OgaResult* OgaMultiModalProcessorCreateTokenizer(
    const OgaMultiModalProcessor* processor,
    OgaTokenizer** tokenizer);

/** Set named tensors on generator parameters. */
OgaResult* OgaGeneratorParamsSetInputs(
    OgaGeneratorParams* params,
    const OgaNamedTensors* named_tensors);

/** Destroy multimodal processor. */
void OgaDestroyMultiModalProcessor(OgaMultiModalProcessor* processor);

/** Destroy images. */
void OgaDestroyImages(OgaImages* images);

/** Destroy named tensors. */
void OgaDestroyNamedTensors(OgaNamedTensors* named_tensors);

// =============================================================================
// Utility
// =============================================================================

/** Set the current GPU device ID. */
OgaResult* OgaSetCurrentGpuDeviceId(int device_id);

/** Shutdown the ORT GenAI library. */
void OgaShutdown();

#ifdef __cplusplus
}
#endif

#endif // ORT_GENAI_C_H
