#ifndef ONNXRUNTIME_H_
#define ONNXRUNTIME_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Basic ONNX Runtime API declarations for linking purposes
// These are minimal stubs to allow compilation to succeed

typedef void* OrtEnv;
typedef void* OrtSession;
typedef void* OrtSessionOptions;
typedef void* OrtMemoryInfo;
typedef void* OrtValue;
typedef void* OrtAllocator;
typedef void* OrtAllocatorWithDefaultOptions;
typedef void* OrtIoBinding;
typedef void* OrtRunOptions;
typedef void* OrtCustomOpDomain;
typedef void* OrtGraphOptimizationLevel;
typedef void* OrtExecutionMode;
typedef void* OrtSession;
typedef void* OrtStatus;
typedef void* OrtLoggingLevel;
typedef void* OrtTensorTypeAndShapeInfo;

typedef enum {
  ORT_OK = 0,
  ORT_FAIL = 1
} OrtErrorCode;

// Minimal API declarations
ORT_API(OrtStatus*, OrtCreateEnv, (uint32_t logging_level, const char* logid, OrtEnv** out));
ORT_API(OrtStatus*, OrtCreateSession, (OrtEnv* env, const char* model_path, OrtSessionOptions* session_options, OrtSession** out));
ORT_API(OrtStatus*, OrtRunSession, (OrtSession* sess, OrtRunOptions* run_options, const char* const* input_names, const OrtValue* const* input_values, size_t input_count, const char* const* output_names, size_t output_count, OrtValue** output_values));
ORT_API(void, OrtReleaseSession, (OrtSession* sess));
ORT_API(void, OrtReleaseEnv, (OrtEnv* env));

#ifdef __cplusplus
}
#endif

#endif  // ONNXRUNTIME_H_