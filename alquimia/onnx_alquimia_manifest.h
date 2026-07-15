/* -*-  mode: c; c-default-style: "google"; indent-tabs-mode: nil -*- */

#ifndef ALQUIMIA_ONNX_ALQUIMIA_MANIFEST_H_
#define ALQUIMIA_ONNX_ALQUIMIA_MANIFEST_H_

#include <stdbool.h>
#include <stddef.h>

typedef struct {
  char *tensor;
  size_t element;
  char *feature;
  char *field;
  int index;
} OnnxAlquimiaInputMappingSpec;

typedef struct {
  char *tensor;
  size_t element;
  char *field;
  int index;
} OnnxAlquimiaOutputMappingSpec;

typedef struct {
  char *model_path;
  OnnxAlquimiaInputMappingSpec *inputs;
  size_t num_inputs;
  OnnxAlquimiaOutputMappingSpec *outputs;
  size_t num_outputs;
} OnnxAlquimiaManifest;

bool OnnxAlquimiaLoadManifest(
    const char *path,
    OnnxAlquimiaManifest *manifest,
    char *error_message,
    size_t error_message_size);

void OnnxAlquimiaFreeManifest(OnnxAlquimiaManifest *manifest);

#endif /* ALQUIMIA_ONNX_ALQUIMIA_MANIFEST_H_ */
