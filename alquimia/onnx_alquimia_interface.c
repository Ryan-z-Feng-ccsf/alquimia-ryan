/* -*-  mode: c; c-default-style: "google"; indent-tabs-mode: nil -*- */

/*
** Alquimia Copyright (c) 2013-2016, The Regents of the University of California,
** through Lawrence Berkeley National Laboratory (subject to receipt of any
** required approvals from the U.S. Dept. of Energy).  All rights reserved.
**
** Alquimia is available under a BSD license. See LICENSE.txt for more
** information.
**
** If you have questions about your rights to use or distribute this software,
** please contact Berkeley Lab's Technology Transfer and Intellectual Property
** Management at TTD@lbl.gov referring to Alquimia (LBNL Ref. 2013-119).
**
** NOTICE.  This software was developed under funding from the U.S. Department
** of Energy.  As such, the U.S. Government has been granted for itself and
** others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide
** license in the Software to reproduce, prepare derivative works, and perform
** publicly and display publicly.  Beginning five (5) years after the date
** permission to assert copyright is obtained from the U.S. Department of Energy,
** and subject to any subsequent five (5) year renewals, the U.S. Government is
** granted for itself and others acting on its behalf a paid-up, nonexclusive,
** irrevocable, worldwide license in the Software to reproduce, prepare derivative
** works, distribute copies to the public, perform publicly and display publicly,
** and to permit others to do so.
*/

#include "alquimia/onnx_alquimia_interface.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ONNX Runtime's C header trips -Wpedantic in Alquimia's -Werror builds.
** Keep the diagnostic suppression scoped to the vendor include. */
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

#include <onnxruntime_c_api.h>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include "alquimia/alquimia_constants.h"
#include "alquimia/alquimia_util.h"
#include "alquimia/onnx_alquimia_manifest.h"

#if ALQUIMIA_HAVE_ONNX

/* Mapping struct for the AlquimiaState*/
typedef enum {
  ALQUIMIA_STRUCT_WATER_DENSITY,                /* state->water_density */
  ALQUIMIA_STRUCT_POROSITY,                     /* state->porosity */
  ALQUIMIA_STRUCT_TEMPERATURE,                  /* state->temperature */
  ALQUIMIA_STRUCT_AQUEOUS_PRESSURE,             /* state->aqueous_pressure */
  ALQUIMIA_STRUCT_TOTAL_MOBILE,                 /* state->total_mobile.data */
  ALQUIMIA_STRUCT_TOTAL_IMMOBILE,               /* state->total_immobile.data */
  ALQUIMIA_STRUCT_MINERAL_VOLUME_FRACTION,       /* state->mineral_volume_fraction.data */
  ALQUIMIA_STRUCT_MINERAL_SPECIFIC_SURFACE_AREA, /* state->mineral_specific_surface_area.data */
  ALQUIMIA_STRUCT_SURFACE_SITE_DENSITY,         /* state->surface_site_density.data */
  ALQUIMIA_STRUCT_CATION_EXCHANGE_CAPACITY,     /* state->cation_exchange_capacity.data */
  ALQUIMIA_STRUCT_GAS_CONCENTRATION             /* state->gas_concentration.data */
} AlquimiaMappedStruct;

typedef struct {
  /* AlquimiaState mobile, immobile, etc.*/
  AlquimiaMappedStruct alquimia_state;
  /* The index for the AlquimiaState.
  ** Remain 0 for scalar(water_density, porosity, temperature, aqueous_pressure)
  */
  /* Align with the alquimia_state */
  int alquimia_state_index;
  /* Adapter-owned copy retained after the manifest is released. Output
  ** mappings leave this NULL because only inputs have condition names. 
  */
  char *feature;
} FeatureMapping;

typedef struct
{
  /* OnnxRuntime API */
  /* Const pointer to the global OrtApi function table. Do NOT release. */
  const OrtApi *g_ort;
  OrtEnv *env;  /* Released by ReleaseEnv() */
  OrtSessionOptions *session_options; /* Released by ReleaseSessionOptions() */
  OrtSession *session; /* Released by ReleaseSession() */
  OrtMemoryInfo *memory_info; /* Released by ReleaseMemoryInfo() */
  OrtAllocator *allocator;  /* Released by ReleaseAllocator() */
  /* Setup owns this temporary representation and releases it before
  ** publishing a successfully initialized engine. */
  OnnxAlquimiaManifest manifest;  /* Released by OnnxAlquimiaFreeManifest()*/

  /* Dynamic input info */
  /* Align with num_inputs in manifest */
  size_t num_inputs;
  /* Align with the tensor in manifest */
  char **input_names; /* Released by OrtAllocator allocator */       
  /* Number of dimensions for each input tensor 
  ** 0 for a scalar tensor
  */
  size_t *input_num_dim;
  /* [batch size, input features] / [input features] */
  /* batch size == -1 means it's dynamic */
  /* Considering the architecture of Alquimia 
  ** Set it as 1
  */
  int64_t **input_dim_values; /* Released by free() */
  /* Each input tensor has their own size */
  size_t *input_total_size; /* Released by free() */
  /* The real numbers for the input tensor */
  double **input_data;  /* Released by free() */
  OrtValue **input_tensor;  /* Released by ReleaseValue() */

  /* Dynamic output info */
  /* Align with num_outputs in manifest */
  size_t num_outputs;
  /* Align with the tensor in manifest */
  char **output_names;  /* Released by OrtAllocator allocator */
  /* Number of dimensions for each output tensor */
  size_t *output_num_dim; /* Released by free() */
  /* [batch size, input features] / [input features] */
  int64_t **output_dim_values;  /* Released by free() */
  /* Each output tensor has their own size */
  size_t *output_total_size;  /* Released by free() */
  /* The real numbers for the output tensor */
  double **output_data; /* Released by free() */
  OrtValue **output_tensor; /* Released by ReleaseValue() */

  /* Flatten non-sequential input/output tensors into 1-dimensional arrays. */
  size_t total_flat_inputs;
  size_t total_flat_outputs;
  FeatureMapping *input_mappings;  /* Array of size total_flat_inputs */
  FeatureMapping *output_mappings; /* Array of size total_flat_outputs */
} OnnxEngineState;

/**
 * @brief Converts an ONNX Runtime status into an Alquimia engine status.
 * @param g_ort ONNX Runtime API used to inspect and release @p status.
 * @param status ONNX status returned by an API call; NULL indicates success.
 * @param alquimia_status Destination for the translated error and message.
 * @return True on success. On failure, releases @p status, records an engine
 *         integrity error, and returns false.
 */
static bool CheckStatus(const OrtApi *g_ort, OrtStatus *status, AlquimiaEngineStatus *alquimia_status)
{
  /* A non-NULL OrtStatus indicates an API failure; extract the error, 
  ** populate our status, and explicitly release the status object. 
  */
  if (status != NULL)
  if (status != NULL)
  {
    const char *msg = g_ort->GetErrorMessage(status);
    alquimia_status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(alquimia_status->message, kAlquimiaMaxStringLength, "ONNX Runtime Error: %s", msg);
    g_ort->ReleaseStatus(status);
    return false;
  }
  return true;
}

/**
 * @brief Maps an exact manifest state-variable name to an AlquimiaState field.
 * @param name Case-sensitive alquimia_state value from the manifest.
 * @param target Receives the corresponding mapping enum on success and remains
 *        unchanged when @p name is unsupported.
 * @return True when @p name identifies a supported scalar or vector field.
 */
static bool ParseStructName(const char *name, AlquimiaMappedStruct *target) {
  /* For the vector in the AlquimiaState */
  if (strcmp(name, "total_mobile") == 0) {
    *target = ALQUIMIA_STRUCT_TOTAL_MOBILE;
  } else if (strcmp(name, "total_immobile") == 0) {
    *target = ALQUIMIA_STRUCT_TOTAL_IMMOBILE;
  } else if (strcmp(name, "mineral_volume_fraction") == 0) {
    *target = ALQUIMIA_STRUCT_MINERAL_VOLUME_FRACTION;
  } else if (strcmp(name, "mineral_specific_surface_area") == 0) {
    *target = ALQUIMIA_STRUCT_MINERAL_SPECIFIC_SURFACE_AREA;
  } else if (strcmp(name, "surface_site_density") == 0) {
    *target = ALQUIMIA_STRUCT_SURFACE_SITE_DENSITY;
  } else if (strcmp(name, "cation_exchange_capacity") == 0) {
    *target = ALQUIMIA_STRUCT_CATION_EXCHANGE_CAPACITY;
  } else if (strcmp(name, "porosity") == 0) {
    *target = ALQUIMIA_STRUCT_POROSITY;
  } else if (strcmp(name, "temperature") == 0) {
    *target = ALQUIMIA_STRUCT_TEMPERATURE;
  } else if (strcmp(name, "aqueous_pressure") == 0) {
    *target = ALQUIMIA_STRUCT_AQUEOUS_PRESSURE;
  } else if (strcmp(name, "water_density") == 0) {
    *target = ALQUIMIA_STRUCT_WATER_DENSITY;
  } else if (strcmp(name, "gas_concentration") == 0) {
    *target = ALQUIMIA_STRUCT_GAS_CONCENTRATION;
  } else {
    return false;
  }
  return true;
}

/**
 * @brief Identifies mappings whose destination is an AlquimiaState scalar.
 * @param alquimia_state Mapping destination to classify.
 * @return True for scalar state fields, whose required mapping index is zero.
 */
static bool IsScalarMapping(AlquimiaMappedStruct alquimia_state)
{
  /* Specifically for the water_density, porosity, temperature, aqueous_pressure */
  return alquimia_state == ALQUIMIA_STRUCT_WATER_DENSITY ||
         alquimia_state == ALQUIMIA_STRUCT_POROSITY ||
         alquimia_state == ALQUIMIA_STRUCT_TEMPERATURE ||
         alquimia_state == ALQUIMIA_STRUCT_AQUEOUS_PRESSURE;
}

/**
 * @brief Selects the problem-metadata name vector for a state mapping.
 * @param meta_data Problem metadata whose name vectors were allocated from the
 *        model sizes.
 * @param alquimia_state State field associated with a model feature.
 * @return The corresponding name vector, or NULL when the mapped state field
 *         has no name representation in AlquimiaProblemMetaData.
 */
static AlquimiaVectorString *MetadataNamesForMapping(
    AlquimiaProblemMetaData *meta_data,
    AlquimiaMappedStruct alquimia_state)
{
  switch (alquimia_state)
  {
  case ALQUIMIA_STRUCT_TOTAL_MOBILE:
  case ALQUIMIA_STRUCT_TOTAL_IMMOBILE:
    return &meta_data->primary_names;
  case ALQUIMIA_STRUCT_MINERAL_VOLUME_FRACTION:
  case ALQUIMIA_STRUCT_MINERAL_SPECIFIC_SURFACE_AREA:
    return &meta_data->mineral_names;
  case ALQUIMIA_STRUCT_SURFACE_SITE_DENSITY:
    return &meta_data->surface_site_names;
  case ALQUIMIA_STRUCT_CATION_EXCHANGE_CAPACITY:
    return &meta_data->ion_exchange_names;
  case ALQUIMIA_STRUCT_GAS_CONCENTRATION:
    return &meta_data->gas_names;
  default:
    return NULL;
  }
}

/**
 * @brief Groups state fields that share one problem-metadata name vector.
 * @param alquimia_state State destination to classify.
 * @return A stable category identifier, or -1 for scalar fields without names.
 *
 * Mobile and immobile totals share primary_names, and both mineral vectors
 * share mineral_names. Setup uses these categories to reject different feature
 * names that AlquimiaProblemMetaData could not represent independently.
 */
static int MetadataNameCategory(AlquimiaMappedStruct alquimia_state)
{
  switch (alquimia_state)
  {
  case ALQUIMIA_STRUCT_TOTAL_MOBILE:
  case ALQUIMIA_STRUCT_TOTAL_IMMOBILE:
    return 0;
  case ALQUIMIA_STRUCT_MINERAL_VOLUME_FRACTION:
  case ALQUIMIA_STRUCT_MINERAL_SPECIFIC_SURFACE_AREA:
    return 1;
  case ALQUIMIA_STRUCT_SURFACE_SITE_DENSITY:
    return 2;
  case ALQUIMIA_STRUCT_CATION_EXCHANGE_CAPACITY:
    return 3;
  case ALQUIMIA_STRUCT_GAS_CONCENTRATION:
    return 4;
  default:
    return -1;
  }
}

/**
 * @brief Copies an input feature name into an allocated Alquimia name vector.
 * @param names Destination name vector.
 * @param index Zero-based destination index.
 * @param value Null-terminated feature name to copy.
 */
static void StoreMetadataName(
    AlquimiaVectorString *names,
    int index,
    const char *value)
{
  if (names == NULL || names->data == NULL || index < 0 || index >= names->size)
  {
    return;
  }

  strncpy(names->data[index], value, kAlquimiaMaxStringLength - 1);
  names->data[index][kAlquimiaMaxStringLength - 1] = '\0';
}

/**
 * @brief Copies a manifest feature name into adapter-owned storage.
 * @param name Null-terminated feature name to copy.
 * @return A newly allocated copy, or NULL on allocation failure.
 *
 * The returned string outlives the parsed manifest and is released with the
 * input mapping array during engine shutdown.
 */
static char *CopyFeatureName(const char *name)
{
  size_t length = strlen(name);
  char *copy = (char *)malloc(length + 1);
  if (copy != NULL)
  {
    memcpy(copy, name, length + 1);
  }
  return copy;
}

/**
 * @brief Converts one validated manifest destination into a runtime mapping.
 * @param alquimia_state Case-sensitive AlquimiaState variable name.
 * @param alquimia_state_index Nonnegative state-vector index from the manifest.
 * @param mapping Receives the runtime destination on success.
 * @param status Receives an engine-integrity error on incompatibility.
 * @return True when the state variable and index can be represented safely.
 *
 * Scalar state variables require index zero. INT_MAX is rejected for vectors
 * so the later size calculation cannot overflow an Alquimia int.
 */
static bool ParseManifestMapping(
    const char *alquimia_state,
    int alquimia_state_index,
    FeatureMapping *mapping,
    AlquimiaEngineStatus *status)
{
  if (!ParseStructName(alquimia_state, &mapping->alquimia_state))
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Unsupported AlquimiaState variable '%s' in ONNX manifest.",
             alquimia_state);
    return false;
  }
  if ((IsScalarMapping(mapping->alquimia_state) &&
       alquimia_state_index != 0) ||
      (!IsScalarMapping(mapping->alquimia_state) &&
       alquimia_state_index == INT_MAX))
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "ONNX manifest AlquimiaState index %d is incompatible with "
             "variable '%s'.", alquimia_state_index, alquimia_state);
    return false;
  }
  mapping->alquimia_state_index = alquimia_state_index;
  return true;
}

/**
 * @brief Expands the AlquimiaSizes category required by one mapping.
 * @param mapping Validated state destination.
 * @param sizes Accumulator initialized to zero before the first mapping.
 *
 * Vector capacity is the highest mapped index plus one. Fields backed by the
 * same Alquimia dimension, notably the two mineral vectors, update one shared
 * size entry.
 */
static void UpdateSizesForMapping(
    const FeatureMapping *mapping,
    AlquimiaSizes *sizes)
{
  int required_size = mapping->alquimia_state_index + 1;
  int *size = NULL;

  switch (mapping->alquimia_state)
  {
  case ALQUIMIA_STRUCT_TOTAL_MOBILE:
    size = &sizes->num_primary;
    break;
  case ALQUIMIA_STRUCT_TOTAL_IMMOBILE:
    size = &sizes->num_sorbed;
    break;
  case ALQUIMIA_STRUCT_MINERAL_VOLUME_FRACTION:
  case ALQUIMIA_STRUCT_MINERAL_SPECIFIC_SURFACE_AREA:
    size = &sizes->num_minerals;
    break;
  case ALQUIMIA_STRUCT_SURFACE_SITE_DENSITY:
    size = &sizes->num_surface_sites;
    break;
  case ALQUIMIA_STRUCT_CATION_EXCHANGE_CAPACITY:
    size = &sizes->num_ion_exchange_sites;
    break;
  case ALQUIMIA_STRUCT_GAS_CONCENTRATION:
    size = &sizes->num_gases;
    break;
  default:
    return;
  }
  if (*size < required_size)
  {
    *size = required_size;
  }
}

/**
 * @brief Resolves a tensor element index to the flat runtime mapping index.
 * @param tensor Case-sensitive tensor name from the manifest.
 * @param tensor_element_index Zero-based flattened index within that tensor.
 * @param num_tensors Number of model tensors in @p names and @p tensor_sizes.
 * @param names ONNX Runtime tensor names in session order.
 * @param tensor_sizes Flattened element counts in session order.
 * @param flat_index Receives the offset into the combined mapping array.
 * @param status Receives an error for unknown, duplicate, or undersized tensors.
 * @return True when exactly one tensor matches and the index is in range.
 *
 * Reaction steps iterate tensors in session order, so this same prefix-sum
 * layout must be used while constructing the mapping arrays.
 */
static bool FindFlatTensorElement(
    const char *tensor,
    size_t tensor_element_index,
    size_t num_tensors,
    char *const *names,
    const size_t *tensor_sizes,
    size_t *flat_index,
    AlquimiaEngineStatus *status)
{
  size_t i;
  size_t offset = 0;
  size_t matching_index = num_tensors;

  for (i = 0; i < num_tensors; ++i)
  {
    if (strcmp(tensor, names[i]) == 0)
    {
      if (matching_index != num_tensors)
      {
        status->error = kAlquimiaErrorEngineIntegrity;
        snprintf(status->message, kAlquimiaMaxStringLength,
                 "ONNX model contains duplicate tensor name '%s'.", tensor);
        return false;
      }
      matching_index = i;
      *flat_index = offset + tensor_element_index;
    }
    offset += tensor_sizes[i];
  }
  if (matching_index == num_tensors)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "ONNX manifest references unknown tensor '%s'.", tensor);
    return false;
  }
  if (tensor_element_index >= tensor_sizes[matching_index])
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "ONNX manifest tensor element index %zu is out of range for "
             "tensor '%s' of size %zu.", tensor_element_index, tensor,
             tensor_sizes[matching_index]);
    return false;
  }
  return true;
}

/**
 * @brief Clears all dimensions before deriving them solely from the manifest.
 * @param sizes Size structure returned through the generic engine interface.
 */
static void InitializeSizes(AlquimiaSizes *sizes)
{
  memset(sizes, 0, sizeof(*sizes));
}

/**
 * @brief Builds complete flat input/output mappings from the parsed manifest.
 * @param onnx_state Inspected model state and destination for runtime mappings.
 * @param sizes Receives dimensions derived from the highest mapped indices.
 * @param status Receives allocation or semantic-validation errors.
 * @return True only when every tensor element index has exactly one mapping.
 *
 * Input feature names are copied before the manifest is released. The seen
 * arrays enforce complete coverage and prevent duplicate tensor indices;
 * name-category checks preserve the metadata representation invariant.
 */
static bool BuildManifestMappings(
    OnnxEngineState *onnx_state,
    AlquimiaSizes *sizes,
    AlquimiaEngineStatus *status)
{
  /* Check if there is duplicate value in the JSON */
  bool *input_seen;
  bool *output_seen;
  size_t i;

  /* Memory Allocation */
  onnx_state->input_mappings = (FeatureMapping *)calloc(
      onnx_state->total_flat_inputs, sizeof(*onnx_state->input_mappings));
  onnx_state->output_mappings = (FeatureMapping *)calloc(
      onnx_state->total_flat_outputs, sizeof(*onnx_state->output_mappings));
  input_seen = (bool *)calloc(onnx_state->total_flat_inputs, sizeof(bool));
  output_seen = (bool *)calloc(onnx_state->total_flat_outputs, sizeof(bool));
  if (onnx_state->input_mappings == NULL ||
      onnx_state->output_mappings == NULL ||
      input_seen == NULL || output_seen == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Memory allocation failed for ONNX manifest mappings.");
    free(input_seen);
    free(output_seen);
    return false;
  }
  /* Initialize AlquimiaSize */
  InitializeSizes(sizes);

  /* For every input tensor */
  for (i = 0; i < onnx_state->manifest.num_inputs; ++i)
  {
    const OnnxAlquimiaInputMappingSpec *spec = &onnx_state->manifest.inputs[i];
    FeatureMapping *mapping;
    size_t flat_index;
    size_t j;

    if (!FindFlatTensorElement(spec->tensor, spec->tensor_element_index,
                               onnx_state->num_inputs,
                               onnx_state->input_names,
                               onnx_state->input_total_size,
                               &flat_index, status))
    {
      free(input_seen);
      free(output_seen);
      return false;
    }
    if (input_seen[flat_index])
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "Duplicate ONNX input mapping for tensor '%s' element index %zu.",
               spec->tensor, spec->tensor_element_index);
      free(input_seen);
      free(output_seen);
      return false;
    }
    mapping = &onnx_state->input_mappings[flat_index];
    if (!ParseManifestMapping(spec->alquimia_state,
                              spec->alquimia_state_index, mapping, status))
    {
      free(input_seen);
      free(output_seen);
      return false;
    }
    for (j = 0; j < onnx_state->total_flat_inputs; ++j)
    {
      const FeatureMapping *other = &onnx_state->input_mappings[j];
      if (input_seen[j] &&
          MetadataNameCategory(other->alquimia_state) >= 0 &&
          MetadataNameCategory(other->alquimia_state) ==
              MetadataNameCategory(mapping->alquimia_state) &&
          other->alquimia_state_index == mapping->alquimia_state_index &&
          strcmp(other->feature, spec->feature) != 0)
      {
        status->error = kAlquimiaErrorEngineIntegrity;
        snprintf(status->message, kAlquimiaMaxStringLength,
                 "Conflicting ONNX feature names '%s' and '%s' for "
                 "AlquimiaState variable '%s' index %d.",
                 other->feature, spec->feature, spec->alquimia_state,
                 spec->alquimia_state_index);
        free(input_seen);
        free(output_seen);
        return false;
      }
    }
    mapping->feature = CopyFeatureName(spec->feature);
    if (mapping->feature == NULL)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "Memory allocation failed for ONNX input feature name.");
      free(input_seen);
      free(output_seen);
      return false;
    }
    input_seen[flat_index] = true;
    UpdateSizesForMapping(mapping, sizes);
  }

  for (i = 0; i < onnx_state->manifest.num_outputs; ++i)
  {
    const OnnxAlquimiaOutputMappingSpec *spec = &onnx_state->manifest.outputs[i];
    FeatureMapping *mapping;
    size_t flat_index;

    if (!FindFlatTensorElement(spec->tensor, spec->tensor_element_index,
                               onnx_state->num_outputs,
                               onnx_state->output_names,
                               onnx_state->output_total_size,
                               &flat_index, status))
    {
      free(input_seen);
      free(output_seen);
      return false;
    }
    if (output_seen[flat_index])
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "Duplicate ONNX output mapping for tensor '%s' element index %zu.",
               spec->tensor, spec->tensor_element_index);
      free(input_seen);
      free(output_seen);
      return false;
    }
    mapping = &onnx_state->output_mappings[flat_index];
    if (!ParseManifestMapping(spec->alquimia_state,
                              spec->alquimia_state_index, mapping, status))
    {
      free(input_seen);
      free(output_seen);
      return false;
    }
    output_seen[flat_index] = true;
    UpdateSizesForMapping(mapping, sizes);
  }

  for (i = 0; i < onnx_state->total_flat_inputs; ++i)
  {
    if (!input_seen[i])
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "ONNX manifest does not map every input tensor element index.");
      free(input_seen);
      free(output_seen);
      return false;
    }
  }
  for (i = 0; i < onnx_state->total_flat_outputs; ++i)
  {
    if (!output_seen[i])
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "ONNX manifest does not map every output tensor element index.");
      free(input_seen);
      free(output_seen);
      return false;
    }
  }

  free(input_seen);
  free(output_seen);
  return true;
}

/**
 * @brief Reads one mapped scalar or vector element from an AlquimiaState.
 * @param state State containing the model input value.
 * @param map Validated destination field and zero-based vector index.
 * @param status Receives an engine integrity error for an unknown field, NULL
 *        vector storage, or an out-of-bounds vector index.
 * @return The mapped value on success, or 0.0 after recording an error.
 */
static double GetAlquimiaValue(
    const AlquimiaState *state,
    FeatureMapping map,
    AlquimiaEngineStatus *status)
{
  switch (map.alquimia_state)
  {
  case ALQUIMIA_STRUCT_TOTAL_MOBILE:
    if (state->total_mobile.data == NULL || map.alquimia_state_index >= state->total_mobile.size)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds total_mobile access: index %d, size %d.", map.alquimia_state_index, state->total_mobile.size);
      return 0.0;
    }
    return state->total_mobile.data[map.alquimia_state_index];

  case ALQUIMIA_STRUCT_TOTAL_IMMOBILE:
    if (state->total_immobile.data == NULL || map.alquimia_state_index >= state->total_immobile.size)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds total_immobile access: index %d, size %d.", map.alquimia_state_index, state->total_immobile.size);
      return 0.0;
    }
    return state->total_immobile.data[map.alquimia_state_index];

  case ALQUIMIA_STRUCT_MINERAL_VOLUME_FRACTION:
      if (state->mineral_volume_fraction.data == NULL || map.alquimia_state_index >= state->mineral_volume_fraction.size) {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds mineral_volume_fraction access: index %d, size %d.", map.alquimia_state_index, state->mineral_volume_fraction.size);
      return 0.0;
    }
    return state->mineral_volume_fraction.data[map.alquimia_state_index];

  case ALQUIMIA_STRUCT_MINERAL_SPECIFIC_SURFACE_AREA:
      if (state->mineral_specific_surface_area.data == NULL || map.alquimia_state_index >= state->mineral_specific_surface_area.size) {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds mineral_specific_surface_area access: index %d, size %d.", map.alquimia_state_index, state->mineral_specific_surface_area.size);
      return 0.0;
    }
    return state->mineral_specific_surface_area.data[map.alquimia_state_index];

  case ALQUIMIA_STRUCT_SURFACE_SITE_DENSITY:
      if (state->surface_site_density.data == NULL || map.alquimia_state_index >= state->surface_site_density.size) {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds surface_site_density access: index %d, size %d.", map.alquimia_state_index, state->surface_site_density.size);
      return 0.0;
    }
    return state->surface_site_density.data[map.alquimia_state_index];

  case ALQUIMIA_STRUCT_CATION_EXCHANGE_CAPACITY:
      if (state->cation_exchange_capacity.data == NULL || map.alquimia_state_index >= state->cation_exchange_capacity.size) {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds cation_exchange_capacity access: index %d, size %d.", map.alquimia_state_index, state->cation_exchange_capacity.size);
      return 0.0;
    }
    return state->cation_exchange_capacity.data[map.alquimia_state_index];

  case ALQUIMIA_STRUCT_POROSITY:
    return state->porosity;

  case ALQUIMIA_STRUCT_TEMPERATURE:
    return state->temperature;

  case ALQUIMIA_STRUCT_AQUEOUS_PRESSURE:
    return state->aqueous_pressure;

  case ALQUIMIA_STRUCT_WATER_DENSITY:
    return state->water_density;

  case ALQUIMIA_STRUCT_GAS_CONCENTRATION:
      if (state->gas_concentration.data == NULL || map.alquimia_state_index >= state->gas_concentration.size) {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds gas_concentration access: index %d, size %d.", map.alquimia_state_index, state->gas_concentration.size);
      return 0.0;
    }
    return state->gas_concentration.data[map.alquimia_state_index];

  default:
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Unknown mapped struct type: %d.", map.alquimia_state);
    return 0.0;
  }
}

/**
 * @brief Writes one model output to a mapped AlquimiaState destination.
 * @param state State that receives the model output value.
 * @param map Validated destination field and zero-based vector index.
 * @param value Model output to assign.
 * @param status Receives an engine integrity error for an unknown field, NULL
 *        vector storage, or an out-of-bounds vector index.
 */
static void SetAlquimiaValue(
    AlquimiaState *state,
    FeatureMapping map,
    double value,
    AlquimiaEngineStatus *status)
{
  switch (map.alquimia_state)
  {
  case ALQUIMIA_STRUCT_TOTAL_MOBILE:
      if (state->total_mobile.data == NULL || map.alquimia_state_index >= state->total_mobile.size) {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds total_mobile write: index %d, size %d.", map.alquimia_state_index, state->total_mobile.size);
      return;
    }
    state->total_mobile.data[map.alquimia_state_index] = value;
    break;

  case ALQUIMIA_STRUCT_TOTAL_IMMOBILE:
      if (state->total_immobile.data == NULL || map.alquimia_state_index >= state->total_immobile.size) {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds total_immobile write: index %d, size %d.", map.alquimia_state_index, state->total_immobile.size);
      return;
    }
    state->total_immobile.data[map.alquimia_state_index] = value;
    break;

  case ALQUIMIA_STRUCT_MINERAL_VOLUME_FRACTION:
    if (state->mineral_volume_fraction.data == NULL || map.alquimia_state_index >= state->mineral_volume_fraction.size)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds mineral_volume_fraction write: index %d, size %d.", map.alquimia_state_index, state->mineral_volume_fraction.size);
      return;
    }
    state->mineral_volume_fraction.data[map.alquimia_state_index] = value;
    break;

  case ALQUIMIA_STRUCT_MINERAL_SPECIFIC_SURFACE_AREA:
    if (state->mineral_specific_surface_area.data == NULL || map.alquimia_state_index >= state->mineral_specific_surface_area.size)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds mineral_specific_surface_area write: index %d, size %d.", map.alquimia_state_index, state->mineral_specific_surface_area.size);
      return;
    }
    state->mineral_specific_surface_area.data[map.alquimia_state_index] = value;
    break;

  case ALQUIMIA_STRUCT_SURFACE_SITE_DENSITY:
    if (state->surface_site_density.data == NULL || map.alquimia_state_index >= state->surface_site_density.size)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds surface_site_density write: index %d, size %d.", map.alquimia_state_index, state->surface_site_density.size);
      return;
    }
    state->surface_site_density.data[map.alquimia_state_index] = value;
    break;

  case ALQUIMIA_STRUCT_CATION_EXCHANGE_CAPACITY:
    if (state->cation_exchange_capacity.data == NULL || map.alquimia_state_index >= state->cation_exchange_capacity.size)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds cation_exchange_capacity write: index %d, size %d.", map.alquimia_state_index, state->cation_exchange_capacity.size);
      return;
    }
    state->cation_exchange_capacity.data[map.alquimia_state_index] = value;
    break;

  case ALQUIMIA_STRUCT_POROSITY:
    state->porosity = value;
    break;

  case ALQUIMIA_STRUCT_TEMPERATURE:
    state->temperature = value;
    break;

  case ALQUIMIA_STRUCT_AQUEOUS_PRESSURE:
    state->aqueous_pressure = value;
    break;

  case ALQUIMIA_STRUCT_WATER_DENSITY:
    state->water_density = value;
    break;

  case ALQUIMIA_STRUCT_GAS_CONCENTRATION:
    if (state->gas_concentration.data == NULL || map.alquimia_state_index >= state->gas_concentration.size)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds gas_concentration write: index %d, size %d.", map.alquimia_state_index, state->gas_concentration.size);
      return;
    }
    state->gas_concentration.data[map.alquimia_state_index] = value;
    break;

  default:
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Unknown mapped struct type: %d.", map.alquimia_state);
    break;
  }
}

/**
 * @brief Releases a partially initialized engine without replacing setup error details.
 * @param onnx_state Address of the local engine pointer. Shutdown releases all
 *        resources initialized so far and sets this pointer to NULL.
 *
 * A stack-backed temporary status satisfies the shutdown contract while the
 * caller's original AlquimiaEngineStatus retains the setup failure.
 */
static void CleanupOnSetupFailure(OnnxEngineState **onnx_state)
{
  char temp_message[kAlquimiaMaxStringLength];
  AlquimiaEngineStatus temp_status = {0};
  temp_status.message = temp_message;
  onnx_alquimia_shutdown(onnx_state, &temp_status);
}

/**
 * @brief Initializes an ONNX engine from a versioned JSON sidecar manifest.
 * @param input_filename Manifest path; relative model paths resolve from its
 *        directory.
 * @param hands_off Unused by this adapter.
 * @param onnx_engine_state Address that receives the initialized engine.
 * @param sizes Receives state-vector sizes derived from explicit mappings.
 * @param functionality Receives the ONNX adapter capability flags.
 * @param status Receives setup errors without being overwritten by cleanup.
 *
 * The destination engine pointer is set to NULL before resource acquisition and
 * remains NULL on every failure. Parsed manifest storage is released before a
 * successful engine is published.
 */
void onnx_alquimia_setup(
    const char *input_filename,
    bool hands_off,
    void *onnx_engine_state,
    AlquimiaSizes *sizes,
    AlquimiaEngineFunctionality *functionality,
    AlquimiaEngineStatus *status)
{
  OnnxEngineState *onnx_state;
  OrtStatus *ort_status;
  FILE *f;
  size_t i, j;

  status->error = kAlquimiaNoError;
  status->message[0] = '\0';

  // Unused
  (void)hands_off;

  if (onnx_engine_state == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Invalid ONNX engine state destination in setup.");
    return;
  }
  *(OnnxEngineState **)onnx_engine_state = NULL;

  onnx_state = (OnnxEngineState *)calloc(1, sizeof(OnnxEngineState));
  if (onnx_state == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Memory allocation failed for OnnxEngineState.");
    return;
  }
  /* Parse the JSON to the manifest */
  if (!OnnxAlquimiaLoadManifest(
          input_filename, &onnx_state->manifest, status->message,
          kAlquimiaMaxStringLength))
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    free(onnx_state);
    return;
  }
  f = fopen(onnx_state->manifest.model_path, "rb");
  if (f == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "ONNX model file not found: %s",
             onnx_state->manifest.model_path);
    OnnxAlquimiaFreeManifest(&onnx_state->manifest);
    free(onnx_state);
    return;
  }
  fclose(f);

  onnx_state->g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (!onnx_state->g_ort)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Failed to load ONNX Runtime API.");
    OnnxAlquimiaFreeManifest(&onnx_state->manifest);
    free(onnx_state);
    return;
  }

  ort_status = onnx_state->g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "onnx_alquimia_engine", &onnx_state->env);
  if (!CheckStatus(onnx_state->g_ort, ort_status, status))
  {
    OnnxAlquimiaFreeManifest(&onnx_state->manifest);
    free(onnx_state);
    return;
  }

  ort_status = onnx_state->g_ort->CreateSessionOptions(&onnx_state->session_options);
  if (!CheckStatus(onnx_state->g_ort, ort_status, status))
  {
    onnx_state->g_ort->ReleaseEnv(onnx_state->env);
    OnnxAlquimiaFreeManifest(&onnx_state->manifest);
    free(onnx_state);
    return;
  }

  ort_status = onnx_state->g_ort->CreateSession(
      onnx_state->env, onnx_state->manifest.model_path,
      onnx_state->session_options, &onnx_state->session);
  if (!CheckStatus(onnx_state->g_ort, ort_status, status))
  {
    onnx_state->g_ort->ReleaseSessionOptions(onnx_state->session_options);
    onnx_state->g_ort->ReleaseEnv(onnx_state->env);
    OnnxAlquimiaFreeManifest(&onnx_state->manifest);
    free(onnx_state);
    return;
  }

  ort_status = onnx_state->g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &onnx_state->memory_info);
  if (!CheckStatus(onnx_state->g_ort, ort_status, status))
  {
    CleanupOnSetupFailure(&onnx_state);
    return;
  }

  ort_status = onnx_state->g_ort->CreateAllocator(onnx_state->session, onnx_state->memory_info, &onnx_state->allocator);
  if (!CheckStatus(onnx_state->g_ort, ort_status, status))
  {
    CleanupOnSetupFailure(&onnx_state);
    return;
  }

  /* Query input/output tensor count */
  size_t num_inputs = 0;
  ort_status = onnx_state->g_ort->SessionGetInputCount(onnx_state->session, &num_inputs);
  if (!CheckStatus(onnx_state->g_ort, ort_status, status))
  {
    CleanupOnSetupFailure(&onnx_state);
    return;
  }
  onnx_state->num_inputs = num_inputs;

  size_t num_outputs = 0;
  ort_status = onnx_state->g_ort->SessionGetOutputCount(onnx_state->session, &num_outputs);
  if (!CheckStatus(onnx_state->g_ort, ort_status, status))
  {
    CleanupOnSetupFailure(&onnx_state);
    return;
  }
  onnx_state->num_outputs = num_outputs;

  /* Guard against empty inputs or outputs */
  if (num_inputs == 0 || num_outputs == 0)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Model has 0 inputs or outputs (inputs: %d, outputs: %d).", (int)num_inputs, (int)num_outputs);
    CleanupOnSetupFailure(&onnx_state);
    return;
  }

  /* Allocate outer arrays for inputs */
  onnx_state->input_names = (char **)calloc(num_inputs, sizeof(char *));
  onnx_state->input_num_dim = (size_t *)calloc(num_inputs, sizeof(size_t));
  onnx_state->input_dim_values = (int64_t **)calloc(num_inputs, sizeof(int64_t *));
  onnx_state->input_total_size = (size_t *)calloc(num_inputs, sizeof(size_t));
  onnx_state->input_data = (double **)calloc(num_inputs, sizeof(double *));
  onnx_state->input_tensor = (OrtValue **)calloc(num_inputs, sizeof(OrtValue *));

  /* Guard */
  if (onnx_state->input_names == NULL || onnx_state->input_num_dim == NULL ||
      onnx_state->input_dim_values == NULL || onnx_state->input_total_size == NULL ||
      onnx_state->input_data == NULL || onnx_state->input_tensor == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Memory allocation failed for input container arrays.");
    CleanupOnSetupFailure(&onnx_state);
    return;
  }

  /* Parse input names and dimensions */
  for (i = 0; i < num_inputs; ++i)
  {
    char *name = NULL;
    /* Get the names of the input tensor[i] */
    ort_status = onnx_state->g_ort->SessionGetInputName(onnx_state->session, i, onnx_state->allocator, &name);
    if (!CheckStatus(onnx_state->g_ort, ort_status, status))
    {
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    onnx_state->input_names[i] = name;

    OrtTypeInfo *type_info = NULL;  /* Released by ReleaseTypeInfo */
    ort_status = onnx_state->g_ort->SessionGetInputTypeInfo(onnx_state->session, i, &type_info);
    if (!CheckStatus(onnx_state->g_ort, ort_status, status))
    {
      CleanupOnSetupFailure(&onnx_state);
      return;
    }

    const OrtTensorTypeAndShapeInfo *tensor_info = NULL;  /* Do not free this value. It will be valid until type_info is freed */
    /* Check if the input data type is double */
    ONNXTensorElementDataType element_type; /* No need to free. enum */
    ort_status = onnx_state->g_ort->CastTypeInfoToTensorInfo(type_info, &tensor_info);
    if (!CheckStatus(onnx_state->g_ort, ort_status, status))
    {
      onnx_state->g_ort->ReleaseTypeInfo(type_info);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    /* Get the input data type */
    ort_status = onnx_state->g_ort->GetTensorElementType(
        tensor_info, &element_type);
    if (!CheckStatus(onnx_state->g_ort, ort_status, status))
    {
      onnx_state->g_ort->ReleaseTypeInfo(type_info);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    /* Check if the input data type is double (important for geoscience) */
    if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)
    {
      onnx_state->g_ort->ReleaseTypeInfo(type_info);
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "ONNX input tensor '%s' must have double elements.",
               onnx_state->input_names[i]);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    /* Record the dimension number */
    size_t dim_count = 0;
    /* Get the number of the dimensions */
    ort_status = onnx_state->g_ort->GetDimensionsCount(tensor_info, &dim_count);
    if (!CheckStatus(onnx_state->g_ort, ort_status, status))
    {
      onnx_state->g_ort->ReleaseTypeInfo(type_info);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }

    /* ONNX scalars report zero dimensions; normalize them to a one-element
    ** tensor so setup and shutdown can use the same tensor path. */
    if (dim_count == 0)
    {
      onnx_state->input_num_dim[i] = 1;
      /* Set up slot for the input data */
      onnx_state->input_dim_values[i] = (int64_t *)calloc(1, sizeof(int64_t));
      if (onnx_state->input_dim_values[i] == NULL)
      {
        onnx_state->g_ort->ReleaseTypeInfo(type_info);
        status->error = kAlquimiaErrorEngineIntegrity;
        snprintf(status->message, kAlquimiaMaxStringLength, "Memory allocation failed for scalar input_dim_values.");
        CleanupOnSetupFailure(&onnx_state);
        return;
      }
      onnx_state->input_dim_values[i][0] = 1;
    }
    /* Vector tensor */
    else
    {
      onnx_state->input_num_dim[i] = dim_count;
      onnx_state->input_dim_values[i] = (int64_t *)calloc(dim_count, sizeof(int64_t));
      if (onnx_state->input_dim_values[i] == NULL)
      {
        onnx_state->g_ort->ReleaseTypeInfo(type_info);
        status->error = kAlquimiaErrorEngineIntegrity;
        snprintf(status->message, kAlquimiaMaxStringLength, "Memory allocation failed for input_dim_values.");
        CleanupOnSetupFailure(&onnx_state);
        return;
      }
      /* Get the dimensions */
      ort_status = onnx_state->g_ort->GetDimensions(tensor_info, onnx_state->input_dim_values[i], dim_count);
      if (!CheckStatus(onnx_state->g_ort, ort_status, status))
      {
        onnx_state->g_ort->ReleaseTypeInfo(type_info);
        CleanupOnSetupFailure(&onnx_state);
        return;
      }
    }

    /* The supported trained models use an unknown leading batch extent. The
    ** interface operates on one state at a time, so that batch is fixed to
    ** one. Other dynamic extents cannot be sized safely at setup. */
    /* Calculate the total input numbers for each tensor */
    size_t total_size = 1; /* batch_size */
    /* For the input dimensions [] */
    for (j = 0; j < onnx_state->input_num_dim[i]; ++j)
    {
      /* [-1, feature numbers] */
      if (onnx_state->input_dim_values[i][j] <= 0)
      {
        /* Only when input_num_dim[i][0] == -1 */
        if (j != 0 || onnx_state->input_num_dim[i] == 1)
        {
          onnx_state->g_ort->ReleaseTypeInfo(type_info);
          status->error = kAlquimiaErrorEngineIntegrity;
          snprintf(status->message, kAlquimiaMaxStringLength,
                   "ONNX input tensor '%s' has an unsupported dynamic extent.",
                   onnx_state->input_names[i]);
          CleanupOnSetupFailure(&onnx_state);
          return;
        }
        onnx_state->input_dim_values[i][j] = 1;
      }
      /* Guard
      ** Check if feature numbers * total size > size_t
      ** In this case, feature numbers * batch size > size_t
      */
      if ((size_t)onnx_state->input_dim_values[i][j] > SIZE_MAX / total_size)
      {
        onnx_state->g_ort->ReleaseTypeInfo(type_info);
        status->error = kAlquimiaErrorEngineIntegrity;
        snprintf(status->message, kAlquimiaMaxStringLength,
                 "ONNX input tensor '%s' element count overflows size_t.",
                 onnx_state->input_names[i]);
        CleanupOnSetupFailure(&onnx_state);
        return;
      }
      total_size *= (size_t)onnx_state->input_dim_values[i][j];
    }
    onnx_state->input_total_size[i] = total_size;

    /* allocate memory for the input data */
    onnx_state->input_data[i] = (double *)calloc(total_size, sizeof(double));
    if (onnx_state->input_data[i] == NULL)
    {
      onnx_state->g_ort->ReleaseTypeInfo(type_info);
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Memory allocation failed for input_data.");
      CleanupOnSetupFailure(&onnx_state);
      return;
    }

    /* The OrtValue wraps input_data; ONNX Runtime does not own that buffer. */
    ort_status = onnx_state->g_ort->CreateTensorWithDataAsOrtValue(
        onnx_state->memory_info, onnx_state->input_data[i], total_size * sizeof(double),
        onnx_state->input_dim_values[i], onnx_state->input_num_dim[i],
        ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, &onnx_state->input_tensor[i]);
    if (!CheckStatus(onnx_state->g_ort, ort_status, status))
    {
      onnx_state->g_ort->ReleaseTypeInfo(type_info);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    /* Release the type info */
    onnx_state->g_ort->ReleaseTypeInfo(type_info);
  }

  onnx_state->output_names = (char **)calloc(num_outputs, sizeof(char *));
  onnx_state->output_num_dim = (size_t *)calloc(num_outputs, sizeof(size_t));
  onnx_state->output_dim_values = (int64_t **)calloc(num_outputs, sizeof(int64_t *));
  onnx_state->output_total_size = (size_t *)calloc(num_outputs, sizeof(size_t));
  onnx_state->output_data = (double **)calloc(num_outputs, sizeof(double *));
  onnx_state->output_tensor = (OrtValue **)calloc(num_outputs, sizeof(OrtValue *));

  if (onnx_state->output_names == NULL || onnx_state->output_num_dim == NULL ||
      onnx_state->output_dim_values == NULL || onnx_state->output_total_size == NULL ||
      onnx_state->output_data == NULL || onnx_state->output_tensor == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Memory allocation failed for output container arrays.");
    CleanupOnSetupFailure(&onnx_state);
    return;
  }

  /* Parse output names and dimensions */
  for (i = 0; i < num_outputs; ++i)
  {
    char *name = NULL;
    /* Get the names of the output tensor[i] */
    ort_status = onnx_state->g_ort->SessionGetOutputName(onnx_state->session, i, onnx_state->allocator, &name);
    if (!CheckStatus(onnx_state->g_ort, ort_status, status))
    {
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    onnx_state->output_names[i] = name;

    OrtTypeInfo *type_info = NULL;  /* Released by ReleaseTypeInfo */
    ort_status = onnx_state->g_ort->SessionGetOutputTypeInfo(onnx_state->session, i, &type_info);
    if (!CheckStatus(onnx_state->g_ort, ort_status, status))
    {
      CleanupOnSetupFailure(&onnx_state);
      return;
    }

    const OrtTensorTypeAndShapeInfo *tensor_info = NULL;  /* Do not free this value. It will be valid until type_info is freed */
    /* Check if the output data type is double */
    ONNXTensorElementDataType element_type; /* No need to free. enum */
    ort_status = onnx_state->g_ort->CastTypeInfoToTensorInfo(type_info, &tensor_info);
    if (!CheckStatus(onnx_state->g_ort, ort_status, status))
    {
      onnx_state->g_ort->ReleaseTypeInfo(type_info);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    /* Get the output data type */
    ort_status = onnx_state->g_ort->GetTensorElementType(
        tensor_info, &element_type);
    if (!CheckStatus(onnx_state->g_ort, ort_status, status))
    {
      onnx_state->g_ort->ReleaseTypeInfo(type_info);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    /* Check if the output data type is double (important for geoscience) */
    if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)
    {
      onnx_state->g_ort->ReleaseTypeInfo(type_info);
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "ONNX output tensor '%s' must have double elements.",
               onnx_state->output_names[i]);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    /* Record the dimension number */
    size_t dim_count = 0;
    /* Get the number of the dimensions */
    ort_status = onnx_state->g_ort->GetDimensionsCount(tensor_info, &dim_count);
    if (!CheckStatus(onnx_state->g_ort, ort_status, status))
    {
      onnx_state->g_ort->ReleaseTypeInfo(type_info);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }

    /* ONNX scalars report zero dimensions; normalize them to a one-element
    ** tensor so setup and shutdown can use the same tensor path. */
    if (dim_count == 0)
    {
      onnx_state->output_num_dim[i] = 1;
      /* Set up slot for the output data */
      onnx_state->output_dim_values[i] = (int64_t *)calloc(1, sizeof(int64_t));
      if (onnx_state->output_dim_values[i] == NULL)
      {
        onnx_state->g_ort->ReleaseTypeInfo(type_info);
        status->error = kAlquimiaErrorEngineIntegrity;
        snprintf(status->message, kAlquimiaMaxStringLength, "Memory allocation failed for scalar output_dim_values.");
        CleanupOnSetupFailure(&onnx_state);
        return;
      }
      onnx_state->output_dim_values[i][0] = 1;
    }
    /* Vector tensor */
    else
    {
      onnx_state->output_num_dim[i] = dim_count;
      onnx_state->output_dim_values[i] = (int64_t *)calloc(dim_count, sizeof(int64_t));
      if (onnx_state->output_dim_values[i] == NULL)
      {
        onnx_state->g_ort->ReleaseTypeInfo(type_info);
        status->error = kAlquimiaErrorEngineIntegrity;
        snprintf(status->message, kAlquimiaMaxStringLength, "Memory allocation failed for output_dim_values.");
        CleanupOnSetupFailure(&onnx_state);
        return;
      }
      /* Get the dimensions */
      ort_status = onnx_state->g_ort->GetDimensions(tensor_info, onnx_state->output_dim_values[i], dim_count);
      if (!CheckStatus(onnx_state->g_ort, ort_status, status))
      {
        onnx_state->g_ort->ReleaseTypeInfo(type_info);
        CleanupOnSetupFailure(&onnx_state);
        return;
      }
    }

    /* Match the single-state batch convention used for outputs. */
    /* Calculate the total output numbers for each tensor */
    size_t total_size = 1; /* batch_size */
    /* For the output dimensions [] */
    for (j = 0; j < onnx_state->output_num_dim[i]; ++j)
    {
      /* [-1, feature numbers] */
      if (onnx_state->output_dim_values[i][j] <= 0)
      {
        /* Only when output_num_dim[i][0] == -1 */
        if (j != 0 || onnx_state->output_num_dim[i] == 1)
        {
          onnx_state->g_ort->ReleaseTypeInfo(type_info);
          status->error = kAlquimiaErrorEngineIntegrity;
          snprintf(status->message, kAlquimiaMaxStringLength,
                   "ONNX output tensor '%s' has an unsupported dynamic extent.",
                   onnx_state->output_names[i]);
          CleanupOnSetupFailure(&onnx_state);
          return;
        }
        onnx_state->output_dim_values[i][j] = 1;
      }
      /* Guard
      ** Check if feature numbers * total size > size_t
      ** In this case, feature numbers * batch size > size_t
      */
      if ((size_t)onnx_state->output_dim_values[i][j] > SIZE_MAX / total_size)
      {
        onnx_state->g_ort->ReleaseTypeInfo(type_info);
        status->error = kAlquimiaErrorEngineIntegrity;
        snprintf(status->message, kAlquimiaMaxStringLength,
                 "ONNX output tensor '%s' element count overflows size_t.",
                 onnx_state->output_names[i]);
        CleanupOnSetupFailure(&onnx_state);
        return;
      }
      total_size *= (size_t)onnx_state->output_dim_values[i][j];
    }
    onnx_state->output_total_size[i] = total_size;

    /* allocate memory for the output data */
    onnx_state->output_data[i] = (double *)calloc(total_size, sizeof(double));
    if (onnx_state->output_data[i] == NULL)
    {
      onnx_state->g_ort->ReleaseTypeInfo(type_info);
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Memory allocation failed for output_data.");
      CleanupOnSetupFailure(&onnx_state);
      return;
    }

    /* The OrtValue wraps output_data; ONNX Runtime does not own that buffer. */
    ort_status = onnx_state->g_ort->CreateTensorWithDataAsOrtValue(
        onnx_state->memory_info, onnx_state->output_data[i], total_size * sizeof(double),
        onnx_state->output_dim_values[i], onnx_state->output_num_dim[i],
        ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, &onnx_state->output_tensor[i]);
    if (!CheckStatus(onnx_state->g_ort, ort_status, status))
    {
      onnx_state->g_ort->ReleaseTypeInfo(type_info);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    /* Release the type info */
    onnx_state->g_ort->ReleaseTypeInfo(type_info);
  }

  /* Each reaction step mutates the reusable tensors and backing buffers in
  ** OnnxEngineState, so shared-state calls are not thread-safe. MPI-only use
  ** remains safe because each process owns its engine state. */
  functionality->thread_safe = false;
  functionality->temperature_dependent = false;
  functionality->pressure_dependent = false;
  functionality->porosity_update = false;
  functionality->operator_splitting = true;
  functionality->global_implicit = false;
  functionality->index_base = 0;

  /* Every flattened tensor value must have a complete, valid mapping. */
  for (i = 0; i < onnx_state->num_inputs; ++i)
  {
    /* Guard
    ** avoid total feature inputs > size_t
    */
    if (onnx_state->input_total_size[i] >
        SIZE_MAX - onnx_state->total_flat_inputs)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "Flattened ONNX input element count overflows size_t.");
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    onnx_state->total_flat_inputs += onnx_state->input_total_size[i];
  }

  for (i = 0; i < onnx_state->num_outputs; ++i)
  {
    /* Guard
    ** avoid total feature outputs > size_t
    */
    if (onnx_state->output_total_size[i] >
        SIZE_MAX - onnx_state->total_flat_outputs)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "Flattened ONNX output element count overflows size_t.");
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    onnx_state->total_flat_outputs += onnx_state->output_total_size[i];
  }

  /* Set up the mapping rules inside the OnnxEngine->mapping struct 
  ** Initialize AlquimiaSize
  */
  if (!BuildManifestMappings(onnx_state, sizes, status))
  {
    CleanupOnSetupFailure(&onnx_state);
    return;
  }

  /* Freed the manifest form the onnx_interface_manifest */
  OnnxAlquimiaFreeManifest(&onnx_state->manifest);
  *(OnnxEngineState **)onnx_engine_state = onnx_state;
}

/**
 * @brief Releases all adapter and ONNX Runtime resources.
 * @param onnx_engine_state Address of the engine pointer created by setup.
 * @param status Receives an invalid-engine error for a NULL engine.
 *
 * Tensor names must be released with the ONNX allocator, while dimensions,
 * buffers, mappings, and copied feature names use the C allocator. On success,
 * the caller's engine pointer is set to NULL.
 */
void onnx_alquimia_shutdown(
    void *onnx_engine_state,
    AlquimiaEngineStatus *status)
{
  /* Input and output tensors are created via independent ORT API calls during setup.
  ** Since there is no strong coupling between them, they are released independently. */
  OnnxEngineState *onnx_state;
  size_t i;

  status->error = kAlquimiaNoError;
  status->message[0] = '\0';

  if (onnx_engine_state == NULL || *(OnnxEngineState **)onnx_engine_state == NULL)
  {
    status->error = kAlquimiaErrorInvalidEngine;
    snprintf(status->message, kAlquimiaMaxStringLength, "Invalid ONNX engine state pointer in shutdown.");
    return;
  }

  onnx_state = *(OnnxEngineState **)onnx_engine_state;
  if (onnx_state->g_ort != NULL)
  {
    /* Release explicit feature mappings if allocated */
    if (onnx_state->input_mappings != NULL)
    {
      for (i = 0; i < onnx_state->total_flat_inputs; ++i)
      {
        free(onnx_state->input_mappings[i].feature);
      }
      free(onnx_state->input_mappings);
      onnx_state->input_mappings = NULL;
    }
    if (onnx_state->output_mappings != NULL)
    {
      free(onnx_state->output_mappings);
      onnx_state->output_mappings = NULL;
    }

    /* Release input tensors and associated buffers */
    if (onnx_state->input_tensor != NULL)
    {
      for (i = 0; i < onnx_state->num_inputs; ++i)
      {
        if (onnx_state->input_tensor[i] != NULL)
        {
          onnx_state->g_ort->ReleaseValue(onnx_state->input_tensor[i]);
        }
      }
      free(onnx_state->input_tensor);
      onnx_state->input_tensor = NULL;
    }
    if (onnx_state->input_data != NULL)
    {
      for (i = 0; i < onnx_state->num_inputs; ++i)
      {
        if (onnx_state->input_data[i] != NULL)
        {
          free(onnx_state->input_data[i]);
        }
      }
      free(onnx_state->input_data);
      onnx_state->input_data = NULL;
    }
    if (onnx_state->input_dim_values != NULL)
    {
      for (i = 0; i < onnx_state->num_inputs; ++i)
      {
        if (onnx_state->input_dim_values[i] != NULL)
        {
          free(onnx_state->input_dim_values[i]);
        }
      }
      free(onnx_state->input_dim_values);
      onnx_state->input_dim_values = NULL;
    }
    if (onnx_state->input_num_dim != NULL)
    {
      free(onnx_state->input_num_dim);
      onnx_state->input_num_dim = NULL;
    }
    if (onnx_state->input_total_size != NULL)
    {
      free(onnx_state->input_total_size);
      onnx_state->input_total_size = NULL;
    }

    /* Release output tensors and associated buffers */
    if (onnx_state->output_tensor != NULL)
    {
      for (i = 0; i < onnx_state->num_outputs; ++i)
      {
        if (onnx_state->output_tensor[i] != NULL)
        {
          onnx_state->g_ort->ReleaseValue(onnx_state->output_tensor[i]);
        }
      }
      free(onnx_state->output_tensor);
      onnx_state->output_tensor = NULL;
    }
    if (onnx_state->output_data != NULL)
    {
      for (i = 0; i < onnx_state->num_outputs; ++i)
      {
        if (onnx_state->output_data[i] != NULL)
        {
          free(onnx_state->output_data[i]);
        }
      }
      free(onnx_state->output_data);
      onnx_state->output_data = NULL;
    }
    if (onnx_state->output_dim_values != NULL)
    {
      for (i = 0; i < onnx_state->num_outputs; ++i)
      {
        if (onnx_state->output_dim_values[i] != NULL)
        {
          free(onnx_state->output_dim_values[i]);
        }
      }
      free(onnx_state->output_dim_values);
      onnx_state->output_dim_values = NULL;
    }
    if (onnx_state->output_num_dim != NULL)
    {
      free(onnx_state->output_num_dim);
      onnx_state->output_num_dim = NULL;
    }
    if (onnx_state->output_total_size != NULL)
    {
      free(onnx_state->output_total_size);
      onnx_state->output_total_size = NULL;
    }

    /* ONNX Runtime allocates tensor names with this allocator. */
    if (onnx_state->allocator != NULL)
    {
      if (onnx_state->input_names != NULL)
      {
        for (i = 0; i < onnx_state->num_inputs; ++i)
        {
          if (onnx_state->input_names[i] != NULL)
          {
            onnx_state->allocator->Free(onnx_state->allocator, onnx_state->input_names[i]);
          }
        }
        free(onnx_state->input_names);
        onnx_state->input_names = NULL;
      }
      if (onnx_state->output_names != NULL)
      {
        for (i = 0; i < onnx_state->num_outputs; ++i)
        {
          if (onnx_state->output_names[i] != NULL)
          {
            onnx_state->allocator->Free(onnx_state->allocator, onnx_state->output_names[i]);
          }
        }
        free(onnx_state->output_names);
        onnx_state->output_names = NULL;
      }
      onnx_state->g_ort->ReleaseAllocator(onnx_state->allocator);
      onnx_state->allocator = NULL;
    }
    if (onnx_state->memory_info != NULL)
    {
      onnx_state->g_ort->ReleaseMemoryInfo(onnx_state->memory_info);
      onnx_state->memory_info = NULL;
    }
    if (onnx_state->session != NULL)
    {
      onnx_state->g_ort->ReleaseSession(onnx_state->session);
      onnx_state->session = NULL;
    }
    if (onnx_state->session_options != NULL)
    {
      onnx_state->g_ort->ReleaseSessionOptions(onnx_state->session_options);
      onnx_state->session_options = NULL;
    }
    if (onnx_state->env != NULL)
    {
      onnx_state->g_ort->ReleaseEnv(onnx_state->env);
      onnx_state->env = NULL;
    }
  }
  OnnxAlquimiaFreeManifest(&onnx_state->manifest);
  free(onnx_state);
  *(OnnxEngineState **)onnx_engine_state = NULL;
}

/**
 * @brief Applies named aqueous constraints to manifest-mapped model inputs.
 * @param onnx_engine_state Address of an initialized engine pointer.
 * @param condition Named constraints to apply; NULL or empty is a no-op.
 * @param props Unused by the ONNX adapter.
 * @param state State receiving values for matching input feature names.
 * @param aux_data Unused by the ONNX adapter.
 * @param status Receives invalid-engine or mapped-state access errors.
 *
 * Constraints absent from @p condition leave their mapped state values
 * unchanged. This preserves the generic condition-processing behavior while
 * allowing callers to initialize only the features they provide.
 */
void onnx_alquimia_processcondition(
    void *onnx_engine_state,
    AlquimiaGeochemicalCondition *condition,
    AlquimiaProperties *props,
    AlquimiaState *state,
    AlquimiaAuxiliaryData *aux_data,
    AlquimiaEngineStatus *status)
{
  OnnxEngineState *onnx_state;
  size_t k;

  status->error = kAlquimiaNoError;
  status->message[0] = '\0';

  (void)props;
  (void)aux_data;

  if (onnx_engine_state == NULL || *(OnnxEngineState **)onnx_engine_state == NULL)
  {
    status->error = kAlquimiaErrorInvalidEngine;
    snprintf(status->message, kAlquimiaMaxStringLength, "Invalid ONNX engine state pointer in ProcessCondition.");
    return;
  }

  onnx_state = *(OnnxEngineState **)onnx_engine_state;

  if (state == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Invalid state pointer in ProcessCondition.");
    return;
  }

  /* An absent condition intentionally preserves all existing state values. */
  if (condition == NULL || condition->aqueous_constraints.data == NULL || condition->aqueous_constraints.size <= 0)
  {
    return;
  }

  /* Input feature names are validated and copied during setup, so this path
  ** performs no manifest parsing or ONNX metadata lookup. */
  for (k = 0; k < onnx_state->total_flat_inputs; ++k)
  {
    const char *feature = onnx_state->input_mappings[k].feature;
    int c_idx;
    AlquimiaAqueousConstraint *matching_constraint = NULL;

    /* Search for aqueous constraint with matching name */
    for (c_idx = 0; c_idx < condition->aqueous_constraints.size; ++c_idx)
    {
      if (strcmp(condition->aqueous_constraints.data[c_idx].primary_species_name, feature) == 0)
      {
        matching_constraint = &condition->aqueous_constraints.data[c_idx];
        break;
      }
    }

    if (matching_constraint != NULL)
    {
      SetAlquimiaValue(state, onnx_state->input_mappings[k],
                       matching_constraint->value, status);
      if (status->error != kAlquimiaNoError)
      {
        return;
      }
    }
  }
}

/**
 * @brief Runs one operator-split ONNX inference and routes its outputs.
 * @param onnx_engine_state Address of an initialized engine pointer.
 * @param delta_t Unused; the model receives only explicitly mapped state data.
 * @param props Unused by the ONNX adapter.
 * @param state Supplies mapped inputs and receives mapped outputs.
 * @param aux_data Unused by the ONNX adapter.
 * @param natural_id Unused by the ONNX adapter.
 * @param status Receives state-access or ONNX Runtime errors.
 *
 * The engine reuses mutable tensor buffers allocated during setup. Calls that
 * share one engine instance are therefore not thread-safe.
 */
void onnx_alquimia_reactionstepoperatorsplit(
    void *onnx_engine_state,
    double delta_t,
    AlquimiaProperties *props,
    AlquimiaState *state,
    AlquimiaAuxiliaryData *aux_data,
    int natural_id,
    AlquimiaEngineStatus *status)
{
  OnnxEngineState *onnx_state;
  OrtStatus *ort_status;
  int i;

  status->error = kAlquimiaNoError;
  status->message[0] = '\0';

  // Unused
  (void)delta_t;
  (void)props;
  (void)aux_data;
  (void)natural_id;

  if (onnx_engine_state == NULL || *(OnnxEngineState **)onnx_engine_state == NULL)
  {
    status->error = kAlquimiaErrorInvalidEngine;
    snprintf(status->message, kAlquimiaMaxStringLength, "Invalid ONNX engine state pointer in reactionstepoperatorsplit.");
    return;
  }

  onnx_state = *(OnnxEngineState **)onnx_engine_state;

  if (state == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Invalid state in reactionstepoperatorsplit.");
    return;
  }

  {
    size_t flat_idx = 0;
    for (i = 0; i < (int)onnx_state->num_inputs; ++i)
    {
      size_t k;
      for (k = 0; k < onnx_state->input_total_size[i]; ++k)
      {
        onnx_state->input_data[i][k] = GetAlquimiaValue(state, onnx_state->input_mappings[flat_idx], status);
        if (status->error != kAlquimiaNoError)
        {
          return;
        }
        flat_idx++;
      }
    }
  }

  /* Run inference using pre-allocated input and output tensors and dynamic names */
  ort_status = onnx_state->g_ort->Run(
      onnx_state->session,
      NULL, /* RunOptions */
      (const char *const *)onnx_state->input_names,
      (const OrtValue *const *)onnx_state->input_tensor,
      onnx_state->num_inputs,
      (const char *const *)onnx_state->output_names,
      onnx_state->num_outputs,
      onnx_state->output_tensor);

  if (!CheckStatus(onnx_state->g_ort, ort_status, status))
  {
    return;
  }

  /* Copy output data back through the required explicit mappings. */
  {
    size_t flat_idx = 0;
    for (i = 0; i < (int)onnx_state->num_outputs; ++i)
    {
      double *out_arr = NULL;
      ort_status = onnx_state->g_ort->GetTensorMutableData(onnx_state->output_tensor[i], (void **)&out_arr);
      if (!CheckStatus(onnx_state->g_ort, ort_status, status))
      {
        return;
      }

      if (out_arr == NULL)
      {
        status->error = kAlquimiaErrorEngineIntegrity;
        snprintf(status->message, kAlquimiaMaxStringLength, "GetTensorMutableData returned NULL output buffer for output tensor %d.", i);
        return;
      }

      size_t k;
      for (k = 0; k < onnx_state->output_total_size[i]; ++k)
      {
        SetAlquimiaValue(state, onnx_state->output_mappings[flat_idx], out_arr[k], status);
        if (status->error != kAlquimiaNoError)
        {
          return;
        }
        flat_idx++;
      }
    }
  }
}

/**
 * @brief Implements the generic auxiliary-output hook as a successful no-op.
 *
 * Version 1 ONNX manifests map only AlquimiaState fields and define no
 * auxiliary outputs.
 */
void onnx_alquimia_getauxiliaryoutput(
    void *onnx_engine_state,
    AlquimiaProperties *props,
    AlquimiaState *state,
    AlquimiaAuxiliaryData *aux_data,
    AlquimiaAuxiliaryOutputData *aux_out,
    AlquimiaEngineStatus *status)
{
  status->error = kAlquimiaNoError;
  status->message[0] = '\0';

  // Unused data
  (void)onnx_engine_state;
  (void)props;
  (void)state;
  (void)aux_data;
  (void)aux_out;
}

/**
 * @brief Copies manifest input feature names into problem metadata vectors.
 * @param onnx_engine_state Address of an initialized engine pointer.
 * @param meta_data Metadata storage allocated from the setup-derived sizes.
 * @param status Receives invalid-engine or invalid-destination errors.
 *
 * Fields that share an Alquimia metadata vector were checked for conflicting
 * names during setup, so each destination index has one representable name.
 */
void onnx_alquimia_getproblemmetadata(
    void *onnx_engine_state,
    AlquimiaProblemMetaData *meta_data,
    AlquimiaEngineStatus *status)
{
  OnnxEngineState *onnx_state;
  size_t i;

  status->error = kAlquimiaNoError;
  status->message[0] = '\0';

  if (onnx_engine_state == NULL || *(OnnxEngineState **)onnx_engine_state == NULL)
  {
    status->error = kAlquimiaErrorInvalidEngine;
    snprintf(status->message, kAlquimiaMaxStringLength, "Invalid ONNX engine state pointer.");
    return;
  }

  onnx_state = *(OnnxEngineState **)onnx_engine_state;
  if (meta_data == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Invalid problem metadata destination for ONNX engine.");
    return;
  }

  for (i = 0; i < onnx_state->total_flat_inputs; ++i)
  {
    FeatureMapping mapping = onnx_state->input_mappings[i];
    AlquimiaVectorString *names = MetadataNamesForMapping(
        meta_data, mapping.alquimia_state);
    StoreMetadataName(names, mapping.alquimia_state_index, mapping.feature);
  }
}

#endif /* ALQUIMIA_HAVE_ONNX */
