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

/* **************************************************************************** 
**
** ONNX Alquimia Interface module
**
** Authors: 
**        Zhuolei Feng, Sergi Molins
**
** Notes:
**
**  * Public function call signatures, including intent, are dictated
**    by the alquimia API.
**
**  * alquimia data structures defined in the AlquimiaContainers_module
**    (alquimia_containers.h) are dictated by the alquimia API.
**
**  * All other function calls (e.g. involving ORT structures) are only
**    used here and not available in the interface
**
**  * It makes use of onnx_alquimia_config.h for reading the input file 
**    with information about the .onnx model file and metadata
**
** **************************************************************************** 
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
#include "alquimia/onnx_alquimia_config.h"

#if ALQUIMIA_HAVE_ONNX

/* --------------------DATA STRUCTURES USED HERE ONLY ----------------------------*/
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
  /* Adapter-owned feature name retained after the onnx_config is released. */
  char *feature;
} FeatureMapping;

typedef struct {
  /* Dynamic input/output info */
  /* Align with the num_inputs/outputs in onnx_config */
  size_t num_tensors;
  /* Align with the tensor in onnx_config */
  char **names;
  /* Number of dimensions for each input/output tensor
  ** 0 for a scalar tensor
  */
  size_t *num_dim;
  /* [batch size, input/output features] / [input/output features]
  ** batch size == -1 means it's dynamic
  ** Considering the architecture of Alquimia, batch size = 1
  */
  int64_t **dim_values; 
  /* Each input/output tensor has their own size */
  size_t *total_size; 
  /* The real numbers for each input/output tensor. */
  double **data;
  OrtValue **tensor;
} OrtTensor;

typedef struct {
  /* OnnxRuntime API */
  /* Const pointer to the global OrtApi function table. Do NOT release. */
  const OrtApi *g_ort;
  OrtEnv *env;
  OrtSessionOptions *session_options;
  OrtSession *session;
  OrtMemoryInfo *memory_info;
  OrtAllocator *allocator;
} Ort;

typedef struct {
  Ort ort;
  /* Retained for named JSON conditions used by ProcessCondition. */
  OnnxAlquimiaConfig onnx_config;
  /* Selects JSON-backed initialization instead of driver constraints. */
  bool hands_off;

  /* Flatten non-sequential input/output tensors into 1-dimensional arrays. */
  size_t total_flat_inputs;
  size_t total_flat_outputs;

  OrtTensor input_tensors;
  OrtTensor output_tensors;

  FeatureMapping *input_mappings;  /* Array of size total_flat_inputs */
  FeatureMapping *output_mappings; /* Array of size total_flat_outputs */
  /* Parallel lookup indicating that the paired phase is also an output. */
  bool *output_has_paired_mapping;
} OnnxEngineState;

/* ------------- HELPER and ONNX-RELATED FUNCTIONS USED HERE ONLY -----------------*/

/**
 * @brief Converts an ONNX Runtime status into an Alquimia engine status.
 * @param[in] g_ort ONNX Runtime API used to inspect and release @p status.
 * @param[in] status ONNX status returned by an API call; NULL indicates success.
 * @param[in,out] alquimia_status Destination for the translated error and message.
 * @return True on success. On failure, releases @p status, records an engine
 *         integrity error, and returns false.
 */
static bool CheckStatus(const OrtApi *g_ort, OrtStatus *status, AlquimiaEngineStatus *alquimia_status)
{
  /* A non-NULL OrtStatus indicates an API failure; extract the error, 
  ** populate our status, and explicitly release the status object. 
  */
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
 * @brief Maps an exact onnx_config state-variable name to an AlquimiaState field.
 * @param[in] name Case-sensitive alquimia_state value from the onnx_config.
 * @param[out] alquimia_state Returns the corresponding mapping enum on success and remains
 *        unchanged when @p name is unsupported.
 * @return True when @p name identifies a supported scalar or vector field.
 */
static bool ParseStructName(const char *name, AlquimiaMappedStruct *alquimia_state) {
  /* For the vector in the AlquimiaState */
  if (strcmp(name, "total_mobile") == 0) {
    *alquimia_state = ALQUIMIA_STRUCT_TOTAL_MOBILE;
  } else if (strcmp(name, "total_immobile") == 0) {
    *alquimia_state = ALQUIMIA_STRUCT_TOTAL_IMMOBILE;
  } else if (strcmp(name, "mineral_volume_fraction") == 0) {
    *alquimia_state = ALQUIMIA_STRUCT_MINERAL_VOLUME_FRACTION;
  } else if (strcmp(name, "mineral_specific_surface_area") == 0) {
    *alquimia_state = ALQUIMIA_STRUCT_MINERAL_SPECIFIC_SURFACE_AREA;
  } else if (strcmp(name, "surface_site_density") == 0) {
    *alquimia_state = ALQUIMIA_STRUCT_SURFACE_SITE_DENSITY;
  } else if (strcmp(name, "cation_exchange_capacity") == 0) {
    *alquimia_state = ALQUIMIA_STRUCT_CATION_EXCHANGE_CAPACITY;
  } else if (strcmp(name, "porosity") == 0) {
    *alquimia_state = ALQUIMIA_STRUCT_POROSITY;
  } else if (strcmp(name, "temperature") == 0) {
    *alquimia_state = ALQUIMIA_STRUCT_TEMPERATURE;
  } else if (strcmp(name, "aqueous_pressure") == 0) {
    *alquimia_state = ALQUIMIA_STRUCT_AQUEOUS_PRESSURE;
  } else if (strcmp(name, "water_density") == 0) {
    *alquimia_state = ALQUIMIA_STRUCT_WATER_DENSITY;
  } else if (strcmp(name, "gas_concentration") == 0) {
    *alquimia_state = ALQUIMIA_STRUCT_GAS_CONCENTRATION;
  } else {
    return false;
  }
  return true;
}

/**
 * @brief Identifies mappings whose destination is an AlquimiaState scalar.
 * @param[in] alquimia_state Mapping destination to classify.
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
 * @param[in] meta_data Problem metadata whose name vectors were allocated from the
 *        model sizes.
 * @param[in] alquimia_state State field associated with a model feature.
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
 * @brief Categorizes state fields that share one problem-metadata name vector.
 * @param[in] alquimia_state State destination to classify.
 * @return A stable category identifier, or -1 for scalar fields without names.
 *
 * Mobile and immobile totals share primary_names, and both mineral vectors
 * share mineral_names. Setup rejects different feature names that metadata
 * could not represent independently.
 */
static int StateNameCategory(AlquimiaMappedStruct alquimia_state)
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
 * @brief Copies a config feature name into an allocated Alquimia name vector.
 * @param[in,out] names Destination name vector.
 * @param[in] index Zero-based destination index.
 * @param[in] value Null-terminated feature name to copy.
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
 * @brief Copies a onnx_config feature name into adapter-owned storage.
 * @param[in] name Null-terminated feature name to copy.
 * @return A newly allocated copy, or NULL on allocation failure.
 *
 * The returned string outlives the parsed onnx_config and is released with its
 * runtime mapping array during engine shutdown.
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
 * @brief Converts one validated onnx_config destination into a runtime mapping.
 * @param[in] alquimia_state Case-sensitive AlquimiaState variable name.
 * @param[in] alquimia_state_index Nonnegative state-vector index from the onnx_config.
 * @param[out] mapping Returns the runtime destination on success.
 * @param[out] status Returns an engine-integrity error on incompatibility.
 * @return True when the state variable and index can be represented safely.
 *
 * Scalar state variables require index zero. INT_MAX is rejected for vectors
 * so the later size calculation cannot overflow an Alquimia int.
 */
static bool ParseConfigMapping(
    const char *alquimia_state,
    int alquimia_state_index,
    FeatureMapping *mapping,
    AlquimiaEngineStatus *status)
{
  /* Parse the AlquimiaState */
  if (!ParseStructName(alquimia_state, &mapping->alquimia_state))
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Unsupported AlquimiaState variable '%s' in ONNX config.",
             alquimia_state);
    return false;
  }
  /* Check if it's a valid scalar/vector in the AlquimiaState
  ** Index = 0 for valid scalar
  */
  if ((IsScalarMapping(mapping->alquimia_state) &&
       alquimia_state_index != 0) ||
      (!IsScalarMapping(mapping->alquimia_state) &&
      // Alquimia size = index + 1
       alquimia_state_index == INT_MAX))
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "ONNX config AlquimiaState index %d is incompatible with "
             "variable '%s'.", alquimia_state_index, alquimia_state);
    return false;
  }
  mapping->alquimia_state_index = alquimia_state_index;
  return true;
}

/**
 * @brief Expands the AlquimiaSizes category required by one mapping.
 * @param[in] mapping Validated state destination.
 * @param[in,out] sizes Accumulator initialized to zero before the first mapping.
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
 * @param[in] tensor Case-sensitive tensor name from the onnx_config.
 * @param[in] tensor_element_index Zero-based flattened index within that tensor.
 * @param[in] num_tensors Number of model tensors in @p names and @p tensor_sizes.
 * @param[in] names ONNX Runtime tensor names in session order.
 * @param[in] tensor_sizes Flattened element counts in session order.
 * @param[out] flat_index Returns the offset into the combined mapping array.
 * @param[out] status Returns an error for unknown, duplicate, or undersized tensors.
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
  /* Initialize sentinel value to track if the target tensor is found. */
  size_t matching_index = num_tensors;

  /* Check if there duplicate values in JSON */
  for (i = 0; i < num_tensors; ++i)
  {
    if (strcmp(tensor, names[i]) == 0)
    {
      /* If matching_index was already set, a duplicate name exists. */
      if (matching_index != num_tensors)
      {
        status->error = kAlquimiaErrorEngineIntegrity;
        snprintf(status->message, kAlquimiaMaxStringLength,
                 "ONNX model contains duplicate tensor name '%s'.", tensor);
        return false;
      }
      /* Find the tensor customized_tensor_element[0] for the first time */
      matching_index = i;
      /* Calculate the absolute flattened index for this specific element. */
      *flat_index = offset + tensor_element_index;
    }
    /* Accumulate tensor offset to track the starting bound of subsequent tensors. */
    offset += tensor_sizes[i];
  }
  /* Reject unknown tensors not present in the model metadata. */
  if (matching_index == num_tensors)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "ONNX config references unknown tensor '%s'.", tensor);
    return false;
  }
  /* Guard against out-of-bounds element indexing within the matched tensor. */
  if (tensor_element_index >= tensor_sizes[matching_index])
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "ONNX config tensor element index %zu is out of range for "
             "tensor '%s' of size %zu.", tensor_element_index, tensor,
             tensor_sizes[matching_index]);
    return false;
  }
  return true;
}

/**
 * @brief Rejects an input feature name already assigned to a tensor element.
 * @param[in] input_mappings Existing flattened input mappings.
 * @param[in] input_seen Marks the mappings that have already been populated.
 * @param[in] total_flat_inputs Number of entries in the mapping and seen arrays.
 * @param[in] feature Case-sensitive feature name to validate.
 * @param[out] status Returns an engine-integrity error for a duplicate name.
 * @return True when @p feature has not been assigned previously.
 *
 * ProcessCondition uses feature names as lookup keys, so exact duplicates
 * would let one condition ambiguously address multiple tensor elements.
 */
static bool ValidateUniqueInputFeature(
    const FeatureMapping *input_mappings,
    const bool *input_seen,
    size_t total_flat_inputs,
    const char *feature,
    AlquimiaEngineStatus *status)
{
  size_t i;

  for (i = 0; i < total_flat_inputs; ++i)
  {
    if (input_seen[i] && strcmp(input_mappings[i].feature, feature) == 0)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "Duplicate ONNX input feature name '%s'.", feature);
      return false;
    }
  }
  return true;
}

/**
 * @brief Rejects a feature name that conflicts at one metadata destination.
 * @param[in] mappings Existing flattened input or output mappings.
 * @param[in] seen Marks mappings that have already been populated.
 * @param[in] num_mappings Number of entries in the mapping and seen arrays.
 * @param[in] mapping Parsed runtime destination for the candidate mapping.
 * @param[in] config_mapping Candidate config mapping and feature name.
 * @param[out] status Returns an engine-integrity error for a conflict.
 * @return True when the candidate is compatible with all populated mappings.
 */
static bool ValidateConsistentFeatureName(
    const FeatureMapping *mappings,
    const bool *seen,
    size_t num_mappings,
    const FeatureMapping *mapping,
    const OnnxAlquimiaMapping *config_mapping,
    AlquimiaEngineStatus *status)
{
  size_t i;

  for (i = 0; i < num_mappings; ++i)
  {
    const FeatureMapping *other = &mappings[i];
    if (seen[i] &&
        StateNameCategory(other->alquimia_state) >= 0 &&
        StateNameCategory(other->alquimia_state) ==
            StateNameCategory(mapping->alquimia_state) &&
        other->alquimia_state_index == mapping->alquimia_state_index &&
        strcmp(other->feature, config_mapping->feature) != 0)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "Conflicting ONNX feature names '%s' and '%s' for "
               "AlquimiaState variable '%s' index %d.",
               other->feature, config_mapping->feature,
               config_mapping->alquimia_state,
               config_mapping->alquimia_state_index);
      return false;
    }
  }
  return true;
}

/**
 * @brief Builds complete flat input/output mappings from the parsed onnx_config.
 * @param[in,out] onnx_state Inspected model state and destination for runtime mappings.
 * @param[in,out] sizes Returns dimensions derived from the highest mapped indices.
 * @param[out] status Returns allocation or semantic-validation errors.
 * @return True only when every tensor element index has exactly one mapping.
 *
 * Input feature names are copied before the onnx_config is released. The seen
 * arrays enforce complete coverage and prevent duplicate tensor indices;
 * name-category checks preserve the metadata representation invariant.
 */
static bool BuildConfigMappings(
    OnnxEngineState *onnx_state,
    AlquimiaSizes *sizes,
    AlquimiaEngineStatus *status)
{
  /* Check if there is duplicate value in the JSON 
  ** They are for the 1-dimension array
  ** The input/output mapping are all 1-dimension arrays
  */
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
             "Memory allocation failed for ONNX config mappings.");
    free(input_seen);
    free(output_seen);
    return false;
  }
  /* Initialize AlquimiaSize */
  memset(sizes, 0, sizeof(*sizes));

  /* For every input tensor */
  for (i = 0; i < onnx_state->onnx_config.num_inputs; ++i)
  {
    /* This points to the real OnnxEngine->onnx_config.inputMapping */
    const OnnxAlquimiaMapping *config_input =
        &onnx_state->onnx_config.inputs[i];
    FeatureMapping *mapping;
    size_t flat_index;

    /* Map the tensor element index into the 1-dimensional flattened array. */
    if (!FindFlatTensorElement(config_input->tensor,
                               config_input->tensor_element_index,
                               onnx_state->input_tensors.num_tensors,
                               onnx_state->input_tensors.names,
                               onnx_state->input_tensors.total_size,
                               &flat_index, status))
    {
      free(input_seen);
      free(output_seen);
      return false;
    }
    /* Find the duplicate value */
    if (input_seen[flat_index])
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "Duplicate ONNX input mapping for tensor '%s' element index %zu.",
               config_input->tensor, config_input->tensor_element_index);
      free(input_seen);
      free(output_seen);
      return false;
    }
    mapping = &onnx_state->input_mappings[flat_index];
    /* Parse the AlquimiaState value to the input mapping */
    if (!ParseConfigMapping(config_input->alquimia_state,
                            config_input->alquimia_state_index, mapping,
                            status))
    {
      free(input_seen);
      free(output_seen);
      return false;
    }
    /* Check the duplicate input feature */
    if (!ValidateUniqueInputFeature(
            onnx_state->input_mappings, input_seen,
            onnx_state->total_flat_inputs, config_input->feature, status))
    {
      free(input_seen);
      free(output_seen);
      return false;
    }
    /* Check if there are conflicted input features that target the same
    ** AlquimiaState metadata name.
    */
    if (!ValidateConsistentFeatureName(
            onnx_state->input_mappings, input_seen,
            onnx_state->total_flat_inputs, mapping, config_input, status))
    {
      free(input_seen);
      free(output_seen);
      return false;
    }
    /* Copy the feature name */
    mapping->feature = CopyFeatureName(config_input->feature);
    if (mapping->feature == NULL)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "Memory allocation failed for ONNX input feature name.");
      free(input_seen);
      free(output_seen);
      return false;
    }
    /* Find the input tensor element for the first time */
    input_seen[flat_index] = true;
    /* Update the respective AlquimiaSize when finding new input tensor element */
    UpdateSizesForMapping(mapping, sizes);
  }

  /* For every output tensor */
  for (i = 0; i < onnx_state->onnx_config.num_outputs; ++i)
  {
    /* This points to the real OnnxEngine->onnx_config.outputMapping */
    const OnnxAlquimiaMapping *config_output =
        &onnx_state->onnx_config.outputs[i];
    FeatureMapping *mapping;
    size_t flat_index;

    /* Map the tensor element index into the 1-dimensional flattened array. */
    if (!FindFlatTensorElement(config_output->tensor,
                               config_output->tensor_element_index,
                               onnx_state->output_tensors.num_tensors,
                               onnx_state->output_tensors.names,
                               onnx_state->output_tensors.total_size,
                               &flat_index, status))
    {
      free(input_seen);
      free(output_seen);
      return false;
    }
    /* Find the duplicate value */
    if (output_seen[flat_index])
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "Duplicate ONNX output mapping for tensor '%s' element index %zu.",
               config_output->tensor, config_output->tensor_element_index);
      free(input_seen);
      free(output_seen);
      return false;
    }
    mapping = &onnx_state->output_mappings[flat_index];
    /* Parse the AlquimiaState value to the output mapping */
    if (!ParseConfigMapping(config_output->alquimia_state,
                            config_output->alquimia_state_index, mapping,
                            status))
    {
      free(input_seen);
      free(output_seen);
      return false;
    }
    // Compare between output mapping and input mappings
    if (!ValidateConsistentFeatureName(
            onnx_state->input_mappings, input_seen,
            onnx_state->total_flat_inputs, mapping, config_output, status) ||
        // Compare between output mapping and output mappings
        !ValidateConsistentFeatureName(
            onnx_state->output_mappings, output_seen,
            onnx_state->total_flat_outputs, mapping, config_output, status))
    {
      free(input_seen);
      free(output_seen);
      return false;
    }
    mapping->feature = CopyFeatureName(config_output->feature);
    if (mapping->feature == NULL)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "Memory allocation failed for ONNX output feature name.");
      free(input_seen);
      free(output_seen);
      return false;
    }
    /* Find the output tensor element for the first time */
    output_seen[flat_index] = true;
    /* Update the respective AlquimiaSize when finding new output tensor element */
    UpdateSizesForMapping(mapping, sizes);
  }

  for (i = 0; i < onnx_state->total_flat_inputs; ++i)
  {
    if (!input_seen[i])
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "ONNX config does not map every input tensor element index.");
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
               "ONNX config does not map every output tensor element index.");
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
 * @param[in] state State containing the model input value.
 * @param[in] mapping Validated destination field and zero-based vector index.
 * @param[out] status Returns an engine integrity error for an unknown field, NULL
 *        vector storage, or an out-of-bounds vector index.
 * @return The mapped value on success, or 0.0 after recording an error.
 */
static double GetAlquimiaValue(
    const AlquimiaState *state,
    FeatureMapping mapping,
    AlquimiaEngineStatus *status)
{
  switch (mapping.alquimia_state)
  {
  case ALQUIMIA_STRUCT_TOTAL_MOBILE:
    if (state->total_mobile.data == NULL || mapping.alquimia_state_index >= state->total_mobile.size)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds total_mobile access: index %d, size %d.", mapping.alquimia_state_index, state->total_mobile.size);
      return 0.0;
    }
    return state->total_mobile.data[mapping.alquimia_state_index];

  case ALQUIMIA_STRUCT_TOTAL_IMMOBILE:
    if (state->total_immobile.data == NULL || mapping.alquimia_state_index >= state->total_immobile.size)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds total_immobile access: index %d, size %d.", mapping.alquimia_state_index, state->total_immobile.size);
      return 0.0;
    }
    return state->total_immobile.data[mapping.alquimia_state_index];

  case ALQUIMIA_STRUCT_MINERAL_VOLUME_FRACTION:
      if (state->mineral_volume_fraction.data == NULL || mapping.alquimia_state_index >= state->mineral_volume_fraction.size) {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds mineral_volume_fraction access: index %d, size %d.", mapping.alquimia_state_index, state->mineral_volume_fraction.size);
      return 0.0;
    }
    return state->mineral_volume_fraction.data[mapping.alquimia_state_index];

  case ALQUIMIA_STRUCT_MINERAL_SPECIFIC_SURFACE_AREA:
      if (state->mineral_specific_surface_area.data == NULL || mapping.alquimia_state_index >= state->mineral_specific_surface_area.size) {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds mineral_specific_surface_area access: index %d, size %d.", mapping.alquimia_state_index, state->mineral_specific_surface_area.size);
      return 0.0;
    }
    return state->mineral_specific_surface_area.data[mapping.alquimia_state_index];

  case ALQUIMIA_STRUCT_SURFACE_SITE_DENSITY:
      if (state->surface_site_density.data == NULL || mapping.alquimia_state_index >= state->surface_site_density.size) {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds surface_site_density access: index %d, size %d.", mapping.alquimia_state_index, state->surface_site_density.size);
      return 0.0;
    }
    return state->surface_site_density.data[mapping.alquimia_state_index];

  case ALQUIMIA_STRUCT_CATION_EXCHANGE_CAPACITY:
      if (state->cation_exchange_capacity.data == NULL || mapping.alquimia_state_index >= state->cation_exchange_capacity.size) {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds cation_exchange_capacity access: index %d, size %d.", mapping.alquimia_state_index, state->cation_exchange_capacity.size);
      return 0.0;
    }
    return state->cation_exchange_capacity.data[mapping.alquimia_state_index];

  case ALQUIMIA_STRUCT_POROSITY:
    return state->porosity;

  case ALQUIMIA_STRUCT_TEMPERATURE:
    return state->temperature;

  case ALQUIMIA_STRUCT_AQUEOUS_PRESSURE:
    return state->aqueous_pressure;

  case ALQUIMIA_STRUCT_WATER_DENSITY:
    return state->water_density;

  case ALQUIMIA_STRUCT_GAS_CONCENTRATION:
      if (state->gas_concentration.data == NULL || mapping.alquimia_state_index >= state->gas_concentration.size) {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds gas_concentration access: index %d, size %d.", mapping.alquimia_state_index, state->gas_concentration.size);
      return 0.0;
    }
    return state->gas_concentration.data[mapping.alquimia_state_index];

  default:
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Unknown mapped struct type: %d.", mapping.alquimia_state);
    return 0.0;
  }
}

/**
 * @brief Writes one model output to a mapped AlquimiaState destination.
 * @param[in,out] state State that receives the model output value.
 * @param[in] mapping Validated destination field and zero-based vector index.
 * @param[in] value Model output to assign.
 * @param[out] status Returns an engine integrity error for an unknown field, NULL
 *        vector storage, or an out-of-bounds vector index.
 */
static void SetAlquimiaValue(
    AlquimiaState *state,
    FeatureMapping mapping,
    double value,
    AlquimiaEngineStatus *status)
{
  switch (mapping.alquimia_state)
  {
  case ALQUIMIA_STRUCT_TOTAL_MOBILE:
      if (state->total_mobile.data == NULL || mapping.alquimia_state_index >= state->total_mobile.size) {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds total_mobile write: index %d, size %d.", mapping.alquimia_state_index, state->total_mobile.size);
      return;
    }
    state->total_mobile.data[mapping.alquimia_state_index] = value;
    break;

  case ALQUIMIA_STRUCT_TOTAL_IMMOBILE:
      if (state->total_immobile.data == NULL || mapping.alquimia_state_index >= state->total_immobile.size) {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds total_immobile write: index %d, size %d.", mapping.alquimia_state_index, state->total_immobile.size);
      return;
    }
    state->total_immobile.data[mapping.alquimia_state_index] = value;
    break;

  case ALQUIMIA_STRUCT_MINERAL_VOLUME_FRACTION:
    if (state->mineral_volume_fraction.data == NULL || mapping.alquimia_state_index >= state->mineral_volume_fraction.size)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds mineral_volume_fraction write: index %d, size %d.", mapping.alquimia_state_index, state->mineral_volume_fraction.size);
      return;
    }
    state->mineral_volume_fraction.data[mapping.alquimia_state_index] = value;
    break;

  case ALQUIMIA_STRUCT_MINERAL_SPECIFIC_SURFACE_AREA:
    if (state->mineral_specific_surface_area.data == NULL || mapping.alquimia_state_index >= state->mineral_specific_surface_area.size)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds mineral_specific_surface_area write: index %d, size %d.", mapping.alquimia_state_index, state->mineral_specific_surface_area.size);
      return;
    }
    state->mineral_specific_surface_area.data[mapping.alquimia_state_index] = value;
    break;

  case ALQUIMIA_STRUCT_SURFACE_SITE_DENSITY:
    if (state->surface_site_density.data == NULL || mapping.alquimia_state_index >= state->surface_site_density.size)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds surface_site_density write: index %d, size %d.", mapping.alquimia_state_index, state->surface_site_density.size);
      return;
    }
    state->surface_site_density.data[mapping.alquimia_state_index] = value;
    break;

  case ALQUIMIA_STRUCT_CATION_EXCHANGE_CAPACITY:
    if (state->cation_exchange_capacity.data == NULL || mapping.alquimia_state_index >= state->cation_exchange_capacity.size)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds cation_exchange_capacity write: index %d, size %d.", mapping.alquimia_state_index, state->cation_exchange_capacity.size);
      return;
    }
    state->cation_exchange_capacity.data[mapping.alquimia_state_index] = value;
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
    if (state->gas_concentration.data == NULL || mapping.alquimia_state_index >= state->gas_concentration.size)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds gas_concentration write: index %d, size %d.", mapping.alquimia_state_index, state->gas_concentration.size);
      return;
    }
    state->gas_concentration.data[mapping.alquimia_state_index] = value;
    break;

  default:
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Unknown mapped struct type: %d.", mapping.alquimia_state);
    break;
  }
}

/**
 * @brief Checks whether the model explicitly outputs one state element.
 * @param[in] onnx_state Engine containing the flattened output mappings.
 * @param[in] alquimia_state State field to find.
 * @param[in] alquimia_state_index Zero-based state-vector index to find.
 * @return True when an output tensor element maps to the requested state element.
 */
static bool HasMappedOutput(
    const OnnxEngineState *onnx_state,
    AlquimiaMappedStruct alquimia_state,
    int alquimia_state_index)
{
  size_t i;

  for (i = 0; i < onnx_state->total_flat_outputs; ++i)
  {
    const FeatureMapping *mapping = &onnx_state->output_mappings[i];
    if (mapping->alquimia_state == alquimia_state &&
        mapping->alquimia_state_index == alquimia_state_index)
    {
      return true;
    }
  }
  return false;
}

/**
 * @brief Caches whether each output explicitly maps its paired phase.
 * @param[in,out] onnx_state Engine containing the completed output mappings.
 * @param[out] status Returns an allocation error for the lookup array.
 * @return True when the output-aligned lookup is initialized.
 *
 * The one-time setup scan keeps paired-output detection out of the repeated
 * inference path.
 */
static bool BuildPairedOutputLookup(
    OnnxEngineState *onnx_state,
    AlquimiaEngineStatus *status)
{
  size_t i;

  onnx_state->output_has_paired_mapping = (bool *)calloc(
      onnx_state->total_flat_outputs,
      sizeof(*onnx_state->output_has_paired_mapping));
  if (onnx_state->output_has_paired_mapping == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Memory allocation failed for ONNX paired output lookup.");
    return false;
  }

  for (i = 0; i < onnx_state->total_flat_outputs; ++i)
  {
    const FeatureMapping *mapping = &onnx_state->output_mappings[i];
    AlquimiaMappedStruct paired_state;

    if (mapping->alquimia_state == ALQUIMIA_STRUCT_TOTAL_MOBILE)
    {
      paired_state = ALQUIMIA_STRUCT_TOTAL_IMMOBILE;
    }
    else if (mapping->alquimia_state == ALQUIMIA_STRUCT_TOTAL_IMMOBILE)
    {
      paired_state = ALQUIMIA_STRUCT_TOTAL_MOBILE;
    }
    else
    {
      continue;
    }
    onnx_state->output_has_paired_mapping[i] = HasMappedOutput(
        onnx_state, paired_state, mapping->alquimia_state_index);
  }
  return true;
}

/**
 * @brief Writes one model output and preserves a paired component total.
 * @param[in,out] state State receiving the model output and any paired update.
 * @param[in] mapping Validated model-output destination.
 * @param[in] has_paired_mapping Whether the paired phase is an explicit output.
 * @param[in] value Model output to assign.
 * @param[out] status Returns mapped-state access errors.
 *
 * When a model outputs only one of total_mobile[i] or total_immobile[i], the
 * paired value changes by the opposite amount so their pre-inference sum is
 * conserved. Explicit model outputs for both values remain authoritative.
 */
static void SetAlquimiaModelOutput(
    AlquimiaState *state,
    FeatureMapping mapping,
    bool has_paired_mapping,
    double value,
    AlquimiaEngineStatus *status)
{
  // Formula: new_mobile + new_immobile = old_mobile + old_immobile
  int index = mapping.alquimia_state_index;
  FeatureMapping paired_mapping;
  double ncomp;
  double mapped_value;
  double paired_value;

  paired_mapping = mapping;

  // The model runs the inference for mobile
  if (mapping.alquimia_state == ALQUIMIA_STRUCT_TOTAL_MOBILE)
  {
    paired_mapping.alquimia_state = ALQUIMIA_STRUCT_TOTAL_IMMOBILE;

    // Invalid immobile
    if (state->total_immobile.data == NULL || index < 0 ||
        index >= state->total_immobile.size)
    {
      SetAlquimiaValue(state, mapping, value, status);
      return;
    }
  }
  else if (mapping.alquimia_state == ALQUIMIA_STRUCT_TOTAL_IMMOBILE)
  {
    paired_mapping.alquimia_state = ALQUIMIA_STRUCT_TOTAL_MOBILE;
    if (state->total_mobile.data == NULL || index < 0 ||
        index >= state->total_mobile.size)
    {
      SetAlquimiaValue(state, mapping, value, status);
      return;
    }
  }
  else
  {
    // Neither mobile nor immobile
    SetAlquimiaValue(state, mapping, value, status);
    return;
  }

  /* Explicit model outputs for both phases remain authoritative. */
  if (has_paired_mapping)
  {
    SetAlquimiaValue(state, mapping, value, status);
    return;
  }

  mapped_value = GetAlquimiaValue(state, mapping, status);
  if (status->error != kAlquimiaNoError)
  {
    return;
  }
  paired_value = GetAlquimiaValue(state, paired_mapping, status);
  if (status->error != kAlquimiaNoError)
  {
    return;
  }
  ncomp = mapped_value + paired_value;

  SetAlquimiaValue(state, mapping, value, status);
  if (status->error != kAlquimiaNoError)
  {
    return;
  }
  SetAlquimiaValue(state, paired_mapping, ncomp - value, status);
}

/**
 * @brief Releases one input or output tensor collection.
 * @param[in,out] tensors Tensor metadata and buffers to release.
 * @param[in] ort ONNX Runtime objects used to release names and OrtValue objects.
 */
static void ReleaseOrtTensors(OrtTensor *tensors, Ort *ort)
{
  size_t i;

  if (tensors->tensor != NULL)
  {
    for (i = 0; i < tensors->num_tensors; ++i)
    {
      if (tensors->tensor[i] != NULL)
      {
        ort->g_ort->ReleaseValue(tensors->tensor[i]);
      }
    }
    free(tensors->tensor);
  }
  if (tensors->data != NULL)
  {
    for (i = 0; i < tensors->num_tensors; ++i)
    {
      free(tensors->data[i]);
    }
    free(tensors->data);
  }
  if (tensors->dim_values != NULL)
  {
    for (i = 0; i < tensors->num_tensors; ++i)
    {
      free(tensors->dim_values[i]);
    }
    free(tensors->dim_values);
  }
  if(tensors->num_dim != NULL)
  {
    free(tensors->num_dim);
  }
  if(tensors->total_size)
  {
    free(tensors->total_size);
  }

  /* ONNX Runtime allocates tensor names with the OrtAllocator. */
  if(ort->allocator != NULL)
  {
    if (tensors->names != NULL)
  {
    for (i = 0; i < tensors->num_tensors; ++i)
    {
      if (tensors->names[i] != NULL)
      {
        ort->allocator->Free(ort->allocator, tensors->names[i]);
      }
    }
    free(tensors->names);
  }
  }
  memset(tensors, 0, sizeof(*tensors));
}

/**
 * @brief Releases a partially initialized engine without replacing setup error details.
 * @param[in,out] onnx_state Address of the local engine pointer. Shutdown releases all
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

/* ------------------------ ALQUIMIA FUNCTIONS -------------------------------*/

/**
 * @brief Initializes an ONNX engine from a versioned JSON sidecar onnx_config.
 * @param[in] input_filename JSON config path; relative model paths resolve from its
 *        directory.
 * @param[in] hands_off Selects named JSON conditions for initialization.
 * @param[in,out] onnx_engine_state Address that receives the initialized engine.
 * @param[in,out] sizes Returns state-vector sizes derived from explicit mappings.
 * @param[out] functionality Returns the ONNX adapter capability flags.
 * @param[out] status Returns setup errors without being overwritten by cleanup.
 *
 * The destination engine pointer is set to NULL before resource acquisition and
 * remains NULL on every failure. Parsed config storage remains engine-owned so
 * named conditions are available to ProcessCondition.
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
  /* Read the JSON file that contains that serves as input 
  ** and provides the metadata mapping model to alquimia struct */
  if (!OnnxAlquimiaLoadConfig(
          input_filename, &onnx_state->onnx_config, status->message,
          kAlquimiaMaxStringLength))
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    free(onnx_state);
    return;
  }
  onnx_state->hands_off = hands_off;
  if (hands_off && onnx_state->onnx_config.num_conditions == 0)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "ONNX hands-off setup requires at least one JSON condition.");
    OnnxAlquimiaFreeConfig(&onnx_state->onnx_config);
    free(onnx_state);
    return;
  }
  f = fopen(onnx_state->onnx_config.model_path, "rb");
  if (f == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "ONNX model file not found: %s",
             onnx_state->onnx_config.model_path);
    OnnxAlquimiaFreeConfig(&onnx_state->onnx_config);
    free(onnx_state);
    return;
  }
  fclose(f);

  onnx_state->ort.g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (!onnx_state->ort.g_ort)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Failed to load ONNX Runtime API.");
    OnnxAlquimiaFreeConfig(&onnx_state->onnx_config);
    free(onnx_state);
    return;
  }

  /* Read the ONNX file that contains that contains the model */ 
  ort_status = onnx_state->ort.g_ort->CreateEnv(
      ORT_LOGGING_LEVEL_WARNING, "onnx_alquimia_engine",
      &onnx_state->ort.env);
  if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
  {
    OnnxAlquimiaFreeConfig(&onnx_state->onnx_config);
    free(onnx_state);
    return;
  }

  ort_status = onnx_state->ort.g_ort->CreateSessionOptions(
      &onnx_state->ort.session_options);
  if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
  {
    onnx_state->ort.g_ort->ReleaseEnv(onnx_state->ort.env);
    OnnxAlquimiaFreeConfig(&onnx_state->onnx_config);
    free(onnx_state);
    return;
  }

  ort_status = onnx_state->ort.g_ort->CreateSession(
      onnx_state->ort.env, onnx_state->onnx_config.model_path,
      onnx_state->ort.session_options, &onnx_state->ort.session);
  if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
  {
    onnx_state->ort.g_ort->ReleaseSessionOptions(
        onnx_state->ort.session_options);
    onnx_state->ort.g_ort->ReleaseEnv(onnx_state->ort.env);
    OnnxAlquimiaFreeConfig(&onnx_state->onnx_config);
    free(onnx_state);
    return;
  }

  ort_status = onnx_state->ort.g_ort->CreateCpuMemoryInfo(
      OrtArenaAllocator, OrtMemTypeDefault,
      &onnx_state->ort.memory_info);
  if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
  {
    CleanupOnSetupFailure(&onnx_state);
    return;
  }

  ort_status = onnx_state->ort.g_ort->CreateAllocator(
      onnx_state->ort.session, onnx_state->ort.memory_info,
      &onnx_state->ort.allocator);
  if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
  {
    CleanupOnSetupFailure(&onnx_state);
    return;
  }

  /* Query input/output tensor count */
  size_t num_inputs = 0;
  ort_status = onnx_state->ort.g_ort->SessionGetInputCount(
      onnx_state->ort.session, &num_inputs);
  if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
  {
    CleanupOnSetupFailure(&onnx_state);
    return;
  }
  onnx_state->input_tensors.num_tensors = num_inputs;

  size_t num_outputs = 0;
  ort_status = onnx_state->ort.g_ort->SessionGetOutputCount(
      onnx_state->ort.session, &num_outputs);
  if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
  {
    CleanupOnSetupFailure(&onnx_state);
    return;
  }
  onnx_state->output_tensors.num_tensors = num_outputs;

  /* Guard against empty inputs or outputs */
  if (num_inputs == 0 || num_outputs == 0)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Model has 0 inputs or outputs (inputs: %d, outputs: %d).", (int)num_inputs, (int)num_outputs);
    CleanupOnSetupFailure(&onnx_state);
    return;
  }

  /* Allocate outer arrays for inputs */
  onnx_state->input_tensors.names =
      (char **)calloc(num_inputs, sizeof(char *));
  onnx_state->input_tensors.num_dim =
      (size_t *)calloc(num_inputs, sizeof(size_t));
  onnx_state->input_tensors.dim_values =
      (int64_t **)calloc(num_inputs, sizeof(int64_t *));
  onnx_state->input_tensors.total_size =
      (size_t *)calloc(num_inputs, sizeof(size_t));
  onnx_state->input_tensors.data =
      (double **)calloc(num_inputs, sizeof(double *));
  onnx_state->input_tensors.tensor =
      (OrtValue **)calloc(num_inputs, sizeof(OrtValue *));

  /* Guard */
  if (onnx_state->input_tensors.names == NULL ||
      onnx_state->input_tensors.num_dim == NULL ||
      onnx_state->input_tensors.dim_values == NULL ||
      onnx_state->input_tensors.total_size == NULL ||
      onnx_state->input_tensors.data == NULL ||
      onnx_state->input_tensors.tensor == NULL)
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
    ort_status = onnx_state->ort.g_ort->SessionGetInputName(
        onnx_state->ort.session, i, onnx_state->ort.allocator, &name);
    if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
    {
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    onnx_state->input_tensors.names[i] = name;

    OrtTypeInfo *ort_type_info = NULL;  /* Released by ReleaseTypeInfo */
    ort_status = onnx_state->ort.g_ort->SessionGetInputTypeInfo(
        onnx_state->ort.session, i, &ort_type_info);
    if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
    {
      CleanupOnSetupFailure(&onnx_state);
      return;
    }

    const OrtTensorTypeAndShapeInfo *tensor_info = NULL;  /* Do not free this value. It will be valid until ort_type_info is freed */
    /* Check if the input data type is double */
    ONNXTensorElementDataType onnx_element_type; /* No need to free. enum */
    ort_status = onnx_state->ort.g_ort->CastTypeInfoToTensorInfo(
        ort_type_info, &tensor_info);
    if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
    {
      onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    /* Get the input data type */
    ort_status = onnx_state->ort.g_ort->GetTensorElementType(
        tensor_info, &onnx_element_type);
    if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
    {
      onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    /* Check if the input data type is double */
    if (onnx_element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)
    {
      onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "ONNX input tensor '%s' must have double elements.",
               onnx_state->input_tensors.names[i]);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    /* Record the dimension number */
    size_t dim_count = 0;
    /* Get the number of the dimensions */
    ort_status = onnx_state->ort.g_ort->GetDimensionsCount(
        tensor_info, &dim_count);
    if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
    {
      onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }

    /* Preserve the model rank. A scalar has zero dimensions and keeps the
    ** NULL dimension array initialized by calloc. */
    onnx_state->input_tensors.num_dim[i] = dim_count;
    if (dim_count > 0)
    {
      onnx_state->input_tensors.dim_values[i] =
          (int64_t *)calloc(dim_count, sizeof(int64_t));
      if (onnx_state->input_tensors.dim_values[i] == NULL)
      {
        onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
        status->error = kAlquimiaErrorEngineIntegrity;
        snprintf(status->message, kAlquimiaMaxStringLength, "Memory allocation failed for input_dim_values.");
        CleanupOnSetupFailure(&onnx_state);
        return;
      }
      /* Get the dimensions */
      ort_status = onnx_state->ort.g_ort->GetDimensions(
          tensor_info, onnx_state->input_tensors.dim_values[i], dim_count);
      if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
      {
        onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
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
    for (j = 0; j < onnx_state->input_tensors.num_dim[i]; ++j)
    {
      /* [-1, feature numbers] */
      if (onnx_state->input_tensors.dim_values[i][j] <= 0)
      {
        /* Only when input_num_dim[i][0] == -1 */
        if (j != 0 || onnx_state->input_tensors.num_dim[i] == 1)
        {
          onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
          status->error = kAlquimiaErrorEngineIntegrity;
          snprintf(status->message, kAlquimiaMaxStringLength,
                   "ONNX input tensor '%s' has an unsupported dynamic extent.",
                   onnx_state->input_tensors.names[i]);
          CleanupOnSetupFailure(&onnx_state);
          return;
        }
        onnx_state->input_tensors.dim_values[i][j] = 1;
      }
      /* Guard
      ** Check if feature numbers * total size > size_t
      ** In this case, feature numbers * batch size > size_t
      */
      if ((size_t)onnx_state->input_tensors.dim_values[i][j] >
          SIZE_MAX / total_size)
      {
        onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
        status->error = kAlquimiaErrorEngineIntegrity;
        snprintf(status->message, kAlquimiaMaxStringLength,
                 "ONNX input tensor '%s' element count overflows size_t.",
                 onnx_state->input_tensors.names[i]);
        CleanupOnSetupFailure(&onnx_state);
        return;
      }
      total_size *= (size_t)onnx_state->input_tensors.dim_values[i][j];
    }
    onnx_state->input_tensors.total_size[i] = total_size;

    /* allocate memory for the input data */
    onnx_state->input_tensors.data[i] =
        (double *)calloc(total_size, sizeof(double));
    if (onnx_state->input_tensors.data[i] == NULL)
    {
      onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Memory allocation failed for input_data.");
      CleanupOnSetupFailure(&onnx_state);
      return;
    }

    /* The OrtValue wraps input_data; ONNX Runtime does not own that buffer. */
    ort_status = onnx_state->ort.g_ort->CreateTensorWithDataAsOrtValue(
        onnx_state->ort.memory_info, onnx_state->input_tensors.data[i],
        total_size * sizeof(double), onnx_state->input_tensors.dim_values[i],
        onnx_state->input_tensors.num_dim[i],
        ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
        &onnx_state->input_tensors.tensor[i]);
    if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
    {
      onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    /* Release the type info */
    onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
  }

  onnx_state->output_tensors.names =
      (char **)calloc(num_outputs, sizeof(char *));
  onnx_state->output_tensors.num_dim =
      (size_t *)calloc(num_outputs, sizeof(size_t));
  onnx_state->output_tensors.dim_values =
      (int64_t **)calloc(num_outputs, sizeof(int64_t *));
  onnx_state->output_tensors.total_size =
      (size_t *)calloc(num_outputs, sizeof(size_t));
  onnx_state->output_tensors.data =
      (double **)calloc(num_outputs, sizeof(double *));
  onnx_state->output_tensors.tensor =
      (OrtValue **)calloc(num_outputs, sizeof(OrtValue *));

  if (onnx_state->output_tensors.names == NULL ||
      onnx_state->output_tensors.num_dim == NULL ||
      onnx_state->output_tensors.dim_values == NULL ||
      onnx_state->output_tensors.total_size == NULL ||
      onnx_state->output_tensors.data == NULL ||
      onnx_state->output_tensors.tensor == NULL)
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
    ort_status = onnx_state->ort.g_ort->SessionGetOutputName(
        onnx_state->ort.session, i, onnx_state->ort.allocator, &name);
    if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
    {
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    onnx_state->output_tensors.names[i] = name;

    OrtTypeInfo *ort_type_info = NULL;  /* Released by ReleaseTypeInfo */
    ort_status = onnx_state->ort.g_ort->SessionGetOutputTypeInfo(
        onnx_state->ort.session, i, &ort_type_info);
    if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
    {
      CleanupOnSetupFailure(&onnx_state);
      return;
    }

    const OrtTensorTypeAndShapeInfo *tensor_info = NULL;  /* Do not free this value. It will be valid until ort_type_info is freed */
    /* Check if the output data type is double */
    ONNXTensorElementDataType onnx_element_type; /* No need to free. enum */
    ort_status = onnx_state->ort.g_ort->CastTypeInfoToTensorInfo(
        ort_type_info, &tensor_info);
    if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
    {
      onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    /* Get the output data type */
    ort_status = onnx_state->ort.g_ort->GetTensorElementType(
        tensor_info, &onnx_element_type);
    if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
    {
      onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    /* Check if the output data type is double (important for geoscience) */
    if (onnx_element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)
    {
      onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "ONNX output tensor '%s' must have double elements.",
               onnx_state->output_tensors.names[i]);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    /* Record the dimension number */
    size_t dim_count = 0;
    /* Get the number of the dimensions */
    ort_status = onnx_state->ort.g_ort->GetDimensionsCount(
        tensor_info, &dim_count);
    if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
    {
      onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }

    /* For a scalar:
    **
    ** - num_dim = 0
    ** - dim_values = NULL
    ** - total_size = 1
    ** - data[i] holds one double
    ** - OrtValue *tensor[i] is a valid rank-0 tensor

    ** Before OrtRun, routing writes the scalar to data[i][0]. The OrtValue already
    ** references that buffer, so inference reads the assigned value directly. Only
    ** the shape pointer is NULL, indicating rank 0. 
    */
    /* Preserve the model rank. A scalar has zero dimensions and keeps the
    ** NULL dimension array initialized by calloc. */
    onnx_state->output_tensors.num_dim[i] = dim_count;
    if (dim_count > 0)
    {
      onnx_state->output_tensors.dim_values[i] =
          (int64_t *)calloc(dim_count, sizeof(int64_t));
      if (onnx_state->output_tensors.dim_values[i] == NULL)
      {
        onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
        status->error = kAlquimiaErrorEngineIntegrity;
        snprintf(status->message, kAlquimiaMaxStringLength, "Memory allocation failed for output_dim_values.");
        CleanupOnSetupFailure(&onnx_state);
        return;
      }
      /* Get the dimensions */
      ort_status = onnx_state->ort.g_ort->GetDimensions(
          tensor_info, onnx_state->output_tensors.dim_values[i], dim_count);
      if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
      {
        onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
        CleanupOnSetupFailure(&onnx_state);
        return;
      }
    }

    /* Match the single-state batch convention used for outputs. */
    /* Calculate the total output numbers for each tensor */
    size_t total_size = 1; /* batch_size */
    /* For the output dimensions [] */
    for (j = 0; j < onnx_state->output_tensors.num_dim[i]; ++j)
    {
      /* [-1, feature numbers] */
      if (onnx_state->output_tensors.dim_values[i][j] <= 0)
      {
        /* Only when output_num_dim[i][0] == -1 */
        if (j != 0 || onnx_state->output_tensors.num_dim[i] == 1)
        {
          onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
          status->error = kAlquimiaErrorEngineIntegrity;
          snprintf(status->message, kAlquimiaMaxStringLength,
                   "ONNX output tensor '%s' has an unsupported dynamic extent.",
                   onnx_state->output_tensors.names[i]);
          CleanupOnSetupFailure(&onnx_state);
          return;
        }
        onnx_state->output_tensors.dim_values[i][j] = 1;
      }
      /* Guard
      ** Check if feature numbers * total size > size_t
      ** In this case, feature numbers * batch size > size_t
      */
      if ((size_t)onnx_state->output_tensors.dim_values[i][j] >
          SIZE_MAX / total_size)
      {
        onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
        status->error = kAlquimiaErrorEngineIntegrity;
        snprintf(status->message, kAlquimiaMaxStringLength,
                 "ONNX output tensor '%s' element count overflows size_t.",
                 onnx_state->output_tensors.names[i]);
        CleanupOnSetupFailure(&onnx_state);
        return;
      }
      total_size *= (size_t)onnx_state->output_tensors.dim_values[i][j];
    }
    onnx_state->output_tensors.total_size[i] = total_size;

    /* allocate memory for the output data */
    onnx_state->output_tensors.data[i] =
        (double *)calloc(total_size, sizeof(double));
    if (onnx_state->output_tensors.data[i] == NULL)
    {
      onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Memory allocation failed for output_data.");
      CleanupOnSetupFailure(&onnx_state);
      return;
    }

    /* The OrtValue wraps output_data; ONNX Runtime does not own that buffer. */
    ort_status = onnx_state->ort.g_ort->CreateTensorWithDataAsOrtValue(
        onnx_state->ort.memory_info, onnx_state->output_tensors.data[i],
        total_size * sizeof(double),
        onnx_state->output_tensors.dim_values[i],
        onnx_state->output_tensors.num_dim[i],
        ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
        &onnx_state->output_tensors.tensor[i]);
    if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
    {
      onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    /* Release the type info */
    onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
  }

  /* functionality is hardwired for now. Mostly fine but need to revisit */
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
  for (i = 0; i < onnx_state->input_tensors.num_tensors; ++i)
  {
    /* Guard
    ** avoid total feature inputs > size_t
    */
    if (onnx_state->input_tensors.total_size[i] >
        SIZE_MAX - onnx_state->total_flat_inputs)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "Flattened ONNX input element count overflows size_t.");
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    onnx_state->total_flat_inputs += onnx_state->input_tensors.total_size[i];
  }

  for (i = 0; i < onnx_state->output_tensors.num_tensors; ++i)
  {
    /* Guard
    ** avoid total feature outputs > size_t
    */
    if (onnx_state->output_tensors.total_size[i] >
        SIZE_MAX - onnx_state->total_flat_outputs)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "Flattened ONNX output element count overflows size_t.");
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    onnx_state->total_flat_outputs +=
        onnx_state->output_tensors.total_size[i];
  }

  /* Set up the mapping rules inside the OnnxEngine->mapping struct 
  ** Initialize AlquimiaSize
  */
  if (!BuildConfigMappings(onnx_state, sizes, status) ||
      // Check if the paired outputs for mobile and immobile exist
      !BuildPairedOutputLookup(onnx_state, status))
  {
    CleanupOnSetupFailure(&onnx_state);
    return;
  }
 
  *(OnnxEngineState **)onnx_engine_state = onnx_state;
}

/**
 * @brief Releases all adapter and ONNX Runtime resources.
 * @param[in,out] onnx_engine_state Address of the engine pointer created by setup.
 * @param[out] status Returns an invalid-engine error for a NULL engine.
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

  /* 
  ** Memory Lifecycle & Cleanup Rules:
  ** 1. Input/output tensors and underlying data arrays are created via 
  **    independent API calls during setup and can be freed in any order.
  ** 2. Core ONNX Runtime objects (session, env, etc.) maintain strict 
  **    dependencies and must be released in reverse order of initialization (LIFO).
  */
  if (onnx_engine_state == NULL || *(OnnxEngineState **)onnx_engine_state == NULL)
  {
    status->error = kAlquimiaErrorInvalidEngine;
    snprintf(status->message, kAlquimiaMaxStringLength, "Invalid ONNX engine state pointer in shutdown.");
    return;
  }

  onnx_state = *(OnnxEngineState **)onnx_engine_state;
  if (onnx_state->ort.g_ort != NULL)
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
      for (i = 0; i < onnx_state->total_flat_outputs; ++i)
      {
        free(onnx_state->output_mappings[i].feature);
      }
      free(onnx_state->output_mappings);
      onnx_state->output_mappings = NULL;
    }
    free(onnx_state->output_has_paired_mapping);
    onnx_state->output_has_paired_mapping = NULL;

    /* Input and output collections own independent tensors and buffers. */
    ReleaseOrtTensors(&onnx_state->input_tensors, &onnx_state->ort);
    ReleaseOrtTensors(&onnx_state->output_tensors, &onnx_state->ort);

    /* Release core ONNX Runtime objects in reverse initialization order. */
    if (onnx_state->ort.allocator != NULL)
    {
      onnx_state->ort.g_ort->ReleaseAllocator(onnx_state->ort.allocator);
      onnx_state->ort.allocator = NULL;
    }
    if (onnx_state->ort.memory_info != NULL)
    {
      onnx_state->ort.g_ort->ReleaseMemoryInfo(
          onnx_state->ort.memory_info);
      onnx_state->ort.memory_info = NULL;
    }
    if (onnx_state->ort.session != NULL)
    {
      onnx_state->ort.g_ort->ReleaseSession(onnx_state->ort.session);
      onnx_state->ort.session = NULL;
    }
    if (onnx_state->ort.session_options != NULL)
    {
      onnx_state->ort.g_ort->ReleaseSessionOptions(
          onnx_state->ort.session_options);
      onnx_state->ort.session_options = NULL;
    }
    if (onnx_state->ort.env != NULL)
    {
      onnx_state->ort.g_ort->ReleaseEnv(onnx_state->ort.env);
      onnx_state->ort.env = NULL;
    }
  }
  OnnxAlquimiaFreeConfig(&onnx_state->onnx_config);
  free(onnx_state);
  *(OnnxEngineState **)onnx_engine_state = NULL;
}

/**
 * @brief Applies JSON or driver conditions to config-mapped model inputs.
 * @param[in] onnx_engine_state Address of an initialized engine pointer.
 * @param[in] condition Condition whose exact name selects JSON values in hands-off
 *        mode, or whose aqueous constraints supply values in normal mode.
 * @param[in] properties [Unused] by the ONNX adapter.
 * @param[in,out] state State receiving values for matching input feature names.
 * @param[in] aux_data [Unused] by the ONNX adapter.
 * @param[out] status Returns invalid-engine or mapped-state access errors.
 *
 * Hands-off mode ignores driver constraint values and requires an exact JSON
 * condition-name match. Normal mode preserves the generic constraint behavior:
 * absent constraints leave their mapped state values unchanged.
 */
void onnx_alquimia_processcondition(
    void *onnx_engine_state,
    AlquimiaGeochemicalCondition *condition,
    AlquimiaProperties *properties,
    AlquimiaState *state,
    AlquimiaAuxiliaryData *aux_data,
    AlquimiaEngineStatus *status)
{
  OnnxEngineState *onnx_state;
  size_t k;

  status->error = kAlquimiaNoError;
  status->message[0] = '\0';

  (void)properties;
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

  // Initial condition provided by JSON
  if (onnx_state->hands_off)
  {
    // Point to the related onnx_config.conditions
    const OnnxAlquimiaCondition *matching_condition = NULL;
    size_t condition_index;
    
    // Passed by the driver
    if (condition != NULL && condition->name != NULL)
    {
      for (condition_index = 0;
           condition_index < onnx_state->onnx_config.num_conditions;
           ++condition_index)
      {
        const OnnxAlquimiaCondition *current_condition =
            &onnx_state->onnx_config.conditions[condition_index];
        // Find the related initial condition in the JSON file
        if (strcmp(current_condition->name, condition->name) == 0)
        {
          matching_condition = current_condition;
          break;
        }
      }
    }
    if (matching_condition == NULL)
    {
      status->error = kAlquimiaErrorUnknownConstraintName;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "Unknown ONNX JSON condition name '%s'.",
               condition != NULL && condition->name != NULL
                   ? condition->name
                   : "");
      return;
    }

    for (k = 0; k < onnx_state->total_flat_inputs; ++k)
    {
      const char *feature = onnx_state->input_mappings[k].feature;
      // Record the key: value pair in JSON
      const OnnxAlquimiaConditionItem *matching_item = NULL;
      size_t item_index;

      for (item_index = 0;
           item_index < matching_condition->num_items;
           ++item_index)
      {
        if (strcmp(matching_condition->items[item_index].feature,
                   feature) == 0)
        {
          matching_item = &matching_condition->items[item_index];
          break;
        }
      }
      // Fail to find the key: value pair
      if (matching_item == NULL)
      {
        status->error = kAlquimiaErrorEngineIntegrity;
        snprintf(status->message, kAlquimiaMaxStringLength,
                 "ONNX JSON condition '%s' is missing input feature '%s'.",
                 matching_condition->name, feature);
        return;
      }
      SetAlquimiaValue(state, onnx_state->input_mappings[k],
                       matching_item->value, status);
      if (status->error != kAlquimiaNoError)
      {
        return;
      }
    }
    return;
  }

  // Initial condition provided by driver
  /* An absent condition intentionally preserves all existing state values. */
  if (condition == NULL || condition->aqueous_constraints.data == NULL || condition->aqueous_constraints.size <= 0)
  {
    return;
  }

  /* Map matching aqueous constraint values into the Alquimia state 
  ** based on the pre-validated input features. */
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
 * @param[in,out] onnx_engine_state Address of an initialized engine pointer.
 * @param[in] delta_t [Unused]; the model receives only explicitly mapped state data.
 * @param[in] properties [Unused] by the ONNX adapter.
 * @param[in,out] state Supplies mapped inputs and receives mapped outputs.
 * @param[in] aux_data [Unused] by the ONNX adapter.
 * @param[in] natural_id [Unused] by the ONNX adapter.
 * @param[out] status Returns state-access or ONNX Runtime errors.
 *
 * The engine reuses mutable tensor buffers allocated during setup. Calls that
 * share one engine instance are therefore not thread-safe.
 */
void onnx_alquimia_reactionstepoperatorsplit(
    void *onnx_engine_state,
    double delta_t,
    AlquimiaProperties *properties,
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

  /* Input/output tensors will be reused
  ** They will be freed in the shutdown*/
  // Unused
  (void)delta_t;
  (void)properties;
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
    size_t flat_index = 0;
    for (i = 0; i < (int)onnx_state->input_tensors.num_tensors; ++i)
    {
      size_t k;
      for (k = 0; k < onnx_state->input_tensors.total_size[i]; ++k)
      {
        onnx_state->input_tensors.data[i][k] = GetAlquimiaValue(
            state, onnx_state->input_mappings[flat_index], status);
        if (status->error != kAlquimiaNoError)
        {
          return;
        }
        flat_index++;
      }
    }
  }

  /* Run inference using pre-allocated input and output tensors and dynamic names */
  ort_status = onnx_state->ort.g_ort->Run(
      onnx_state->ort.session,
      NULL, /* RunOptions */
      (const char *const *)onnx_state->input_tensors.names,
      (const OrtValue *const *)onnx_state->input_tensors.tensor,
      onnx_state->input_tensors.num_tensors,
      (const char *const *)onnx_state->output_tensors.names,
      onnx_state->output_tensors.num_tensors,
      onnx_state->output_tensors.tensor);

  if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
  {
    return;
  }

  /* Copy output data back through the required explicit mappings. */
  {
    size_t flat_index = 0;
    for (i = 0; i < (int)onnx_state->output_tensors.num_tensors; ++i)
    {
      /* Temporary output array used to store the data from the tensor */
      double *out_arr = NULL;
      /* Extract the data from the output tensor */
      ort_status = onnx_state->ort.g_ort->GetTensorMutableData(
          onnx_state->output_tensors.tensor[i], (void **)&out_arr);
      if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
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
      for (k = 0; k < onnx_state->output_tensors.total_size[i]; ++k)
      {
        SetAlquimiaModelOutput(
            state, onnx_state->output_mappings[flat_index],
            onnx_state->output_has_paired_mapping[flat_index], out_arr[k],
            status);
        if (status->error != kAlquimiaNoError)
        {
          return;
        }
        flat_index++;
      }
    }
  }
}

/**
 * @brief Implements the generic auxiliary-output hook as a successful no-op.
 *
 * Version 1 ONNX configs map only AlquimiaState fields and define no
 * auxiliary outputs.
 */
void onnx_alquimia_getauxiliaryoutput(
    void *onnx_engine_state,
    AlquimiaProperties *properties,
    AlquimiaState *state,
    AlquimiaAuxiliaryData *aux_data,
    AlquimiaAuxiliaryOutputData *aux_out,
    AlquimiaEngineStatus *status)
{
  status->error = kAlquimiaNoError;
  status->message[0] = '\0';

  // Unused data
  (void)onnx_engine_state;
  (void)properties;
  (void)state;
  (void)aux_data;
  (void)aux_out;
}

/**
 * @brief Copies config input/output feature names into problem metadata vectors.
 * @param[in] onnx_engine_state Address of an initialized engine pointer.
 * @param[in,out] meta_data Metadata storage allocated from the setup-derived sizes.
 * @param[out] status Returns invalid-engine or invalid-destination errors.
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
    /* Directly point to the OnnxEngine-> input_mappings */ 
    FeatureMapping mapping = onnx_state->input_mappings[i];
    AlquimiaVectorString *metadata_name = MetadataNamesForMapping(
        meta_data, mapping.alquimia_state);
    /* All the manipulation is to the OnnxEngine */
    StoreMetadataName(metadata_name, mapping.alquimia_state_index, mapping.feature);
  }
  for (i = 0; i < onnx_state->total_flat_outputs; ++i)
  {
    FeatureMapping mapping = onnx_state->output_mappings[i];
    AlquimiaVectorString *metadata_name = MetadataNamesForMapping(
        meta_data, mapping.alquimia_state);
    StoreMetadataName(metadata_name, mapping.alquimia_state_index,
                      mapping.feature);
  }
}

#endif /* ALQUIMIA_HAVE_ONNX */
