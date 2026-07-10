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

#include <errno.h>
#include <limits.h>
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

#if ALQUIMIA_HAVE_ONNX

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

/* Custom ONNX metadata maps each flattened tensor value to one AlquimiaState
** field. Scalar fields ignore target_index; vector fields use a zero-based
** index into the selected Alquimia vector. */
typedef struct {
  AlquimiaMappedStruct target_struct;
  int target_index;
} FeatureMapping;

typedef struct
{
  const OrtApi *g_ort;
  OrtEnv *env;
  OrtSessionOptions *session_options;
  OrtSession *session;
  OrtModelMetadata *metadata;
  OrtMemoryInfo *memory_info;
  OrtAllocator *allocator;

  /* Dynamic input info */
  size_t num_inputs;
  char **input_names;
  size_t *input_num_dim;
  int64_t **input_dim_values;
  size_t *input_total_size;
  double **input_data;
  OrtValue **input_tensor;

  /* Dynamic output info */
  size_t num_outputs;
  char **output_names;
  size_t *output_num_dim;
  int64_t **output_dim_values;
  size_t *output_total_size;
  double **output_data;
  OrtValue **output_tensor;

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
 * @brief Maps an exact metadata field name to an AlquimiaState destination.
 * @param name Case-sensitive AlquimiaState field name from model metadata.
 * @param target Receives the corresponding mapping enum on success and remains
 *        unchanged when @p name is unsupported.
 * @return True when @p name identifies a supported scalar or vector field.
 */
static bool ParseStructName(const char *name, AlquimiaMappedStruct *target) {
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
 * @brief Looks up one value in the model's custom metadata map.
 * @param onnx_state Engine state that owns the metadata and allocator.
 * @param key Exact metadata key to query.
 * @return An ONNX-allocated value on success, or NULL when the key is absent or
 *         lookup fails. The caller must release a non-NULL value with the
 *         allocator stored in @p onnx_state.
 */
static char *LookupMetadata(const OnnxEngineState *onnx_state, const char *key)
{
  char *value = NULL;
  OrtStatus *status = onnx_state->g_ort->ModelMetadataLookupCustomMetadataMap(
      onnx_state->metadata, onnx_state->allocator, key, &value);
  if (status != NULL)
  {
    onnx_state->g_ort->ReleaseStatus(status);
    return NULL;
  }
  return value;
}

/**
 * @brief Parses a complete decimal string as a nonnegative C int.
 * @param value Null-terminated metadata value to parse.
 * @param result Receives the parsed value on success and remains unchanged on
 *        malformed input, overflow, or a negative value.
 * @return True when strtol consumes the complete string and produces a value
 *         in [0, INT_MAX].
 */
static bool ParseNonnegativeInteger(const char *value, int *result)
{
  char *end = NULL;
  long parsed_value;

  errno = 0;
  parsed_value = strtol(value, &end, 10);
  if (errno == ERANGE || end == value || *end != '\0' ||
      parsed_value < 0 || parsed_value > INT_MAX)
  {
    return false;
  }

  *result = (int)parsed_value;
  return true;
}

/**
 * @brief Applies an optional ONNX metadata value to one Alquimia size field.
 * @param onnx_state Engine state used to query and free metadata values.
 * @param key Metadata key whose value must be a nonnegative integer.
 * @param size_field Existing inferred/default size, overwritten only when the
 *        key is present and valid.
 * @param status Receives an engine integrity error for malformed values.
 * @return True when the key is unavailable or valid; false when a returned
 *         value is invalid. Any value allocated by ONNX is freed before return.
 */
static bool ParseSizeMetadata(
    const OnnxEngineState *onnx_state,
    const char *key,
    int *size_field,
    AlquimiaEngineStatus *status)
{
  char *value = LookupMetadata(onnx_state, key);

  if (value == NULL)
  {
    return true;
  }

  if (!ParseNonnegativeInteger(value, size_field))
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Invalid nonnegative integer '%s' in ONNX metadata key '%s'.",
             value, key);
    onnx_state->allocator->Free(onnx_state->allocator, value);
    return false;
  }

  onnx_state->allocator->Free(onnx_state->allocator, value);
  return true;
}

/**
 * @brief Identifies mappings whose destination is an AlquimiaState scalar.
 * @param target_struct Mapping destination to classify.
 * @return True for scalar state fields, whose required metadata index is zero.
 */
static bool IsScalarMapping(AlquimiaMappedStruct target_struct)
{
  return target_struct == ALQUIMIA_STRUCT_WATER_DENSITY ||
         target_struct == ALQUIMIA_STRUCT_POROSITY ||
         target_struct == ALQUIMIA_STRUCT_TEMPERATURE ||
         target_struct == ALQUIMIA_STRUCT_AQUEOUS_PRESSURE;
}

/**
 * @brief Returns the allocated AlquimiaState capacity for a mapping target.
 * @param target_struct Scalar or vector destination selected by metadata.
 * @param sizes Alquimia dimensions used to allocate state vectors.
 * @return Vector capacity for vector fields, or one for scalar fields.
 */
static int MappingTargetSize(
    AlquimiaMappedStruct target_struct,
    const AlquimiaSizes *sizes)
{
  switch (target_struct)
  {
  case ALQUIMIA_STRUCT_TOTAL_MOBILE:
    return sizes->num_primary;
  case ALQUIMIA_STRUCT_TOTAL_IMMOBILE:
    return sizes->num_sorbed;
  case ALQUIMIA_STRUCT_MINERAL_VOLUME_FRACTION:
  case ALQUIMIA_STRUCT_MINERAL_SPECIFIC_SURFACE_AREA:
    return sizes->num_minerals;
  case ALQUIMIA_STRUCT_SURFACE_SITE_DENSITY:
    return sizes->num_surface_sites;
  case ALQUIMIA_STRUCT_CATION_EXCHANGE_CAPACITY:
    return sizes->num_ion_exchange_sites;
  case ALQUIMIA_STRUCT_GAS_CONCENTRATION:
    return sizes->num_gases;
  default:
    return 1;
  }
}

/**
 * @brief Builds and validates one flattened tensor-to-state mapping.
 * @param onnx_state Engine state used to query and free metadata values.
 * @param mapping_prefix Either "input_feature_map" or "output_feature_map".
 * @param flat_index Zero-based position in the flattened tensor sequence.
 * @param sizes Alquimia dimensions used for destination bounds validation.
 * @param mapping Receives the mapping only after all metadata validates.
 * @param status Receives a precise engine integrity error on failure.
 * @return True when both required keys exist, the field is supported, and its
 *         index matches the scalar or vector destination. Metadata strings are
 *         freed on every return path.
 */
static bool ParseFeatureMapping(
    const OnnxEngineState *onnx_state,
    const char *mapping_prefix,
    size_t flat_index,
    const AlquimiaSizes *sizes,
    FeatureMapping *mapping,
    AlquimiaEngineStatus *status)
{
  char struct_key[128];
  char index_key[128];
  char *struct_value;
  char *index_value;
  AlquimiaMappedStruct target_struct;
  int target_index;
  int target_size;

  snprintf(struct_key, sizeof(struct_key), "%s_%zu_struct",
           mapping_prefix, flat_index);
  snprintf(index_key, sizeof(index_key), "%s_%zu_index",
           mapping_prefix, flat_index);
  struct_value = LookupMetadata(onnx_state, struct_key);
  index_value = LookupMetadata(onnx_state, index_key);

  if (struct_value == NULL || index_value == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Missing required ONNX mapping metadata key '%s'.",
             struct_value == NULL ? struct_key : index_key);
    if (struct_value != NULL)
    {
      onnx_state->allocator->Free(onnx_state->allocator, struct_value);
    }
    if (index_value != NULL)
    {
      onnx_state->allocator->Free(onnx_state->allocator, index_value);
    }
    return false;
  }

  if (!ParseStructName(struct_value, &target_struct))
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Invalid AlquimiaState field '%s' in ONNX metadata key '%s'.",
             struct_value, struct_key);
    onnx_state->allocator->Free(onnx_state->allocator, struct_value);
    onnx_state->allocator->Free(onnx_state->allocator, index_value);
    return false;
  }

  if (!ParseNonnegativeInteger(index_value, &target_index))
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Invalid nonnegative integer '%s' in ONNX metadata key '%s'.",
             index_value, index_key);
    onnx_state->allocator->Free(onnx_state->allocator, struct_value);
    onnx_state->allocator->Free(onnx_state->allocator, index_value);
    return false;
  }

  target_size = MappingTargetSize(target_struct, sizes);
  if ((IsScalarMapping(target_struct) && target_index != 0) ||
      (!IsScalarMapping(target_struct) && target_index >= target_size))
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "ONNX mapping '%s' index %d is incompatible with AlquimiaState field '%s' of size %d.",
             mapping_prefix, target_index, struct_value, target_size);
    onnx_state->allocator->Free(onnx_state->allocator, struct_value);
    onnx_state->allocator->Free(onnx_state->allocator, index_value);
    return false;
  }

  mapping->target_struct = target_struct;
  mapping->target_index = target_index;
  onnx_state->allocator->Free(onnx_state->allocator, struct_value);
  onnx_state->allocator->Free(onnx_state->allocator, index_value);
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
  switch (map.target_struct)
  {
  case ALQUIMIA_STRUCT_TOTAL_MOBILE:
    if (state->total_mobile.data == NULL || map.target_index >= state->total_mobile.size)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds total_mobile access: index %d, size %d.", map.target_index, state->total_mobile.size);
      return 0.0;
    }
    return state->total_mobile.data[map.target_index];

  case ALQUIMIA_STRUCT_TOTAL_IMMOBILE:
    if (state->total_immobile.data == NULL || map.target_index >= state->total_immobile.size)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds total_immobile access: index %d, size %d.", map.target_index, state->total_immobile.size);
      return 0.0;
    }
    return state->total_immobile.data[map.target_index];

  case ALQUIMIA_STRUCT_MINERAL_VOLUME_FRACTION:
      if (state->mineral_volume_fraction.data == NULL || map.target_index >= state->mineral_volume_fraction.size) {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds mineral_volume_fraction access: index %d, size %d.", map.target_index, state->mineral_volume_fraction.size);
      return 0.0;
    }
    return state->mineral_volume_fraction.data[map.target_index];

  case ALQUIMIA_STRUCT_MINERAL_SPECIFIC_SURFACE_AREA:
      if (state->mineral_specific_surface_area.data == NULL || map.target_index >= state->mineral_specific_surface_area.size) {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds mineral_specific_surface_area access: index %d, size %d.", map.target_index, state->mineral_specific_surface_area.size);
      return 0.0;
    }
    return state->mineral_specific_surface_area.data[map.target_index];

  case ALQUIMIA_STRUCT_SURFACE_SITE_DENSITY:
      if (state->surface_site_density.data == NULL || map.target_index >= state->surface_site_density.size) {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds surface_site_density access: index %d, size %d.", map.target_index, state->surface_site_density.size);
      return 0.0;
    }
    return state->surface_site_density.data[map.target_index];

  case ALQUIMIA_STRUCT_CATION_EXCHANGE_CAPACITY:
      if (state->cation_exchange_capacity.data == NULL || map.target_index >= state->cation_exchange_capacity.size) {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds cation_exchange_capacity access: index %d, size %d.", map.target_index, state->cation_exchange_capacity.size);
      return 0.0;
    }
    return state->cation_exchange_capacity.data[map.target_index];

  case ALQUIMIA_STRUCT_POROSITY:
    return state->porosity;

  case ALQUIMIA_STRUCT_TEMPERATURE:
    return state->temperature;

  case ALQUIMIA_STRUCT_AQUEOUS_PRESSURE:
    return state->aqueous_pressure;

  case ALQUIMIA_STRUCT_WATER_DENSITY:
    return state->water_density;

  case ALQUIMIA_STRUCT_GAS_CONCENTRATION:
      if (state->gas_concentration.data == NULL || map.target_index >= state->gas_concentration.size) {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds gas_concentration access: index %d, size %d.", map.target_index, state->gas_concentration.size);
      return 0.0;
    }
    return state->gas_concentration.data[map.target_index];

  default:
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Unknown mapped struct type: %d.", map.target_struct);
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
  switch (map.target_struct)
  {
  case ALQUIMIA_STRUCT_TOTAL_MOBILE:
      if (state->total_mobile.data == NULL || map.target_index >= state->total_mobile.size) {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds total_mobile write: index %d, size %d.", map.target_index, state->total_mobile.size);
      return;
    }
    state->total_mobile.data[map.target_index] = value;
    break;

  case ALQUIMIA_STRUCT_TOTAL_IMMOBILE:
      if (state->total_immobile.data == NULL || map.target_index >= state->total_immobile.size) {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds total_immobile write: index %d, size %d.", map.target_index, state->total_immobile.size);
      return;
    }
    state->total_immobile.data[map.target_index] = value;
    break;

  case ALQUIMIA_STRUCT_MINERAL_VOLUME_FRACTION:
    if (state->mineral_volume_fraction.data == NULL || map.target_index >= state->mineral_volume_fraction.size)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds mineral_volume_fraction write: index %d, size %d.", map.target_index, state->mineral_volume_fraction.size);
      return;
    }
    state->mineral_volume_fraction.data[map.target_index] = value;
    break;

  case ALQUIMIA_STRUCT_MINERAL_SPECIFIC_SURFACE_AREA:
    if (state->mineral_specific_surface_area.data == NULL || map.target_index >= state->mineral_specific_surface_area.size)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds mineral_specific_surface_area write: index %d, size %d.", map.target_index, state->mineral_specific_surface_area.size);
      return;
    }
    state->mineral_specific_surface_area.data[map.target_index] = value;
    break;

  case ALQUIMIA_STRUCT_SURFACE_SITE_DENSITY:
    if (state->surface_site_density.data == NULL || map.target_index >= state->surface_site_density.size)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds surface_site_density write: index %d, size %d.", map.target_index, state->surface_site_density.size);
      return;
    }
    state->surface_site_density.data[map.target_index] = value;
    break;

  case ALQUIMIA_STRUCT_CATION_EXCHANGE_CAPACITY:
    if (state->cation_exchange_capacity.data == NULL || map.target_index >= state->cation_exchange_capacity.size)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds cation_exchange_capacity write: index %d, size %d.", map.target_index, state->cation_exchange_capacity.size);
      return;
    }
    state->cation_exchange_capacity.data[map.target_index] = value;
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
    if (state->gas_concentration.data == NULL || map.target_index >= state->gas_concentration.size)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Out-of-bounds gas_concentration write: index %d, size %d.", map.target_index, state->gas_concentration.size);
      return;
    }
    state->gas_concentration.data[map.target_index] = value;
    break;

  default:
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Unknown mapped struct type: %d.", map.target_struct);
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
  const char *model_path;
  FILE *f;
  size_t i, j;

  status->error = kAlquimiaNoError;
  status->message[0] = '\0';

  // Unused
  (void)hands_off;

  /* Determine and check model file path */
  if (input_filename == NULL || strlen(input_filename) == 0)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Model file path not provided.");
    return;
  }
  model_path = input_filename;

  f = fopen(model_path, "r");
  if (f == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Model file not found: %s", model_path);
    return;
  }
  fclose(f);

  onnx_state = (OnnxEngineState *)calloc(1, sizeof(OnnxEngineState));
  if (onnx_state == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Memory allocation failed for OnnxEngineState.");
    return;
  }

  onnx_state->g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (!onnx_state->g_ort)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Failed to load ONNX Runtime API.");
    free(onnx_state);
    return;
  }

  ort_status = onnx_state->g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "onnx_alquimia_engine", &onnx_state->env);
  if (!CheckStatus(onnx_state->g_ort, ort_status, status))
  {
    free(onnx_state);
    return;
  }

  ort_status = onnx_state->g_ort->CreateSessionOptions(&onnx_state->session_options);
  if (!CheckStatus(onnx_state->g_ort, ort_status, status))
  {
    onnx_state->g_ort->ReleaseEnv(onnx_state->env);
    free(onnx_state);
    return;
  }

  ort_status = onnx_state->g_ort->CreateSession(onnx_state->env, model_path, onnx_state->session_options, &onnx_state->session);
  if (!CheckStatus(onnx_state->g_ort, ort_status, status))
  {
    onnx_state->g_ort->ReleaseSessionOptions(onnx_state->session_options);
    onnx_state->g_ort->ReleaseEnv(onnx_state->env);
    free(onnx_state);
    return;
  }

  ort_status = onnx_state->g_ort->SessionGetModelMetadata(onnx_state->session, &onnx_state->metadata);
  if (!CheckStatus(onnx_state->g_ort, ort_status, status))
  {
    CleanupOnSetupFailure(&onnx_state);
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

  /* Query input/output count */
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
    ort_status = onnx_state->g_ort->SessionGetInputName(onnx_state->session, i, onnx_state->allocator, &name);
    if (!CheckStatus(onnx_state->g_ort, ort_status, status))
    {
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    onnx_state->input_names[i] = name;

    OrtTypeInfo *type_info = NULL;
    ort_status = onnx_state->g_ort->SessionGetInputTypeInfo(onnx_state->session, i, &type_info);
    if (!CheckStatus(onnx_state->g_ort, ort_status, status))
    {
      CleanupOnSetupFailure(&onnx_state);
      return;
    }

    const OrtTensorTypeAndShapeInfo *tensor_info = NULL;
    ort_status = onnx_state->g_ort->CastTypeInfoToTensorInfo(type_info, &tensor_info);
    if (!CheckStatus(onnx_state->g_ort, ort_status, status))
    {
      onnx_state->g_ort->ReleaseTypeInfo(type_info);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }

    size_t dim_count = 0;
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
      ort_status = onnx_state->g_ort->GetDimensions(tensor_info, onnx_state->input_dim_values[i], dim_count);
      if (!CheckStatus(onnx_state->g_ort, ort_status, status))
      {
        onnx_state->g_ort->ReleaseTypeInfo(type_info);
        CleanupOnSetupFailure(&onnx_state);
        return;
      }
    }

    /* Symbolic and dynamic axes are reported as non-positive dimensions.
    ** This interface allocates fixed reusable buffers at setup, so unknown
    ** extents are treated as one element. */
    size_t total_size = 1;
    for (j = 0; j < onnx_state->input_num_dim[i]; ++j)
    {
      if (onnx_state->input_dim_values[i][j] <= 0)
      {
        onnx_state->input_dim_values[i][j] = 1;
      }
      total_size *= (size_t)onnx_state->input_dim_values[i][j];
    }
    onnx_state->input_total_size[i] = total_size;

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

    onnx_state->g_ort->ReleaseTypeInfo(type_info);
  }

  sizes->num_primary = 0;
  for (i = 0; i < num_inputs; ++i)
  {
    if (onnx_state->input_total_size[i] >
        (size_t)(INT_MAX - sizes->num_primary))
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "Flattened ONNX input size exceeds the Alquimia integer size limit.");
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    sizes->num_primary += (int)onnx_state->input_total_size[i];
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
    ort_status = onnx_state->g_ort->SessionGetOutputName(onnx_state->session, i, onnx_state->allocator, &name);
    if (!CheckStatus(onnx_state->g_ort, ort_status, status))
    {
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    onnx_state->output_names[i] = name;

    OrtTypeInfo *type_info = NULL;
    ort_status = onnx_state->g_ort->SessionGetOutputTypeInfo(onnx_state->session, i, &type_info);
    if (!CheckStatus(onnx_state->g_ort, ort_status, status))
    {
      CleanupOnSetupFailure(&onnx_state);
      return;
    }

    const OrtTensorTypeAndShapeInfo *tensor_info = NULL;
    ort_status = onnx_state->g_ort->CastTypeInfoToTensorInfo(type_info, &tensor_info);
    if (!CheckStatus(onnx_state->g_ort, ort_status, status))
    {
      onnx_state->g_ort->ReleaseTypeInfo(type_info);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }

    size_t dim_count = 0;
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
      ort_status = onnx_state->g_ort->GetDimensions(tensor_info, onnx_state->output_dim_values[i], dim_count);
      if (!CheckStatus(onnx_state->g_ort, ort_status, status))
      {
        onnx_state->g_ort->ReleaseTypeInfo(type_info);
        CleanupOnSetupFailure(&onnx_state);
        return;
      }
    }

    /* Symbolic and dynamic axes are reported as non-positive dimensions.
    ** This interface allocates fixed reusable buffers at setup, so unknown
    ** extents are treated as one element. */
    size_t total_size = 1;
    for (j = 0; j < onnx_state->output_num_dim[i]; ++j)
    {
      if (onnx_state->output_dim_values[i][j] <= 0)
      {
        onnx_state->output_dim_values[i][j] = 1;
      }
      total_size *= (size_t)onnx_state->output_dim_values[i][j];
    }
    onnx_state->output_total_size[i] = total_size;

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

    onnx_state->g_ort->ReleaseTypeInfo(type_info);
  }

  sizes->num_sorbed = 0;
  sizes->num_minerals = 0;
  sizes->num_aqueous_complexes = 0;
  sizes->num_aqueous_kinetics = 0;
  sizes->num_surface_sites = 0;
  sizes->num_ion_exchange_sites = 0;
  sizes->num_isotherm_species = 0;
  sizes->num_gases = 0;
  sizes->num_aux_integers = 0;
  sizes->num_aux_doubles = 0;

  /* Overwrite Alquimia sizes from metadata when valid values are present. */
  if (!ParseSizeMetadata(onnx_state, "sizes_num_primary",
                         &sizes->num_primary, status) ||
      !ParseSizeMetadata(onnx_state, "sizes_num_sorbed",
                         &sizes->num_sorbed, status) ||
      !ParseSizeMetadata(onnx_state, "sizes_num_minerals",
                         &sizes->num_minerals, status) ||
      !ParseSizeMetadata(onnx_state, "sizes_num_surface_sites",
                         &sizes->num_surface_sites, status) ||
      !ParseSizeMetadata(onnx_state, "sizes_num_ion_exchange_sites",
                         &sizes->num_ion_exchange_sites, status) ||
      !ParseSizeMetadata(onnx_state, "sizes_num_aqueous_complexes",
                         &sizes->num_aqueous_complexes, status) ||
      !ParseSizeMetadata(onnx_state, "sizes_num_aqueous_kinetics",
                         &sizes->num_aqueous_kinetics, status) ||
      !ParseSizeMetadata(onnx_state, "sizes_num_isotherm_species",
                         &sizes->num_isotherm_species, status) ||
      !ParseSizeMetadata(onnx_state, "sizes_num_gases",
                         &sizes->num_gases, status) ||
      !ParseSizeMetadata(onnx_state, "sizes_num_aux_integers",
                         &sizes->num_aux_integers, status) ||
      !ParseSizeMetadata(onnx_state, "sizes_num_aux_doubles",
                         &sizes->num_aux_doubles, status))
  {
    CleanupOnSetupFailure(&onnx_state);
    return;
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
  size_t total_flat_inputs = 0;
  for (i = 0; i < onnx_state->num_inputs; ++i)
  {
    total_flat_inputs += onnx_state->input_total_size[i];
  }
  onnx_state->total_flat_inputs = total_flat_inputs;

  size_t total_flat_outputs = 0;
  for (i = 0; i < onnx_state->num_outputs; ++i)
  {
    total_flat_outputs += onnx_state->output_total_size[i];
  }
  onnx_state->total_flat_outputs = total_flat_outputs;

  onnx_state->input_mappings =
      (FeatureMapping *)calloc(total_flat_inputs, sizeof(FeatureMapping));
  onnx_state->output_mappings =
      (FeatureMapping *)calloc(total_flat_outputs, sizeof(FeatureMapping));
  if (onnx_state->input_mappings == NULL ||
      onnx_state->output_mappings == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Memory allocation failed for explicit feature mapping arrays.");
    CleanupOnSetupFailure(&onnx_state);
    return;
  }

  for (i = 0; i < total_flat_inputs; ++i)
  {
    if (!ParseFeatureMapping(onnx_state, "input_feature_map", i, sizes,
                             &onnx_state->input_mappings[i], status))
    {
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
  }

  for (i = 0; i < total_flat_outputs; ++i)
  {
    if (!ParseFeatureMapping(onnx_state, "output_feature_map", i, sizes,
                             &onnx_state->output_mappings[i], status))
    {
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
  }

  *(OnnxEngineState **)onnx_engine_state = onnx_state;
}

/*
** Shutdown ONNX Chemistry Engine
*/
void onnx_alquimia_shutdown(
    void *onnx_engine_state,
    AlquimiaEngineStatus *status)
{
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
    if (onnx_state->metadata != NULL)
    {
      onnx_state->g_ort->ReleaseModelMetadata(onnx_state->metadata);
      onnx_state->metadata = NULL;
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
  free(onnx_state);
  *(OnnxEngineState **)onnx_engine_state = NULL;
}

/*
** Process Condition
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

  /* If condition has no aqueous constraints, we have nothing to assign */
  if (condition == NULL || condition->aqueous_constraints.data == NULL || condition->aqueous_constraints.size <= 0)
  {
    return;
  }

  /* Dynamic mapping of geochemical condition aqueous constraints to state fields */
  for (k = 0; k < onnx_state->total_flat_inputs; ++k)
  {
    char key[128];
    char *feature_name;
    int c_idx;
    AlquimiaAqueousConstraint *matching_constraint = NULL;

    snprintf(key, sizeof(key), "feature_%d", (int)k);
    feature_name = LookupMetadata(onnx_state, key);

    if (feature_name == NULL)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "Missing required ONNX feature metadata key '%s'.", key);
      return;
    }

    /* Search for aqueous constraint with matching name */
    for (c_idx = 0; c_idx < condition->aqueous_constraints.size; ++c_idx)
    {
      if (strcmp(condition->aqueous_constraints.data[c_idx].primary_species_name, feature_name) == 0)
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
        onnx_state->allocator->Free(onnx_state->allocator, feature_name);
        return;
      }
    }

    onnx_state->allocator->Free(onnx_state->allocator, feature_name);
  }
}

/*
** Operator-split reaction step
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

/*
** Get Problem Metadata
*/
void onnx_alquimia_getproblemmetadata(
    void *onnx_engine_state,
    AlquimiaProblemMetaData *meta_data,
    AlquimiaEngineStatus *status)
{
  OnnxEngineState *onnx_state;
  int64_t num_keys;
  char **keys;
  OrtStatus *ort_status;
  int i, j;
  bool debug = false;

  status->error = kAlquimiaNoError;
  status->message[0] = '\0';

  if (onnx_engine_state == NULL || *(OnnxEngineState **)onnx_engine_state == NULL)
  {
    status->error = kAlquimiaErrorInvalidEngine;
    snprintf(status->message, kAlquimiaMaxStringLength, "Invalid ONNX engine state pointer.");
    return;
  }

  onnx_state = *(OnnxEngineState **)onnx_engine_state;

  num_keys = 0;
  keys = NULL;
  ort_status = onnx_state->g_ort->ModelMetadataGetCustomMetadataMapKeys(
      onnx_state->metadata, onnx_state->allocator, &keys, &num_keys);

  if (ort_status != NULL)
  {
    snprintf(status->message, kAlquimiaMaxStringLength, "ONNX ModelMetadataGetCustomMetadataMapKeys failed: %s", onnx_state->g_ort->GetErrorMessage(ort_status));
    onnx_state->g_ort->ReleaseStatus(ort_status);
    status->error = kAlquimiaErrorEngineIntegrity;
    return;
  }

  /* Gracefully handle empty or invalid metadata keys */
  if (keys == NULL || num_keys <= 0)
  {
    if (keys != NULL)
    {
      onnx_state->allocator->Free(onnx_state->allocator, keys);
    }
    return;
  }

  for (i = 0; i < num_keys; i++)
  {
    char *value = NULL;
    ort_status = onnx_state->g_ort->ModelMetadataLookupCustomMetadataMap(
        onnx_state->metadata, onnx_state->allocator, keys[i], &value);

    if (ort_status != NULL)
    {
      snprintf(status->message, kAlquimiaMaxStringLength, "ONNX Lookup failed for key '%s': %s", keys[i], onnx_state->g_ort->GetErrorMessage(ort_status));
      onnx_state->g_ort->ReleaseStatus(ort_status);
      status->error = kAlquimiaErrorEngineIntegrity;

      /* Cleanup on failure:
      ** We only free keys starting from the current index `i` up to `num_keys - 1`.
      ** Keys at indices `0` to `i-1` were already freed by standard deallocation
      ** at the end of the previous successful iterations of this loop. Freeing
      ** them again here would cause a double-free crash!
      */
      onnx_state->allocator->Free(onnx_state->allocator, keys[i]);
      for (j = i + 1; j < num_keys; j++)
      {
        onnx_state->allocator->Free(onnx_state->allocator, keys[j]);
      }
      onnx_state->allocator->Free(onnx_state->allocator, keys);
      return;
    }

    /* Print custom metadata in debug mode only */
    if (debug)
    {
      printf("Custom Metadata - Key: %s, Value: %s\n", keys[i], value);
    }

    /* Store the feature name in meta_data if key matches "feature_X" */
    int feat_idx = -1;
    if (sscanf(keys[i], "feature_%d", &feat_idx) == 1)
    {
      if (meta_data != NULL && feat_idx >= 0 && feat_idx < meta_data->primary_names.size)
      {
        strncpy(meta_data->primary_names.data[feat_idx], value, kAlquimiaMaxStringLength - 1);
        meta_data->primary_names.data[feat_idx][kAlquimiaMaxStringLength - 1] = '\0';
      }
    }

    /* Standard deallocation */
    onnx_state->allocator->Free(onnx_state->allocator, value);
    onnx_state->allocator->Free(onnx_state->allocator, keys[i]);
  }

  if (num_keys > 0 && keys != NULL)
  {
    onnx_state->allocator->Free(onnx_state->allocator, keys);
  }
}

#endif /* ALQUIMIA_HAVE_ONNX */
