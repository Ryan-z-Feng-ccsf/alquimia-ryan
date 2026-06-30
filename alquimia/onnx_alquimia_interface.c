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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ==========================================================================
   🌟 Third-Party Library Syntax Sandbox (Compiler Diagnostic Shield)
   --------------------------------------------------------------------------
   In the global scope, onnxruntime_c_api.h appends an extra trailing ';'
   to the ORT_RUNTIME_CLASS macro. This expands into a duplicate semicolon,
   creating an isolated empty declaration outside of any function body.

   Since Alquimia enables strict compliance via `-Wpedantic` and `-Werror`,
   GCC/Clang treats this minor third-party flaw as a Fatal Error.

   Solution:
   1. `push`: Back up the current strict project-wide warning configurations.
   2. `ignored "-Wpedantic"`: Mute ISO C compliance warnings during inclusion.
   3. `pop`: Instantly restore strict rules to protect our own codebase.
   ========================================================================== */
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

typedef struct
{
  const OrtApi *g_ort;
  OrtEnv *env;
  OrtSessionOptions *session_options;
  OrtSession *session;
  OrtModelMetadata *metadata;
  OrtMemoryInfo *memory_info;
  OrtAllocator *allocator;
  OrtValue *input_tensor;
  OrtValue *output_tensor;
  double input_data[5];  /* Buffer for [1, 5] static input */
  double output_data[1]; /* Buffer for [1, 1] static output */
} OnnxEngineState;

/*
** Helper to check OrtStatus, write the error message to AlquimiaEngineStatus,
** release OrtStatus, and return false on error.
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

/*
** Set up ONNX Chemistry Engine
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
  const char *model_path;
  FILE *f;
  int64_t input_shape[2];
  int64_t output_shape[2];

  status->error = kAlquimiaNoError;
  status->message[0] = '\0';

  // Unused
  (void)hands_off;

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

  /* Create Env */
  ort_status = onnx_state->g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "onnx_alquimia_engine", &onnx_state->env);
  if (!CheckStatus(onnx_state->g_ort, ort_status, status))
  {
    free(onnx_state);
    return;
  }

  /* Create Session Options */
  ort_status = onnx_state->g_ort->CreateSessionOptions(&onnx_state->session_options);
  if (!CheckStatus(onnx_state->g_ort, ort_status, status))
  {
    onnx_state->g_ort->ReleaseEnv(onnx_state->env);
    free(onnx_state);
    return;
  }

  /* Determine and check model file path */
  model_path = "alquimia/mixed/lsurf_model_float_64.onnx";
  if (input_filename != NULL && strlen(input_filename) > 0)
  {
    model_path = input_filename;
  }

  f = fopen(model_path, "r");
  if (f == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Model file not found: %s", model_path);
    onnx_state->g_ort->ReleaseSessionOptions(onnx_state->session_options);
    onnx_state->g_ort->ReleaseEnv(onnx_state->env);
    free(onnx_state);
    return;
  }
  fclose(f);

  /* Create Session */
  ort_status = onnx_state->g_ort->CreateSession(onnx_state->env, model_path, onnx_state->session_options, &onnx_state->session);
  if (!CheckStatus(onnx_state->g_ort, ort_status, status))
  {
    onnx_state->g_ort->ReleaseSessionOptions(onnx_state->session_options);
    onnx_state->g_ort->ReleaseEnv(onnx_state->env);
    free(onnx_state);
    return;
  }

  /* Get Model Metadata */
  ort_status = onnx_state->g_ort->SessionGetModelMetadata(onnx_state->session, &onnx_state->metadata);
  if (!CheckStatus(onnx_state->g_ort, ort_status, status))
  {
    onnx_state->g_ort->ReleaseSession(onnx_state->session);
    onnx_state->g_ort->ReleaseSessionOptions(onnx_state->session_options);
    onnx_state->g_ort->ReleaseEnv(onnx_state->env);
    free(onnx_state);
    return;
  }

  /* Create CPU Memory Info */
  ort_status = onnx_state->g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &onnx_state->memory_info);
  if (!CheckStatus(onnx_state->g_ort, ort_status, status))
  {
    onnx_state->g_ort->ReleaseModelMetadata(onnx_state->metadata);
    onnx_state->g_ort->ReleaseSession(onnx_state->session);
    onnx_state->g_ort->ReleaseSessionOptions(onnx_state->session_options);
    onnx_state->g_ort->ReleaseEnv(onnx_state->env);
    free(onnx_state);
    return;
  }

  /* Create Allocator */
  ort_status = onnx_state->g_ort->CreateAllocator(onnx_state->session, onnx_state->memory_info, &onnx_state->allocator);
  if (!CheckStatus(onnx_state->g_ort, ort_status, status))
  {
    onnx_state->g_ort->ReleaseMemoryInfo(onnx_state->memory_info);
    onnx_state->g_ort->ReleaseModelMetadata(onnx_state->metadata);
    onnx_state->g_ort->ReleaseSession(onnx_state->session);
    onnx_state->g_ort->ReleaseSessionOptions(onnx_state->session_options);
    onnx_state->g_ort->ReleaseEnv(onnx_state->env);
    free(onnx_state);
    return;
  }

  /* Create static re-usable Input Tensor */
  input_shape[0] = 1;
  input_shape[1] = 5;
  ort_status = onnx_state->g_ort->CreateTensorWithDataAsOrtValue(
      onnx_state->memory_info, onnx_state->input_data, sizeof(onnx_state->input_data),
      input_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, &onnx_state->input_tensor);
  if (!CheckStatus(onnx_state->g_ort, ort_status, status))
  {
    onnx_state->g_ort->ReleaseAllocator(onnx_state->allocator);
    onnx_state->g_ort->ReleaseMemoryInfo(onnx_state->memory_info);
    onnx_state->g_ort->ReleaseModelMetadata(onnx_state->metadata);
    onnx_state->g_ort->ReleaseSession(onnx_state->session);
    onnx_state->g_ort->ReleaseSessionOptions(onnx_state->session_options);
    onnx_state->g_ort->ReleaseEnv(onnx_state->env);
    free(onnx_state);
    return;
  }

  /* Create static re-usable Output Tensor */
  output_shape[0] = 1;
  output_shape[1] = 1;
  ort_status = onnx_state->g_ort->CreateTensorWithDataAsOrtValue(
      onnx_state->memory_info, onnx_state->output_data, sizeof(onnx_state->output_data),
      output_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, &onnx_state->output_tensor);
  if (!CheckStatus(onnx_state->g_ort, ort_status, status))
  {
    onnx_state->g_ort->ReleaseValue(onnx_state->input_tensor);
    onnx_state->g_ort->ReleaseAllocator(onnx_state->allocator);
    onnx_state->g_ort->ReleaseMemoryInfo(onnx_state->memory_info);
    onnx_state->g_ort->ReleaseModelMetadata(onnx_state->metadata);
    onnx_state->g_ort->ReleaseSession(onnx_state->session);
    onnx_state->g_ort->ReleaseSessionOptions(onnx_state->session_options);
    onnx_state->g_ort->ReleaseEnv(onnx_state->env);
    free(onnx_state);
    return;
  }

  /* Initialize sizes and functionality */
  sizes->num_primary = 5;
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

  functionality->thread_safe = true;
  functionality->temperature_dependent = false;
  functionality->pressure_dependent = false;
  functionality->porosity_update = false;
  functionality->operator_splitting = true;
  functionality->global_implicit = false;
  functionality->index_base = 0;

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

  status->error = kAlquimiaNoError;
  status->message[0] = '\0';

  if (onnx_engine_state == NULL)
  {
    status->error = kAlquimiaErrorInvalidEngine;
    snprintf(status->message, kAlquimiaMaxStringLength, "Invalid ONNX engine state pointer in shutdown.");
    return;
  }

  onnx_state = (OnnxEngineState *)onnx_engine_state;
  if (onnx_state->g_ort != NULL)
  {
    if (onnx_state->output_tensor != NULL)
    {
      onnx_state->g_ort->ReleaseValue(onnx_state->output_tensor);
      onnx_state->output_tensor = NULL;
    }
    if (onnx_state->input_tensor != NULL)
    {
      onnx_state->g_ort->ReleaseValue(onnx_state->input_tensor);
      onnx_state->input_tensor = NULL;
    }
    if (onnx_state->allocator != NULL)
    {
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
  double input_vals[] = {-6.67778070526608, -4.54327863489071, -25.419, -20.100, -37.489};
  int i;

  status->error = kAlquimiaNoError;
  status->message[0] = '\0';

  // Unused
  (void)onnx_engine_state;
  (void)condition;
  (void)props;
  (void)aux_data;

  if (state == NULL || state->total_mobile.data == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Invalid state or total_mobile.data in ProcessCondition.");
    return;
  }

  if (state->total_mobile.size < 5)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "state->total_mobile.size is less than 5.");
    return;
  }

  for (i = 0; i < 5; ++i)
  {
    state->total_mobile.data[i] = input_vals[i];
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
  const char *input_names[1];
  const char *output_names[1];
  OrtStatus *ort_status;
  double *out_arr;
  int i;

  status->error = kAlquimiaNoError;
  status->message[0] = '\0';

  // Unused
  (void)delta_t;
  (void)props;
  (void)aux_data;
  (void)natural_id;

  if (onnx_engine_state == NULL)
  {
    status->error = kAlquimiaErrorInvalidEngine;
    snprintf(status->message, kAlquimiaMaxStringLength, "Invalid ONNX engine state pointer in reactionstepoperatorsplit.");
    return;
  }

  if (state == NULL || state->total_mobile.data == NULL || state->total_mobile.size < 5)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Invalid state data in reactionstepoperatorsplit.");
    return;
  }

  onnx_state = (OnnxEngineState *)onnx_engine_state;

  /* Copy data from state->total_mobile.data into pre-allocated input_data buffer */
  for (i = 0; i < 5; ++i)
  {
    onnx_state->input_data[i] = state->total_mobile.data[i];
  }

  input_names[0] = "double_input";
  output_names[0] = "double_output";

  /* Run inference using pre-allocated input and output tensors */
  ort_status = onnx_state->g_ort->Run(
      onnx_state->session,
      NULL, /* RunOptions */
      input_names,
      (const OrtValue *const *)&onnx_state->input_tensor,
      1,
      output_names,
      1,
      &onnx_state->output_tensor);

  if (!CheckStatus(onnx_state->g_ort, ort_status, status))
  {
    return;
  }

  if (onnx_state->output_tensor == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "ONNX Run completed but output_tensor is NULL.");
    return;
  }

  out_arr = NULL;
  ort_status = onnx_state->g_ort->GetTensorMutableData(onnx_state->output_tensor, (void **)&out_arr);
  if (!CheckStatus(onnx_state->g_ort, ort_status, status))
  {
    return;
  }

  if (out_arr == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "ONNX GetTensorMutableData returned NULL data pointer.");
    return;
  }

  /* Write the prediction back to state->total_mobile.data[0] */
  state->total_mobile.data[0] = out_arr[0];
}

/*
** Get Auxiliary Output
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

  status->error = kAlquimiaNoError;
  status->message[0] = '\0';

  if (onnx_engine_state == NULL)
  {
    status->error = kAlquimiaErrorInvalidEngine;
    snprintf(status->message, kAlquimiaMaxStringLength, "Invalid ONNX engine state pointer.");
    return;
  }

  onnx_state = (OnnxEngineState *)onnx_engine_state;

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

    /* Print custom metadata */
    printf("Custom Metadata - Key: %s, Value: %s\n", keys[i], value);

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
