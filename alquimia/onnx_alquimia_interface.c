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

  /* Sentinel value sharing */
  int num_primary;

  /* Dynamic input info */
  size_t num_inputs;
  char **input_names;
  size_t *input_num_dim;
  int64_t **input_dim_values;
  size_t *input_total_size;
  double **input_data;
  OrtValue **input_tensor;
  size_t *input_offsets;

  /* Dynamic output info */
  size_t num_outputs;
  char **output_names;
  size_t *output_num_dim;
  int64_t **output_dim_values;
  size_t *output_total_size;
  double **output_data;
  OrtValue **output_tensor;
  size_t *output_offsets;
} OnnxEngineState;

/* Helper to check OrtStatus */
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

static void CleanupOnSetupFailure(OnnxEngineState *onnx_state)
{
  /* Safely tear down any resources that were already allocated during setup.
  ** We pass a temporary status struct to onnx_alquimia_shutdown to prevent
  ** overwriting the original setup failure error stored in 'status'. */
  AlquimiaEngineStatus temp_status;
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
    CleanupOnSetupFailure(onnx_state);
    return;
  }

  ort_status = onnx_state->g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &onnx_state->memory_info);
  if (!CheckStatus(onnx_state->g_ort, ort_status, status))
  {
    CleanupOnSetupFailure(onnx_state);
    return;
  }

  ort_status = onnx_state->g_ort->CreateAllocator(onnx_state->session, onnx_state->memory_info, &onnx_state->allocator);
  if (!CheckStatus(onnx_state->g_ort, ort_status, status))
  {
    CleanupOnSetupFailure(onnx_state);
    return;
  }

  /* Query input/output count */
  size_t num_inputs = 0;
  ort_status = onnx_state->g_ort->SessionGetInputCount(onnx_state->session, &num_inputs);
  if (!CheckStatus(onnx_state->g_ort, ort_status, status))
  {
    CleanupOnSetupFailure(onnx_state);
    return;
  }
  onnx_state->num_inputs = num_inputs;

  size_t num_outputs = 0;
  ort_status = onnx_state->g_ort->SessionGetOutputCount(onnx_state->session, &num_outputs);
  if (!CheckStatus(onnx_state->g_ort, ort_status, status))
  {
    CleanupOnSetupFailure(onnx_state);
    return;
  }
  onnx_state->num_outputs = num_outputs;

  /* Guard against empty inputs or outputs */
  if (num_inputs == 0 || num_outputs == 0)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Model has 0 inputs or outputs (inputs: %d, outputs: %d).", (int)num_inputs, (int)num_outputs);
    CleanupOnSetupFailure(onnx_state);
    return;
  }

  /* Allocate outer arrays for inputs */
  onnx_state->input_names = (char **)calloc(num_inputs, sizeof(char *));
  onnx_state->input_num_dim = (size_t *)calloc(num_inputs, sizeof(size_t));
  onnx_state->input_dim_values = (int64_t **)calloc(num_inputs, sizeof(int64_t *));
  onnx_state->input_total_size = (size_t *)calloc(num_inputs, sizeof(size_t));
  onnx_state->input_data = (double **)calloc(num_inputs, sizeof(double *));
  onnx_state->input_tensor = (OrtValue **)calloc(num_inputs, sizeof(OrtValue *));
  onnx_state->input_offsets = (size_t *)calloc(num_inputs + 1, sizeof(size_t));

  if (onnx_state->input_names == NULL || onnx_state->input_num_dim == NULL ||
      onnx_state->input_dim_values == NULL || onnx_state->input_total_size == NULL ||
      onnx_state->input_data == NULL || onnx_state->input_tensor == NULL ||
      onnx_state->input_offsets == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Memory allocation failed for input container arrays.");
    CleanupOnSetupFailure(onnx_state);
    return;
  }

  /* Parse input names and dimensions */
  for (i = 0; i < num_inputs; ++i)
  {
    char *name = NULL;
    ort_status = onnx_state->g_ort->SessionGetInputName(onnx_state->session, i, onnx_state->allocator, &name);
    if (!CheckStatus(onnx_state->g_ort, ort_status, status))
    {
      CleanupOnSetupFailure(onnx_state);
      return;
    }
    onnx_state->input_names[i] = name;

    OrtTypeInfo *type_info = NULL;
    ort_status = onnx_state->g_ort->SessionGetInputTypeInfo(onnx_state->session, i, &type_info);
    if (!CheckStatus(onnx_state->g_ort, ort_status, status))
    {
      CleanupOnSetupFailure(onnx_state);
      return;
    }

    const OrtTensorTypeAndShapeInfo *tensor_info = NULL;
    ort_status = onnx_state->g_ort->CastTypeInfoToTensorInfo(type_info, &tensor_info);
    if (!CheckStatus(onnx_state->g_ort, ort_status, status))
    {
      onnx_state->g_ort->ReleaseTypeInfo(type_info);
      CleanupOnSetupFailure(onnx_state);
      return;
    }

    size_t dim_count = 0;
    ort_status = onnx_state->g_ort->GetDimensionsCount(tensor_info, &dim_count);
    if (!CheckStatus(onnx_state->g_ort, ort_status, status))
    {
      onnx_state->g_ort->ReleaseTypeInfo(type_info);
      CleanupOnSetupFailure(onnx_state);
      return;
    }

    /* Zero-Dim / Scalar Guard */
    if (dim_count == 0)
    {
      onnx_state->input_num_dim[i] = 1;
      onnx_state->input_dim_values[i] = (int64_t *)calloc(1, sizeof(int64_t));
      if (onnx_state->input_dim_values[i] == NULL)
      {
        onnx_state->g_ort->ReleaseTypeInfo(type_info);
        status->error = kAlquimiaErrorEngineIntegrity;
        snprintf(status->message, kAlquimiaMaxStringLength, "Memory allocation failed for scalar input_dim_values.");
        CleanupOnSetupFailure(onnx_state);
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
        CleanupOnSetupFailure(onnx_state);
        return;
      }
      ort_status = onnx_state->g_ort->GetDimensions(tensor_info, onnx_state->input_dim_values[i], dim_count);
      if (!CheckStatus(onnx_state->g_ort, ort_status, status))
      {
        onnx_state->g_ort->ReleaseTypeInfo(type_info);
        CleanupOnSetupFailure(onnx_state);
        return;
      }
    }

    /* In-place Dimension Correction */
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
      CleanupOnSetupFailure(onnx_state);
      return;
    }

    /* Create dynamic reusable Input Tensor */
    ort_status = onnx_state->g_ort->CreateTensorWithDataAsOrtValue(
        onnx_state->memory_info, onnx_state->input_data[i], total_size * sizeof(double),
        onnx_state->input_dim_values[i], onnx_state->input_num_dim[i],
        ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, &onnx_state->input_tensor[i]);
    if (!CheckStatus(onnx_state->g_ort, ort_status, status))
    {
      onnx_state->g_ort->ReleaseTypeInfo(type_info);
      CleanupOnSetupFailure(onnx_state);
      return;
    }

    onnx_state->g_ort->ReleaseTypeInfo(type_info);
  }

  /* Compute Input Cumulative Offsets & num_primary */
  onnx_state->input_offsets[0] = 0;
  for (i = 0; i < num_inputs; ++i)
  {
    int64_t last_dim = onnx_state->input_dim_values[i][onnx_state->input_num_dim[i] - 1];
    onnx_state->input_offsets[i + 1] = onnx_state->input_offsets[i] + (size_t)last_dim;
  }
  int num_primary = (int)onnx_state->input_offsets[num_inputs];
  onnx_state->num_primary = num_primary;
  sizes->num_primary = num_primary;

  /* Allocate outer arrays for outputs */
  onnx_state->output_names = (char **)calloc(num_outputs, sizeof(char *));
  onnx_state->output_num_dim = (size_t *)calloc(num_outputs, sizeof(size_t));
  onnx_state->output_dim_values = (int64_t **)calloc(num_outputs, sizeof(int64_t *));
  onnx_state->output_total_size = (size_t *)calloc(num_outputs, sizeof(size_t));
  onnx_state->output_data = (double **)calloc(num_outputs, sizeof(double *));
  onnx_state->output_tensor = (OrtValue **)calloc(num_outputs, sizeof(OrtValue *));
  onnx_state->output_offsets = (size_t *)calloc(num_outputs + 1, sizeof(size_t));

  if (onnx_state->output_names == NULL || onnx_state->output_num_dim == NULL ||
      onnx_state->output_dim_values == NULL || onnx_state->output_total_size == NULL ||
      onnx_state->output_data == NULL || onnx_state->output_tensor == NULL ||
      onnx_state->output_offsets == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Memory allocation failed for output container arrays.");
    CleanupOnSetupFailure(onnx_state);
    return;
  }

  /* Parse output names and dimensions */
  for (i = 0; i < num_outputs; ++i)
  {
    char *name = NULL;
    ort_status = onnx_state->g_ort->SessionGetOutputName(onnx_state->session, i, onnx_state->allocator, &name);
    if (!CheckStatus(onnx_state->g_ort, ort_status, status))
    {
      CleanupOnSetupFailure(onnx_state);
      return;
    }
    onnx_state->output_names[i] = name;

    OrtTypeInfo *type_info = NULL;
    ort_status = onnx_state->g_ort->SessionGetOutputTypeInfo(onnx_state->session, i, &type_info);
    if (!CheckStatus(onnx_state->g_ort, ort_status, status))
    {
      CleanupOnSetupFailure(onnx_state);
      return;
    }

    const OrtTensorTypeAndShapeInfo *tensor_info = NULL;
    ort_status = onnx_state->g_ort->CastTypeInfoToTensorInfo(type_info, &tensor_info);
    if (!CheckStatus(onnx_state->g_ort, ort_status, status))
    {
      onnx_state->g_ort->ReleaseTypeInfo(type_info);
      CleanupOnSetupFailure(onnx_state);
      return;
    }

    size_t dim_count = 0;
    ort_status = onnx_state->g_ort->GetDimensionsCount(tensor_info, &dim_count);
    if (!CheckStatus(onnx_state->g_ort, ort_status, status))
    {
      onnx_state->g_ort->ReleaseTypeInfo(type_info);
      CleanupOnSetupFailure(onnx_state);
      return;
    }

    /* Zero-Dim / Scalar Guard */
    if (dim_count == 0)
    {
      onnx_state->output_num_dim[i] = 1;
      onnx_state->output_dim_values[i] = (int64_t *)calloc(1, sizeof(int64_t));
      if (onnx_state->output_dim_values[i] == NULL)
      {
        onnx_state->g_ort->ReleaseTypeInfo(type_info);
        status->error = kAlquimiaErrorEngineIntegrity;
        snprintf(status->message, kAlquimiaMaxStringLength, "Memory allocation failed for scalar output_dim_values.");
        CleanupOnSetupFailure(onnx_state);
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
        CleanupOnSetupFailure(onnx_state);
        return;
      }
      ort_status = onnx_state->g_ort->GetDimensions(tensor_info, onnx_state->output_dim_values[i], dim_count);
      if (!CheckStatus(onnx_state->g_ort, ort_status, status))
      {
        onnx_state->g_ort->ReleaseTypeInfo(type_info);
        CleanupOnSetupFailure(onnx_state);
        return;
      }
    }

    /* In-place Dimension Correction */
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
      CleanupOnSetupFailure(onnx_state);
      return;
    }

    /* Create dynamic reusable Output Tensor */
    ort_status = onnx_state->g_ort->CreateTensorWithDataAsOrtValue(
        onnx_state->memory_info, onnx_state->output_data[i], total_size * sizeof(double),
        onnx_state->output_dim_values[i], onnx_state->output_num_dim[i],
        ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, &onnx_state->output_tensor[i]);
    if (!CheckStatus(onnx_state->g_ort, ort_status, status))
    {
      onnx_state->g_ort->ReleaseTypeInfo(type_info);
      CleanupOnSetupFailure(onnx_state);
      return;
    }

    onnx_state->g_ort->ReleaseTypeInfo(type_info);
  }

  /* Compute Output Cumulative Offsets */
  onnx_state->output_offsets[0] = 0;
  for (i = 0; i < num_outputs; ++i)
  {
    int64_t last_dim = onnx_state->output_dim_values[i][onnx_state->output_num_dim[i] - 1];
    onnx_state->output_offsets[i + 1] = onnx_state->output_offsets[i] + (size_t)last_dim;
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
  size_t i;

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
    if (onnx_state->input_offsets != NULL)
    {
      free(onnx_state->input_offsets);
      onnx_state->input_offsets = NULL;
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
    if (onnx_state->output_offsets != NULL)
    {
      free(onnx_state->output_offsets);
      onnx_state->output_offsets = NULL;
    }

    /* Release allocator and names */
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
  status->error = kAlquimiaNoError;
  status->message[0] = '\0';

  /* Keeping ProcessCondition empty/stub for now */
  (void)onnx_engine_state;
  (void)condition;
  (void)props;
  (void)state;
  (void)aux_data;
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

  if (onnx_engine_state == NULL)
  {
    status->error = kAlquimiaErrorInvalidEngine;
    snprintf(status->message, kAlquimiaMaxStringLength, "Invalid ONNX engine state pointer in reactionstepoperatorsplit.");
    return;
  }

  onnx_state = (OnnxEngineState *)onnx_engine_state;

  if (state == NULL || state->total_mobile.data == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Invalid state or total_mobile.data in reactionstepoperatorsplit.");
    return;
  }

  /* Array Bounds Verification */
  if (state->total_mobile.size < onnx_state->input_offsets[onnx_state->num_inputs] ||
      state->total_mobile.size < onnx_state->output_offsets[onnx_state->num_outputs])
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Array bounds check failed in reactionstepoperatorsplit. "
             "state->total_mobile.size (%d) is smaller than required input offset (%d) "
             "or output offset (%d).",
             (int)state->total_mobile.size,
             (int)onnx_state->input_offsets[onnx_state->num_inputs],
             (int)onnx_state->output_offsets[onnx_state->num_outputs]);
    return;
  }

  /* Copy data from state->total_mobile.data into pre-allocated input_data buffers using offsets */
  for (i = 0; i < (int)onnx_state->num_inputs; ++i)
  {
    int64_t last_dim = onnx_state->input_dim_values[i][onnx_state->input_num_dim[i] - 1];
    int64_t k;
    for (k = 0; k < last_dim; ++k)
    {
      onnx_state->input_data[i][k] = state->total_mobile.data[onnx_state->input_offsets[i] + k];
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

  /* Copy output data back to state->total_mobile.data using offsets */
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

    int64_t last_dim = onnx_state->output_dim_values[i][onnx_state->output_num_dim[i] - 1];
    int k;
    for (k = 0; k < last_dim; ++k)
    {
      state->total_mobile.data[onnx_state->output_offsets[i] + k] = out_arr[k];
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
