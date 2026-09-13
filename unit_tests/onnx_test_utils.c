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
** Shared ONNX unit-test utility implementation.
**
** Authors:
**        Zhuolei Feng, Sergi Molins
**
** Notes:
**
**  * This file implements helper logic shared by the ONNX unit tests.
**  * It is not a test suite by itself. It supports the M/R/F/C/E/L case
**    series declared in the individual test files.
**  * Helpers here centralize Alquimia interface setup, engine shutdown,
**    state allocation, condition initialization, inference execution,
**    temporary config handling, and test diagnostics.
**
** ****************************************************************************
*/

#include "onnx_test_utils.h"

#if ALQUIMIA_HAVE_ONNX

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "alquimia/onnx_alquimia_config.h"

/**
 * @brief Records an ONNX test failure with optional engine-status diagnostics.
 */
void OnnxRecordTestFailure(
    int *failure_count,
    const char *test_id,
    const char *issue,
    const char *condition,
    const AlquimiaEngineStatus *status,
    const char *file,
    int line)
{
  fprintf(stderr, "%s: %s\n"
                  "  Failed check: %s\n",
          test_id, issue, condition);
  if (status != NULL)
  {
    fprintf(stderr, "  Actual status: error=%d, message=\"%s\"\n",
            status->error, status->message != NULL ? status->message : "");
  }
  fprintf(stderr, "  Location: %s:%d\n", file, line);
  ++(*failure_count);
}

/**
 * @brief Verifies that an invalid ONNX setup case fails with expected text.
 */
void OnnxCheckSetupFailureAt(
    int *failure_count,
    const OnnxSetupFailureCase *test_case,
    const char *file,
    int line)
{
  AlquimiaEngineFunctionality functionality = {0};
  AlquimiaEngineStatus status = {0};
  AlquimiaInterface interface = {0};
  AlquimiaSizes sizes = {0};
  char path[ONNX_TEST_PATH_SIZE];
  void *onnx_engine_state = NULL;

  if (!OnnxCreateInterface(&interface, &status))
  {
    OnnxRecordTestFailure(
        failure_count, test_case->test_id,
        "Unable to create the ONNX interface",
        "status.error == kAlquimiaNoError", &status, file, line);
    FreeAlquimiaEngineStatus(&status);
    return;
  }

  if (!OnnxTestCasePath(test_case->relative_path, path, sizeof(path)))
  {
    OnnxRecordTestFailure(
        failure_count, test_case->test_id,
        "Unable to construct the ONNX test-case path", "OnnxTestCasePath",
        NULL, file, line);
    FreeAlquimiaEngineStatus(&status);
    return;
  }

  interface.Setup(path, false, &onnx_engine_state, &sizes, &functionality,
                  &status);
  if (status.error == kAlquimiaNoError)
  {
    OnnxRecordTestFailure(
        failure_count, test_case->test_id, test_case->message_fragment,
        "status.error != kAlquimiaNoError", &status, file, line);
  }
  if (onnx_engine_state != NULL)
  {
    OnnxRecordTestFailure(
        failure_count, test_case->test_id,
        "Setup failure published a non-NULL engine state",
        "onnx_engine_state == NULL", NULL, file, line);
  }
  if (test_case->message_fragment != NULL)
  {
    if (status.message == NULL ||
        strstr(status.message, test_case->message_fragment) == NULL)
    {
      OnnxRecordTestFailure(
          failure_count, test_case->test_id, test_case->message_fragment,
          "status.message contains message_fragment", &status, file, line);
    }
  }

  interface.Shutdown(&onnx_engine_state, &status);
  if (status.error != kAlquimiaErrorInvalidEngine)
  {
    OnnxRecordTestFailure(
        failure_count, "E12",
        "Shutdown after failed setup did not reject the NULL engine",
        "status.error == kAlquimiaErrorInvalidEngine", &status, file, line);
  }
  if (onnx_engine_state != NULL)
  {
    OnnxRecordTestFailure(
        failure_count, "E12",
        "Shutdown after failed setup changed the engine pointer",
        "onnx_engine_state == NULL", NULL, file, line);
  }
  FreeAlquimiaEngineStatus(&status);
}

/**
 * @brief Checks an EX8 prediction against its expected reference value.
 */
bool OnnxCheckEx8Prediction(
    const char *test_id,
    const char *feature,
    double actual,
    double expected)
{
  double tolerance = fabs(expected) * 1.0e-10 + 1.0e-18;

  if (!isfinite(actual) || fabs(actual - expected) > tolerance)
  {
    fprintf(stderr,
            "%s: %s prediction was %.17g; expected %.17g "
            "within %.17g.\n",
            test_id, feature, actual, expected, tolerance);
    return false;
  }
  return true;
}

/**
 * @brief Builds an ONNX test path under a source-relative folder.
 */
static bool OnnxPathInFolder(
    const char *folder,
    const char *relative_path,
    char *path,
    size_t size)
{
  int result = snprintf(path, size, "%s/%s/%s",
                        CMAKE_CURRENT_SOURCE_DIR, folder, relative_path);
  return result >= 0 && (size_t)result < size;
}

/**
 * @brief Builds a path under unit_tests/onnx_test_cases.
 */
bool OnnxTestCasePath(const char *relative_path, char *path, size_t size)
{
  return OnnxPathInFolder("onnx_test_cases", relative_path, path, size);
}

/**
 * @brief Builds a path under the shared models directory.
 */
bool OnnxModelPath(const char *relative_path, char *path, size_t size)
{
  return OnnxPathInFolder("../models", relative_path, path, size);
}

/**
 * @brief Creates the ONNX Alquimia interface and initializes its status.
 */
bool OnnxCreateInterface(
    AlquimiaInterface *interface,
    AlquimiaEngineStatus *status)
{
  AllocateAlquimiaEngineStatus(status);
  CreateAlquimiaInterface("ONNX", interface, status);
  return status->error == kAlquimiaNoError;
}

/**
 * @brief Sets up an ONNX test engine from an absolute config path.
 */
bool OnnxSetupEngineAtPath(
    const char *path,
    bool hands_off,
    OnnxTestEngine *engine)
{
  memset(engine, 0, sizeof(*engine));
  if (!OnnxCreateInterface(&engine->interface, &engine->status))
  {
    return false;
  }

  engine->interface.Setup(path, hands_off, &engine->engine_state,
                          &engine->sizes, &engine->functionality,
                          &engine->status);
  return engine->status.error == kAlquimiaNoError;
}

/**
 * @brief Sets up an ONNX engine from a unit-test config path.
 */
bool OnnxSetupEngine(
    const char *relative_path,
    bool hands_off,
    OnnxTestEngine *engine)
{
  char config_path[ONNX_TEST_PATH_SIZE];

  if (!OnnxTestCasePath(relative_path, config_path, sizeof(config_path)))
  {
    return false;
  }
  return OnnxSetupEngineAtPath(config_path, hands_off, engine);
}

/**
 * @brief Sets up an ONNX engine from a shared-model config path.
 */
bool OnnxSetupSharedModelEngine(
    const char *relative_path,
    bool hands_off,
    OnnxTestEngine *engine)
{
  char config_path[ONNX_TEST_PATH_SIZE];

  if (!OnnxModelPath(relative_path, config_path, sizeof(config_path)))
  {
    return false;
  }
  return OnnxSetupEngineAtPath(config_path, hands_off, engine);
}

/**
 * @brief Shuts down an ONNX test engine and always releases status storage.
 */
bool OnnxShutdownEngine(OnnxTestEngine *engine)
{
  bool success = engine->engine_state == NULL;

  /* Failed setup may leave only status storage to release. */
  if (engine->engine_state != NULL && engine->interface.Shutdown != NULL)
  {
    engine->interface.Shutdown(&engine->engine_state, &engine->status);
    success = engine->status.error == kAlquimiaNoError &&
        engine->engine_state == NULL;
  }
  if (!success)
  {
    fprintf(stderr, "Shutdown failed: %s\n",
            engine->status.message != NULL ? engine->status.message : "");
  }
  FreeAlquimiaEngineStatus(&engine->status);
  return success;
}

/**
 * @brief Returns setup from a unit-test config path and reports success.
 */
bool OnnxRequireSetupEngine(
    const char *relative_path,
    bool hands_off,
    OnnxTestEngine *engine)
{
  if (!OnnxSetupEngine(relative_path, hands_off, engine))
  {
    fprintf(stderr, "Unable to set up ONNX engine '%s': %s\n", relative_path,
            engine->status.message != NULL ? engine->status.message : "");
    return false;
  }
  return true;
}

/**
 * @brief Returns setup from a shared-model config path and reports success.
 */
bool OnnxRequireSharedModelEngine(
    const char *relative_path,
    bool hands_off,
    OnnxTestEngine *engine)
{
  if (!OnnxSetupSharedModelEngine(relative_path, hands_off, engine))
  {
    fprintf(stderr, "Unable to set up ONNX model '%s': %s\n", relative_path,
            engine->status.message != NULL ? engine->status.message : "");
    return false;
  }
  return true;
}

/**
 * @brief Allocates an Alquimia state sized for the ONNX test engine.
 */
void OnnxAllocateState(const OnnxTestEngine *engine, AlquimiaState *state)
{
  memset(state, 0, sizeof(*state));
  AllocateAlquimiaState(&engine->sizes, state);
  state->porosity = 1.0;
}

/**
 * @brief Initializes one aqueous constraint by feature name and value.
 */
void OnnxInitializeConstraint(
    AlquimiaGeochemicalCondition *condition,
    int index,
    const char *name,
    double value)
{
  AlquimiaAqueousConstraint *constraint =
      &condition->aqueous_constraints.data[index];

  AllocateAlquimiaAqueousConstraint(constraint);
  strcpy(constraint->primary_species_name, name);
  strcpy(constraint->constraint_type, "total");
  constraint->value = value;
}

/**
 * @brief Runs one operator-split ONNX inference step and reports success.
 */
bool OnnxRunInference(OnnxTestEngine *engine, AlquimiaState *state)
{
  AlquimiaAuxiliaryData aux_data = {0};
  AlquimiaProperties properties = {0};
  properties.saturation = 1.0;

  engine->interface.ReactionStepOperatorSplit(
      &engine->engine_state, 1.0, &properties, state, &aux_data, 0,
      &engine->status);
  if (engine->status.error != kAlquimiaNoError)
  {
    fprintf(stderr, "ONNX inference failed: %s\n",
            engine->status.message != NULL ? engine->status.message : "");
    return false;
  }
  return true;
}

/**
 * @brief Applies one named JSON condition to an ONNX state and reports success.
 */
bool OnnxApplyNamedCondition(
    OnnxTestEngine *engine,
    const char *condition_name,
    AlquimiaState *state)
{
  AlquimiaAuxiliaryData auxiliary_data = {0};
  AlquimiaGeochemicalCondition condition = {0};
  AlquimiaProperties properties = {0};
  properties.saturation = 1.0;

  AllocateAlquimiaGeochemicalCondition(
      (int)strlen(condition_name), 0, 0, &condition);
  strcpy(condition.name, condition_name);
  engine->interface.ProcessCondition(
      &engine->engine_state, &condition, &properties, state, &auxiliary_data,
      &engine->status);
  if (engine->status.error != kAlquimiaNoError)
  {
    fprintf(stderr, "Unable to apply ONNX condition '%s': %s\n",
            condition_name,
            engine->status.message != NULL ? engine->status.message : "");
    FreeAlquimiaGeochemicalCondition(&condition);
    return false;
  }
  FreeAlquimiaGeochemicalCondition(&condition);
  return true;
}

/**
 * @brief Returns true when two floating-point values are within tolerance.
 */
bool OnnxCloseEnough(double actual, double expected, double tolerance)
{
  if (!isfinite(actual) || fabs(actual - expected) > tolerance)
  {
    fprintf(stderr,
            "Actual prediction was %.17g; expected %.17g "
            "within %.17g.\n",
            actual, expected, tolerance);
    return false;
  }
  return true;
}

/**
 * @brief Writes a temporary ONNX JSON config used by parser/setup tests.
 */
bool OnnxWriteTemporaryConfig(const char *contents)
{
  FILE *file = fopen(ONNX_TEST_TEMP_CONFIG, "wb");
  size_t length = strlen(contents);

  if (file == NULL)
  {
    fprintf(stderr, "Unable to open temporary config '%s'.\n",
            ONNX_TEST_TEMP_CONFIG);
    return false;
  }
  if (fwrite(contents, 1, length, file) != length)
  {
    fprintf(stderr, "Unable to write temporary config '%s'.\n",
            ONNX_TEST_TEMP_CONFIG);
    fclose(file);
    OnnxRemoveTemporaryConfig(__func__);
    return false;
  }
  if (fclose(file) != 0)
  {
    fprintf(stderr, "Unable to close temporary config '%s'.\n",
            ONNX_TEST_TEMP_CONFIG);
    OnnxRemoveTemporaryConfig(__func__);
    return false;
  }
  return true;
}

/**
 * @brief Removes the temporary ONNX JSON config used by parser/setup tests.
 */
bool OnnxRemoveTemporaryConfig(const char *test_id)
{
  if (remove(ONNX_TEST_TEMP_CONFIG) != 0)
  {
    fprintf(stderr, "%s could not remove temporary config '%s'.\n",
            test_id, ONNX_TEST_TEMP_CONFIG);
    return false;
  }
  return true;
}

/**
 * @brief Requires config parsing to fail with the expected diagnostic text.
 */
bool OnnxExpectConfigParseFailure(
    const char *test_id,
    const char *json,
    const char *expected_message)
{
  OnnxAlquimiaConfig config = {0};
  char error_message[ONNX_TEST_ERROR_MESSAGE_SIZE] = {0};
  bool success = false;

  if (!OnnxWriteTemporaryConfig(json))
  {
    return false;
  }
  if (OnnxAlquimiaLoadConfig(
      ONNX_TEST_TEMP_CONFIG, &config, error_message, sizeof(error_message)))
  {
    fprintf(stderr, "%s unexpectedly parsed.\n", test_id);
    goto cleanup;
  }
  if (strstr(error_message, expected_message) == NULL)
  {
    fprintf(stderr, "%s error was '%s'; expected '%s'.\n", test_id,
            error_message, expected_message);
    goto cleanup;
  }
  success = true;

cleanup:
  OnnxAlquimiaFreeConfig(&config);
  if (!OnnxRemoveTemporaryConfig(test_id))
  {
    success = false;
  }
  return success;
}

/**
 * @brief Checks setup failure diagnostics and releases temporary resources.
 */
bool OnnxExpectConfigSetupFailure(
    AlquimiaInterface *interface,
    AlquimiaEngineStatus *status,
    const char *test_id,
    const char *json,
    const char *expected_message)
{
  AlquimiaEngineFunctionality functionality = {0};
  AlquimiaSizes sizes = {0};
  void *onnx_engine_state = NULL;
  bool success = false;

  if (!OnnxWriteTemporaryConfig(json))
  {
    return false;
  }
  interface->Setup(ONNX_TEST_TEMP_CONFIG, false, &onnx_engine_state, &sizes,
                   &functionality, status);
  if (status->error == kAlquimiaNoError)
  {
    fprintf(stderr, "%s unexpectedly set up.\n", test_id);
    goto cleanup;
  }
  if (status->message == NULL ||
      strstr(status->message, expected_message) == NULL)
  {
    fprintf(stderr, "%s error was '%s'; expected '%s'.\n", test_id,
            status->message != NULL ? status->message : "", expected_message);
    goto cleanup;
  }
  if (onnx_engine_state != NULL)
  {
    fprintf(stderr, "%s published a non-NULL engine state.\n", test_id);
    goto cleanup;
  }
  success = true;

cleanup:
  if (onnx_engine_state != NULL)
  {
    interface->Shutdown(&onnx_engine_state, status);
    if (status->error != kAlquimiaNoError || onnx_engine_state != NULL)
    {
      success = false;
    }
  }
  if (!OnnxRemoveTemporaryConfig(test_id))
  {
    success = false;
  }
  return success;
}

#endif
