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
** ONNX failure and lifecycle unit tests.
**
** Authors:
**        Zhuolei Feng, Sergi Molins
**
** Notes:
**
**  * This file verifies ONNX adapter robustness: standard condition
**    behavior, deliberate error handling, and lifecycle edge cases.
**  * Test IDs use one prefix plus a two-digit stable case number.
**      C01, C02, ...: standard ProcessCondition behavior.
**      E01, E02, ...: deliberate setup/runtime/guard failures.
**      L01: lifecycle and cleanup special cases.
**  * Example: E01 is the first error-handling case in this file.
**  * The tests intentionally check exact error classes and message
**    fragments, so production diagnostic text changes can affect them.
**  * Tightly coupled: validates by matching substrings in the interface's
**    error message.
**
** ****************************************************************************
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "alquimia/alquimia_constants.h"
#include "alquimia/alquimia_interface.h"
#include "alquimia/alquimia_memory.h"
#include "onnx_test_utils.h"

#if ALQUIMIA_HAVE_ONNX

static int num_failures = 0;

/* ---------- Standard Cases ---------- */

/**
 * @brief Verifies no-op ProcessCondition cases.
 *
 * | C01 | NULL condition | No-op success |
 * | C02 | Condition has no aqueous constraints | No-op success |
 */
static void TestC01ConditionNoOpCases(void)
{
  AlquimiaAuxiliaryData aux_data = {0};
  AlquimiaGeochemicalCondition empty_condition = {0};
  AlquimiaProperties properties = {0};
  OnnxTestEngine engine;
  AlquimiaState state = {0};

  if (!OnnxSetupEngine("deterministic/identity_double.json", false, &engine))
  {
    ONNX_TEST_EXPECT(
        &num_failures, "C01", "Unable to set up the ONNX test engine",
        engine.status.error == kAlquimiaNoError, &engine.status);
    FreeAlquimiaEngineStatus(&engine.status);
    return;
  }

  /* | C01 | `condition == NULL` | No operation(No-op) success | */
  OnnxAllocateState(&engine, &state);
  state.total_mobile.data[0] = 17.0;

  engine.interface.ProcessCondition(
      &engine.engine_state, NULL, &properties, &state, &aux_data,
      &engine.status);
  ONNX_TEST_EXPECT(
      &num_failures, "C01", "NULL condition was not accepted as a no-op",
      engine.status.error == kAlquimiaNoError, &engine.status);
  ONNX_TEST_EXPECT(
      &num_failures, "C01", "NULL condition changed the existing state",
      state.total_mobile.data[0] == 17.0, NULL);

  /* | C02 | Condition has no aqueous constraints | No-op success | */
  AllocateAlquimiaGeochemicalCondition(5, 0, 0, &empty_condition);
  engine.interface.ProcessCondition(
      &engine.engine_state, &empty_condition, &properties, &state, &aux_data,
      &engine.status);
  ONNX_TEST_EXPECT(
      &num_failures, "C02",
      "Empty aqueous constraints were not accepted as a no-op",
      engine.status.error == kAlquimiaNoError, &engine.status);
  ONNX_TEST_EXPECT(
      &num_failures, "C02",
      "Empty aqueous constraints changed the existing state",
      state.total_mobile.data[0] == 17.0, NULL);
  FreeAlquimiaGeochemicalCondition(&empty_condition);
  FreeAlquimiaState(&state);
  ONNX_TEST_EXPECT(
      &num_failures, "C01/C02", "Condition engine shutdown failed",
      OnnxShutdownEngine(&engine), NULL);
}

/**
 * @brief Verifies condition routing into mixed scalar and vector destinations.
 *
 * | C03 | Scalar and vector destinations are mixed | Values route correctly |
 */
static void TestC03MixedCondition(void)
{
  AlquimiaAuxiliaryData aux_data = {0};
  AlquimiaGeochemicalCondition condition = {0};
  AlquimiaProperties properties = {0};
  OnnxTestEngine engine;
  AlquimiaState state = {0};

  if (!OnnxSetupEngine("deterministic/mixed_scalar_vector.json", false,
                       &engine))
  {
    ONNX_TEST_EXPECT(
        &num_failures, "C03", "Unable to set up the ONNX test engine",
        engine.status.error == kAlquimiaNoError, &engine.status);
    FreeAlquimiaEngineStatus(&engine.status);
    return;
  }

  OnnxAllocateState(&engine, &state);
  AllocateAlquimiaGeochemicalCondition(5, 3, 0, &condition);
  OnnxInitializeConstraint(&condition, 0, "porosity_feature", 0.31);
  OnnxInitializeConstraint(&condition, 1, "mobile_feature", 4.25);
  OnnxInitializeConstraint(&condition, 2, "gas_feature", 8.5);

  engine.interface.ProcessCondition(
      &engine.engine_state, &condition, &properties, &state, &aux_data,
      &engine.status);
  ONNX_TEST_EXPECT(
      &num_failures, "C03", "Mixed scalar/vector condition processing failed",
      engine.status.error == kAlquimiaNoError, &engine.status);
  ONNX_TEST_EXPECT(
      &num_failures, "C03", "Porosity constraint was inaccurate",
      state.porosity == 0.31, NULL);
  ONNX_TEST_EXPECT(
      &num_failures, "C03", "Mobile-total storage is NULL",
      state.total_mobile.data != NULL, NULL);
  ONNX_TEST_EXPECT(
      &num_failures, "C03",
      "Mobile-total storage is too small for mapped index 1",
      state.total_mobile.size > 1, NULL);
  if (state.total_mobile.data != NULL && state.total_mobile.size > 1)
  {
    ONNX_TEST_EXPECT(
        &num_failures, "C03", "Mobile-total constraint was inaccurate",
        state.total_mobile.data[1] == 4.25, NULL);
  }
  ONNX_TEST_EXPECT(
      &num_failures, "C03", "Gas-concentration storage is NULL",
      state.gas_concentration.data != NULL, NULL);
  ONNX_TEST_EXPECT(
      &num_failures, "C03",
      "Gas-concentration storage is too small for mapped index 0",
      state.gas_concentration.size > 0, NULL);
  if (state.gas_concentration.data != NULL &&
      state.gas_concentration.size > 0)
  {
    ONNX_TEST_EXPECT(
        &num_failures, "C03", "Gas-concentration constraint was inaccurate",
        state.gas_concentration.data[0] == 8.5, NULL);
  }

  FreeAlquimiaGeochemicalCondition(&condition);
  FreeAlquimiaState(&state);
  ONNX_TEST_EXPECT(
      &num_failures, "C03", "Mixed-condition engine shutdown failed",
      OnnxShutdownEngine(&engine), NULL);
}

/**
 * @brief Verifies named JSON conditions can initialize independent cells.
 *
 * | C04 | Named JSON condition is applied to multiple cells | No sharing |
 */
static void TestC04NamedJsonCondition(void)
{
  AlquimiaAuxiliaryData aux_data = {0};
  AlquimiaGeochemicalCondition condition = {0};
  AlquimiaProperties properties = {0};
  OnnxTestEngine engine;
  AlquimiaState first_state = {0};
  AlquimiaState second_state = {0};

  if (!OnnxSetupEngine("deterministic/named_condition.json", true, &engine))
  {
    ONNX_TEST_EXPECT(
        &num_failures, "C04", "Unable to set up the ONNX test engine",
        engine.status.error == kAlquimiaNoError, &engine.status);
    FreeAlquimiaEngineStatus(&engine.status);
    return;
  }

  OnnxAllocateState(&engine, &first_state);
  OnnxAllocateState(&engine, &second_state);
  AllocateAlquimiaGeochemicalCondition(
      (int)strlen("initial"), 0, 0, &condition);
  strcpy(condition.name, "initial");

  /* Data from the driver side */
  first_state.total_mobile.data[0] = 91.0;
  engine.interface.ProcessCondition(
      &engine.engine_state, &condition, &properties, &first_state, &aux_data,
      &engine.status);
  ONNX_TEST_EXPECT(
      &num_failures, "C04", "Named JSON condition was not applied",
      engine.status.error == kAlquimiaNoError, &engine.status);
  ONNX_TEST_EXPECT(
      &num_failures, "C04",
      "JSON input value was routed to the wrong state location",
      first_state.total_mobile.data[0] == 9.999999999999999e-06, NULL);

  first_state.total_mobile.data[0] = 12.0;
  engine.interface.ProcessCondition(
      &engine.engine_state, &condition, &properties, &second_state, &aux_data,
      &engine.status);
  ONNX_TEST_EXPECT(
      &num_failures, "C04", "Second-cell JSON condition was not applied",
      engine.status.error == kAlquimiaNoError, &engine.status);
  ONNX_TEST_EXPECT(
      &num_failures, "C04",
      "Second cell did not receive its own JSON input value",
      second_state.total_mobile.data[0] == 9.999999999999999e-06, NULL);
  ONNX_TEST_EXPECT(
      &num_failures, "C04", "Initializing a second cell changed the first cell",
      first_state.total_mobile.data[0] == 12.0, NULL);

  FreeAlquimiaGeochemicalCondition(&condition);
  FreeAlquimiaState(&first_state);
  FreeAlquimiaState(&second_state);
  ONNX_TEST_EXPECT(
      &num_failures, "C04", "JSON-condition engine shutdown failed",
      OnnxShutdownEngine(&engine), NULL);
}

/**
 * @brief Verifies driver-supplied constraints are honored in normal mode.
 *
 * | C05 | Driver constraints are used in normal mode | Values route correctly |
 */
static void TestC05DriverCondition(void)
{
  AlquimiaAuxiliaryData aux_data = {0};
  AlquimiaGeochemicalCondition condition = {0};
  AlquimiaProperties properties = {0};
  OnnxTestEngine engine;
  AlquimiaState state = {0};

  if (!OnnxSetupEngine("deterministic/named_condition.json", false, &engine))
  {
    ONNX_TEST_EXPECT(
        &num_failures, "C05", "Unable to set up the ONNX test engine",
        engine.status.error == kAlquimiaNoError, &engine.status);
    FreeAlquimiaEngineStatus(&engine.status);
    return;
  }

  OnnxAllocateState(&engine, &state);
  AllocateAlquimiaGeochemicalCondition(
      (int)strlen("driver"), 1, 0, &condition);
  strcpy(condition.name, "driver");
  OnnxInitializeConstraint(&condition, 0, "H", 3.25);

  engine.interface.ProcessCondition(
      &engine.engine_state, &condition, &properties, &state, &aux_data,
      &engine.status);
  ONNX_TEST_EXPECT(
      &num_failures, "C05", "Normal mode rejected driver constraints",
      engine.status.error == kAlquimiaNoError, &engine.status);
  ONNX_TEST_EXPECT(
      &num_failures, "C05",
      "Normal mode did not preserve driver constraint routing",
      state.total_mobile.data[0] == 3.25, NULL);

  FreeAlquimiaGeochemicalCondition(&condition);
  FreeAlquimiaState(&state);
  ONNX_TEST_EXPECT(
      &num_failures, "C05", "Normal-mode engine shutdown failed",
      OnnxShutdownEngine(&engine), NULL);
}

/* ---------- Error Cases ---------- */

/**
 * @brief Verifies raw ONNX model files are rejected as config input.
 *
 * | E03 | Raw ONNX file is passed as setup input | Strict JSON error |
 */
static void TestE03RawModelRejected(void)
{
  AlquimiaEngineFunctionality functionality = {0};
  AlquimiaEngineStatus status = {0};
  AlquimiaInterface interface = {0};
  AlquimiaSizes sizes = {0};
  char path[ONNX_TEST_PATH_SIZE];
  void *onnx_engine_state = NULL;

  if (!OnnxCreateInterface(&interface, &status))
  {
    ONNX_TEST_EXPECT(
        &num_failures, "E03", "Unable to create the ONNX interface",
        status.error == kAlquimiaNoError, &status);
    FreeAlquimiaEngineStatus(&status);
    return;
  }
  ONNX_TEST_EXPECT(
      &num_failures, "E03", "Unable to construct the ONNX model path",
      OnnxModelPath("alsurf_nn/zn_h_regressor_integrated_1D.onnx", path,
                    sizeof(path)), NULL);
  interface.Setup(path, false, &onnx_engine_state, &sizes, &functionality,
                  &status);
  ONNX_TEST_EXPECT(
      &num_failures, "E03", "Raw ONNX file was not rejected as a config",
      status.error == kAlquimiaErrorEngineIntegrity, &status);
  ONNX_TEST_EXPECT(
      &num_failures, "E03",
      "Raw ONNX rejection did not report a strict JSON error",
      status.message != NULL && strstr(status.message, "strict JSON") != NULL,
      &status);
  ONNX_TEST_EXPECT(
      &num_failures, "E03",
      "Raw ONNX rejection published a non-NULL engine state",
      onnx_engine_state == NULL, NULL);
  FreeAlquimiaEngineStatus(&status);
}

/**
 * @brief Runs setup cases that must fail before publishing an engine state.
 */
static void RunSetupErrorCases(void)
{
  static const OnnxSetupFailureCase cases[] = {
      {"E01", "invalid_models/config_does_not_exist.json",
       "Unable to read ONNX config"},
      {"E02-malformed", "invalid_models/malformed_config.json",
       "not valid strict JSON"},
      {"E02-trailing", "invalid_models/trailing_content.json",
       "not valid strict JSON"},
      {"E04", "invalid_models/missing_model.json",
       "ONNX model file not found"},
      {"E05", "invalid_models/not_onnx.json", "ONNX Runtime Error"},
      {"E06", "invalid_models/invalid_graph.json", "ONNX Runtime Error"},
      {"E07-zero-inputs", "invalid_models/zero_inputs.json",
       "Model has 0 inputs or outputs"},
      {"E07-zero-outputs", "invalid_models/zero_outputs.json",
       "ONNX Runtime Error"},
      {"E08", "invalid_models/unsupported_input_type.json",
       "ONNX input tensor 'input' must have double elements"},
      {"E09", "invalid_models/unsupported_output_type.json",
       "ONNX output tensor 'output' must have double elements"},
      {"E10", "invalid_models/dynamic_dimension.json",
       "unsupported dynamic extent"},
      {"E11", "invalid_models/runtime_setup_error.json",
       "ONNX Runtime Error"}};
  size_t i;

  for (i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
  {
    OnnxCheckSetupFailure(&num_failures, &cases[i]);
  }
  TestE03RawModelRejected();
}

/**
 * @brief Verifies condition-processing guards reject invalid state storage.
 *
 * | E13 | State pointer is NULL | Integrity error |
 * | E14 | Mapped state-vector storage is NULL | Integrity error |
 * | E15 | Runtime state vector is undersized | Integrity error |
 */
static void TestE13ConditionGuardFailures(void)
{
  AlquimiaAuxiliaryData aux_data = {0};
  AlquimiaGeochemicalCondition condition = {0};
  AlquimiaProperties properties = {0};
  OnnxTestEngine engine;
  AlquimiaState state = {0};
  double *saved_data;
  int saved_size;

  if (!OnnxSetupEngine("deterministic/identity_double.json", false, &engine))
  {
    ONNX_TEST_EXPECT(
        &num_failures, "E13", "Unable to set up the ONNX test engine",
        engine.status.error == kAlquimiaNoError, &engine.status);
    FreeAlquimiaEngineStatus(&engine.status);
    return;
  }
  /* | E13 | State pointer is `NULL` | Integrity error | */
  engine.interface.ProcessCondition(
      &engine.engine_state, NULL, &properties, NULL, &aux_data,
      &engine.status);
  ONNX_TEST_EXPECT(
      &num_failures, "E13", "NULL state pointer was not rejected",
      engine.status.error == kAlquimiaErrorEngineIntegrity, &engine.status);

  /* | E14 | Mapped state-vector storage is `NULL` | Integrity error | */
  OnnxAllocateState(&engine, &state);
  AllocateAlquimiaGeochemicalCondition(5, 1, 0, &condition);
  OnnxInitializeConstraint(&condition, 0, "identity_input", 23.0);

  saved_data = state.total_mobile.data;
  state.total_mobile.data = NULL;
  engine.interface.ProcessCondition(
      &engine.engine_state, &condition, &properties, &state, &aux_data,
      &engine.status);
  ONNX_TEST_EXPECT(
      &num_failures, "E14",
      "NULL mapped state-vector storage was not rejected",
      engine.status.error == kAlquimiaErrorEngineIntegrity, &engine.status);
  state.total_mobile.data = saved_data;

  /* | C07 | Runtime state vector is smaller than the derived setup size | Integrity error | */
  saved_size = state.total_mobile.size;
  state.total_mobile.size = 0;
  engine.interface.ProcessCondition(
      &engine.engine_state, &condition, &properties, &state, &aux_data,
      &engine.status);
  ONNX_TEST_EXPECT(
      &num_failures, "E15",
      "Undersized runtime state vector was not rejected",
      engine.status.error == kAlquimiaErrorEngineIntegrity, &engine.status);
  state.total_mobile.size = saved_size;

  FreeAlquimiaGeochemicalCondition(&condition);
  FreeAlquimiaState(&state);
  ONNX_TEST_EXPECT(
      &num_failures, "E13-E15", "Condition-guard engine shutdown failed",
      OnnxShutdownEngine(&engine), NULL);
}

/**
 * @brief Verifies unknown JSON condition names fail without mutating state.
 *
 * | E16 | Unknown JSON condition name | Error; state unchanged |
 */
static void TestE16UnknownJsonCondition(void)
{
  AlquimiaAuxiliaryData aux_data = {0};
  AlquimiaGeochemicalCondition condition = {0};
  AlquimiaProperties properties = {0};
  OnnxTestEngine engine;
  AlquimiaState state = {0};

  if (!OnnxSetupEngine("deterministic/named_condition.json", true, &engine))
  {
    ONNX_TEST_EXPECT(
        &num_failures, "E16", "Unable to set up the ONNX test engine",
        engine.status.error == kAlquimiaNoError, &engine.status);
    FreeAlquimiaEngineStatus(&engine.status);
    return;
  }

  OnnxAllocateState(&engine, &state);
  AllocateAlquimiaGeochemicalCondition(
      (int)strlen("Initial"), 0, 0, &condition);
  strcpy(condition.name, "Initial");
  state.total_mobile.data[0] = 44.0;

  engine.interface.ProcessCondition(
      &engine.engine_state, &condition, &properties, &state, &aux_data,
      &engine.status);
  ONNX_TEST_EXPECT(
      &num_failures, "E16", "Unknown JSON condition name was not rejected",
      engine.status.error == kAlquimiaErrorUnknownConstraintName,
      &engine.status);
  ONNX_TEST_EXPECT(
      &num_failures, "E16", "Unknown JSON condition changed the state",
      state.total_mobile.data[0] == 44.0, NULL);

  FreeAlquimiaGeochemicalCondition(&condition);
  FreeAlquimiaState(&state);
  ONNX_TEST_EXPECT(
      &num_failures, "E16", "Unknown-condition engine shutdown failed",
      OnnxShutdownEngine(&engine), NULL);
}

/**
 * @brief Verifies hands-off setup requires at least one JSON condition.
 *
 * | E17 | Hands-off setup lacks JSON conditions | Integrity error |
 */
static void TestE17HandsOffRequiresConditions(void)
{
  AlquimiaEngineFunctionality functionality = {0};
  AlquimiaEngineStatus status = {0};
  AlquimiaInterface interface = {0};
  AlquimiaSizes sizes = {0};
  char path[ONNX_TEST_PATH_SIZE];
  void *onnx_engine_state = NULL;

  if (!OnnxCreateInterface(&interface, &status))
  {
    ONNX_TEST_EXPECT(
        &num_failures, "E17", "Unable to create the ONNX interface",
        status.error == kAlquimiaNoError, &status);
    FreeAlquimiaEngineStatus(&status);
    return;
  }

  ONNX_TEST_EXPECT(
      &num_failures, "E17", "Unable to construct the ONNX test-case path",
      OnnxTestCasePath("deterministic/identity_double.json", path,
                       sizeof(path)), NULL);
  interface.Setup(path, true, &onnx_engine_state, &sizes, &functionality,
                  &status);
  ONNX_TEST_EXPECT(
      &num_failures, "E17",
      "Hands-off setup accepted a config without conditions",
      status.error == kAlquimiaErrorEngineIntegrity, &status);
  ONNX_TEST_EXPECT(
      &num_failures, "E17", "Missing-condition setup error was unclear",
      status.message != NULL &&
          strstr(status.message,
                 "requires at least one JSON condition") != NULL,
      &status);
  ONNX_TEST_EXPECT(
      &num_failures, "E17",
      "Failed hands-off setup published an engine state",
      onnx_engine_state == NULL, NULL);
  FreeAlquimiaEngineStatus(&status);
}

/**
 * @brief Verifies shutdown reports invalid null inputs without crashing.
 *
 * | E18 | Shutdown receives null pointers | Invalid-engine error; no crash |
 */
static void TestE18NullShutdown(void)
{
  AlquimiaEngineStatus status = {0};
  AlquimiaInterface interface = {0};
  void *onnx_engine_state = NULL;

  if (!OnnxCreateInterface(&interface, &status))
  {
    ONNX_TEST_EXPECT(
        &num_failures, "E18", "Unable to create the ONNX interface",
        status.error == kAlquimiaNoError, &status);
    FreeAlquimiaEngineStatus(&status);
    return;
  }

  interface.Shutdown(NULL, &status);
  ONNX_TEST_EXPECT(
      &num_failures, "E18",
      "Shutdown did not reject a NULL state destination",
      status.error == kAlquimiaErrorInvalidEngine, &status);
  interface.Shutdown(&onnx_engine_state, &status);
  ONNX_TEST_EXPECT(
      &num_failures, "E18", "Shutdown did not reject a NULL engine state",
      status.error == kAlquimiaErrorInvalidEngine, &status);
  ONNX_TEST_EXPECT(
      &num_failures, "E18", "Invalid shutdown changed the NULL engine state",
      onnx_engine_state == NULL, NULL);
  FreeAlquimiaEngineStatus(&status);
}

/* ---------- Special Cases ---------- */

/**
 * @brief Verifies repeated setup, inference, and shutdown lifecycle cleanup.
 *
 * | L01 | Shutdown follows successful setup | Engine state becomes NULL |
 */
static void TestL01RepeatedLifecycle(void)
{
  int iteration;

  for (iteration = 0; iteration < 16; ++iteration)
  {
    AlquimiaAuxiliaryData aux_data = {0};
    AlquimiaProperties properties = {0};
    OnnxTestEngine engine;
    AlquimiaState state = {0};

    if (!OnnxSetupEngine("deterministic/identity_double.json", false,
                         &engine))
    {
      ONNX_TEST_EXPECT(
          &num_failures, "L01", "Unable to set up the ONNX test engine",
          engine.status.error == kAlquimiaNoError, &engine.status);
      FreeAlquimiaEngineStatus(&engine.status);
      return;
    }

    OnnxAllocateState(&engine, &state);
    state.total_mobile.data[0] = (double)iteration + 0.5;
    engine.interface.ReactionStepOperatorSplit(
        &engine.engine_state, 1.0, &properties, &state, &aux_data, iteration,
        &engine.status);
    ONNX_TEST_EXPECT(
        &num_failures, "L01", "Repeated lifecycle inference failed",
        engine.status.error == kAlquimiaNoError, &engine.status);
    ONNX_TEST_EXPECT(
        &num_failures, "L01",
        "Repeated lifecycle inference returned the wrong value",
        state.total_mobile.data[0] == (double)iteration + 0.5, NULL);

    FreeAlquimiaState(&state);
    ONNX_TEST_EXPECT(
        &num_failures, "L01", "Repeated lifecycle shutdown failed",
        OnnxShutdownEngine(&engine), NULL);
  }
}

/* ---------- Runners ---------- */

/**
 * @brief Runs standard ONNX condition-processing cases.
 */
static void RunStandardCases(void)
{
  TestC01ConditionNoOpCases();
  TestC03MixedCondition();
  TestC04NamedJsonCondition();
  TestC05DriverCondition();
}

/**
 * @brief Runs all deliberate ONNX error-handling cases.
 */
static void RunErrorCases(void)
{
  RunSetupErrorCases();
  TestE13ConditionGuardFailures();
  TestE16UnknownJsonCondition();
  TestE17HandsOffRequiresConditions();
  TestE18NullShutdown();
}

/**
 * @brief Runs ONNX lifecycle and cleanup special cases.
 */
static void RunSpecialCases(void)
{
  TestL01RepeatedLifecycle();
}

#endif

/**
 * @brief Runs ONNX failure, condition, and lifecycle test groups.
 */
int main(void)
{
#if ALQUIMIA_HAVE_ONNX
  RunStandardCases();
  RunErrorCases();
  RunSpecialCases();

  if (num_failures != 0)
  {
    fprintf(stderr, "ONNX failure/lifecycle tests had %d failure(s).\n",
            num_failures);
    return EXIT_FAILURE;
  }
  printf("ONNX failure/lifecycle cases C01-C05, E01-E18, and L01 passed.\n");
#else
  printf("ONNX not enabled. Skipping ONNX failure/lifecycle tests.\n");
#endif
  return EXIT_SUCCESS;
}
