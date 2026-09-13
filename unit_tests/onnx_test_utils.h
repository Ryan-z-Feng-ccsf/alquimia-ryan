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
** Shared ONNX unit-test utilities.
**
** Authors:
**        Zhuolei Feng, Sergi Molins
**
** Notes:
**
**  * This header declares helpers shared by the ONNX unit-test files.
**  * It keeps setup, shutdown, path construction, temporary config files,
**    condition setup, inference execution, numeric checks, and diagnostic
**    failure reporting out of the individual test files.
**  * The test files own the case ordering and IDs:
**      Mxx: mapping/config cases.
**      Rxx: routing/inference cases.
**      Fxx: model-family compatibility cases.
**      Cxx: standard condition-routing cases.
**      Exx: deliberate error-handling cases.
**      Lxx: lifecycle special cases.
**  * The two digits are stable case numbers within a prefix, for example
**    R01 is the first routing case and E01 is the first error case.
**
** ****************************************************************************
*/

#ifndef ALQUIMIA_ONNX_TEST_UTILS_H_
#define ALQUIMIA_ONNX_TEST_UTILS_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>

#include "alquimia/alquimia_constants.h"
#include "alquimia/alquimia_interface.h"
#include "alquimia/alquimia_memory.h"

#if ALQUIMIA_HAVE_ONNX

/* ---------- macro definition ----------*/

#define ONNX_TEST_PATH_SIZE 2048
#define ONNX_TEST_ERROR_MESSAGE_SIZE 512
#define ONNX_TEST_TEMP_CONFIG "test_alquimia_onnx_config_case.json"

#define ONNX_TEST_EX8_MODEL_PATH                                      \
  CMAKE_CURRENT_SOURCE_DIR                                               \
  "/../models/ex8_nn/ex8_nn_1d.onnx"
#define ONNX_TEST_EX8_NAMED_CONFIG                                    \
  CMAKE_CURRENT_SOURCE_DIR                                               \
  "/onnx_test_cases/deterministic/named_condition.json"
#define ONNX_TEST_EX8_RELATIVE_CONFIG                                 \
  CMAKE_CURRENT_SOURCE_DIR                                               \
  "/onnx_test_cases/configs/ex8_relative.json"

#define ONNX_TEST_VALID_INPUT_0                                          \
  "{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0,"       \
  "\"feature\":\"H\",\"alquimia_state\":\"total_mobile\","              \
  "\"alquimia_state_index\":0}"
#define ONNX_TEST_VALID_INPUT_1                                          \
  "{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":1,"       \
  "\"feature\":\"Zn\",\"alquimia_state\":\"total_mobile\","             \
  "\"alquimia_state_index\":1}"
#define ONNX_TEST_VALID_OUTPUT_0                                         \
  "{\"tensor\":\"sorbed_output_raw\",\"tensor_element_index\":0,"        \
  "\"feature\":\"H\",\"alquimia_state\":\"total_immobile\","            \
  "\"alquimia_state_index\":0}"
#define ONNX_TEST_VALID_OUTPUT_1                                         \
  "{\"tensor\":\"sorbed_output_raw\",\"tensor_element_index\":1,"        \
  "\"feature\":\"Zn\",\"alquimia_state\":\"total_immobile\","           \
  "\"alquimia_state_index\":1}"
#define ONNX_TEST_VALID_OUTPUT \
  ONNX_TEST_VALID_OUTPUT_0 "," ONNX_TEST_VALID_OUTPUT_1

/*
** =====================================================================
** Note on macro definition:
** Avoid defining macros as simple if-statements: #define macro(x) if(){}
**
** If used inside an unbraced if-statement:
**     if (condition)
**         CHECK_CASE(test_id, expr);
**
** It expands to:
**     if (condition)
**         if (test_id) {};  <-- The habitual ';' ends the outer 'if'!
** =====================================================================
*/

/**
 * @brief For non-fatal collected checks.
 */
#define ONNX_TEST_EXPECT(failure_count, test_id, issue, condition, status) \
  do                                                                       \
  {                                                                        \
    if (!(condition))                                                      \
    {                                                                      \
      OnnxRecordTestFailure(                                               \
          failure_count, test_id, issue, #condition, status, __FILE__,      \
          __LINE__);                                                       \
    }                                                                      \
  } while (0)

/**
 * @brief Records a failed prerequisite and jumps to the caller's cleanup label.
 * @note Initialize owned resources before any requirement. Cleanup must use
 *       ONNX_TEST_EXPECT for checks so failures cannot jump back into cleanup.
 */
#define ONNX_TEST_REQUIRE(failure_count, condition)                        \
  do                                                                      \
  {                                                                       \
    if (!(condition))                                                     \
    {                                                                     \
      OnnxRecordTestFailure(                                              \
          failure_count, __func__, "Required condition failed",            \
          #condition, NULL, __FILE__, __LINE__);                            \
      goto cleanup;                                                       \
    }                                                                     \
  } while (0)

#define OnnxCheckSetupFailure(failure_count, test_case) \
  OnnxCheckSetupFailureAt(                              \
      failure_count, test_case, __FILE__, __LINE__)

/* --------- Test Struct ---------- */

/* Elements needed for interface set up */
typedef struct
{
  AlquimiaEngineStatus status;
  AlquimiaInterface interface;
  AlquimiaSizes sizes;
  AlquimiaEngineFunctionality functionality;
  void *engine_state;
} OnnxTestEngine;

typedef struct
{
  const char *test_id;
  const char *relative_path;
  const char *message_fragment;
} OnnxSetupFailureCase;

typedef struct
{
  const char *test_id;
  const char *config_path;
  double expected_h;
  double expected_zn;
} OnnxEx8ModelCase;

/* ---------- Test Helper Function ---------- */

/* Fallible helpers return success. Callers record failures and clean up owned
 * resources, including engine status after failed setup. */

bool OnnxTestCasePath(const char *relative_path, char *path, size_t size);

bool OnnxModelPath(const char *relative_path, char *path, size_t size);

bool OnnxCreateInterface(
    AlquimiaInterface *interface,
    AlquimiaEngineStatus *status);

bool OnnxSetupEngineAtPath(
    const char *path,
    bool hands_off,
    OnnxTestEngine *engine);

bool OnnxSetupEngine(
    const char *relative_path,
    bool hands_off,
    OnnxTestEngine *engine);

bool OnnxSetupSharedModelEngine(
    const char *relative_path,
    bool hands_off,
    OnnxTestEngine *engine);

bool OnnxShutdownEngine(OnnxTestEngine *engine);

bool OnnxRequireSetupEngine(
    const char *relative_path,
    bool hands_off,
    OnnxTestEngine *engine);

bool OnnxRequireSharedModelEngine(
    const char *relative_path,
    bool hands_off,
    OnnxTestEngine *engine);

void OnnxAllocateState(const OnnxTestEngine *engine, AlquimiaState *state);

void OnnxInitializeConstraint(
    AlquimiaGeochemicalCondition *condition,
    int index,
    const char *name,
    double value);

bool OnnxRunInference(OnnxTestEngine *engine, AlquimiaState *state);

bool OnnxApplyNamedCondition(
    OnnxTestEngine *engine,
    const char *condition_name,
    AlquimiaState *state);

bool OnnxCloseEnough(double actual, double expected, double tolerance);

void OnnxRecordTestFailure(
    int *failure_count,
    const char *test_id,
    const char *issue,
    const char *condition,
    const AlquimiaEngineStatus *status,
    const char *file,
    int line);

void OnnxCheckSetupFailureAt(
    int *failure_count,
    const OnnxSetupFailureCase *test_case,
    const char *file,
    int line);

bool OnnxCheckEx8Prediction(
    const char *test_id,
    const char *feature,
    double actual,
    double expected);

bool OnnxWriteTemporaryConfig(const char *contents);

bool OnnxRemoveTemporaryConfig(const char *test_id);

bool OnnxExpectConfigParseFailure(
    const char *test_id,
    const char *config_contents,
    const char *expected_message);

bool OnnxExpectConfigSetupFailure(
    AlquimiaInterface *interface,
    AlquimiaEngineStatus *status,
    const char *test_id,
    const char *config_contents,
    const char *expected_message);

#endif

#endif
