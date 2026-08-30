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
** ONNX mapping/configuration unit tests.
**
** Authors:
**        Zhuolei Feng, Sergi Molins
**
** Notes:
**
**  * This file verifies the strict JSON config contract, feature-to-state
**    mapping metadata, and relative model-path resolution.
**  * Test IDs use one prefix plus a two-digit stable case number.
**      M01, M02, ...: standard mapping/config success cases.
**      E01, E02, ...: parser/setup contract failures.
**  * Example: M01 is the first mapping success case; E01 is the first
**    mapping/config error case.
**  * Tightly coupled: validates by matching substrings in the interface's
**    error message.
**  * Test JSON is assembled in each test function.
**
** ****************************************************************************
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "alquimia/alquimia_constants.h"
#include "alquimia/alquimia_interface.h"
#include "alquimia/alquimia_memory.h"
#include "alquimia/alquimia_util.h"
#include "alquimia/onnx_alquimia_config.h"
#include "onnx_test_utils.h"

#if ALQUIMIA_HAVE_ONNX

/* ---------- Standard Cases ---------- */

/**
 * @brief Verifies valid named conditions parse and preserve extra features.
 *
 * | M01 | Conditions cover all required inputs and include extra features | Configuration parses successfully |
 */
static void TestM01ConditionConfigSuccess(void)
{
  static const char valid_config[] =
      "{\"schema_version\":1,\"model\":\"model.onnx\","
      "\"conditions\":{\"initial\":{\"H\":1e-5,"
      "\"Zn\":1e-7,\"unused_feature\":7}},"
      "\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "," ONNX_TEST_VALID_INPUT_1 "],"
      "\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}";

  OnnxAlquimiaConfig config = {0};
  char error_message[ONNX_TEST_ERROR_MESSAGE_SIZE] = {0};

  printf("  M01 conditions cover inputs and allow extra features\n");
  OnnxWriteTemporaryConfig(valid_config);

  if (!OnnxAlquimiaLoadConfig(
      ONNX_TEST_TEMP_CONFIG, &config, error_message, sizeof(error_message)))
  {
    fprintf(stderr, "M01 could not parse valid conditions: %s\n",
            error_message);
    OnnxRemoveTemporaryConfig("M01");
    exit(EXIT_FAILURE);
  }

  ONNX_TEST_REQUIRE(config.num_conditions == 1);
  ONNX_TEST_REQUIRE(strcmp(config.conditions[0].name, "initial") == 0);
  ONNX_TEST_REQUIRE(config.conditions[0].num_items == 3);
  ONNX_TEST_REQUIRE(
      strcmp(config.conditions[0].items[0].feature, "H") == 0);
  ONNX_TEST_REQUIRE(config.conditions[0].items[0].value == 1.0e-5);
  ONNX_TEST_REQUIRE(
      strcmp(config.conditions[0].items[2].feature, "unused_feature") == 0);
  ONNX_TEST_REQUIRE(config.conditions[0].items[2].value == 7.0);
  OnnxAlquimiaFreeConfig(&config);
  ONNX_TEST_REQUIRE(config.conditions == NULL);
  ONNX_TEST_REQUIRE(config.num_conditions == 0);

  OnnxRemoveTemporaryConfig("M01");
}

/**
 * @brief Verifies mapping metadata names are published by state category.
 *
 * | M02 | Feature names target different state categories | Correct problem-metadata vectors are populated |
 */
static void TestM02SuccessfulMappingMetadata(void)
{
  static const char category_config[] =
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":["
      "{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"aqueous_feature\",\"alquimia_state\":\"total_mobile\","
      "\"alquimia_state_index\":0},"
      "{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":1,"
      "\"feature\":\"gas_feature\",\"alquimia_state\":\"gas_concentration\","
      "\"alquimia_state_index\":0}],\"outputs\":[{"
      "\"tensor\":\"sorbed_output_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"output_feature\",\"alquimia_state\":\"total_mobile\","
      "\"alquimia_state_index\":1},{"
      "\"tensor\":\"sorbed_output_raw\",\"tensor_element_index\":1,"
      "\"feature\":\"output_gas_feature\","
      "\"alquimia_state\":\"gas_concentration\","
      "\"alquimia_state_index\":1}]}";

  OnnxTestEngine engine;
  AlquimiaProblemMetaData meta_data;

  OnnxWriteTemporaryConfig(category_config);

  if (!OnnxSetupEngineAtPath(ONNX_TEST_TEMP_CONFIG, false, &engine))
  {
    fprintf(stderr, "M02 setup failed: %s\n", engine.status.message);
  }

  ONNX_TEST_REQUIRE(engine.status.error == kAlquimiaNoError);
  ONNX_TEST_REQUIRE(engine.engine_state != NULL);
  ONNX_TEST_REQUIRE(engine.sizes.num_primary == 2);
  ONNX_TEST_REQUIRE(engine.sizes.num_gases == 2);

  AllocateAlquimiaProblemMetaData(&engine.sizes, &meta_data);
  engine.interface.GetProblemMetaData(
      &engine.engine_state, &meta_data, &engine.status);

  ONNX_TEST_REQUIRE(engine.status.error == kAlquimiaNoError);
  ONNX_TEST_REQUIRE(strcmp(meta_data.primary_names.data[0],
                         "aqueous_feature") == 0);
  ONNX_TEST_REQUIRE(strcmp(meta_data.primary_names.data[1],
                         "output_feature") == 0);
  ONNX_TEST_REQUIRE(strcmp(meta_data.gas_names.data[0], "gas_feature") == 0);
  ONNX_TEST_REQUIRE(strcmp(meta_data.gas_names.data[1],
                         "output_gas_feature") == 0);

  FreeAlquimiaProblemMetaData(&meta_data);

  ONNX_TEST_REQUIRE(OnnxShutdownEngine(&engine));

  OnnxRemoveTemporaryConfig("M02");
}

/**
 * @brief Verifies relative ONNX model paths resolve from the config directory.
 *
 * | M03 | Model path is relative to the config | Correct path is resolved from the config directory |
 */
static void TestM03RelativeModelPath(void)
{
  OnnxTestEngine engine;

  if (!OnnxSetupEngineAtPath(ONNX_TEST_ALSURF_RELATIVE_CONFIG, false, &engine))
  {
    fprintf(stderr, "M03 setup failed: %s\n", engine.status.message);
  }

  ONNX_TEST_REQUIRE(engine.status.error == kAlquimiaNoError);
  ONNX_TEST_REQUIRE(engine.engine_state != NULL);
  ONNX_TEST_REQUIRE(OnnxShutdownEngine(&engine));
}

/* ---------- Error Cases ---------- */

/**
 * @brief Verifies invalid condition config schemas fail during parsing.
 *
 * | E01 | Invalid condition schema | Parser rejects the configuration with a specific error message |
 */
static void TestE01ConditionConfigFailures(void)
{
  /* Conditions must be an object. */
  OnnxExpectConfigParseFailure(
      "E01 conditions must be an object",
      "{\"schema_version\":1,\"model\":\"model.onnx\","
      "\"conditions\":[],\"inputs\":[],\"outputs\":[]}",
      "conditions must be an object");

  /* Duplicate top-level conditions property. */
  OnnxExpectConfigParseFailure(
      "E01 duplicate conditions property",
      "{\"schema_version\":1,\"model\":\"model.onnx\","
      "\"conditions\":{},\"conditions\":{},\"inputs\":[],\"outputs\":[]}",
      "Duplicate property 'conditions'");

  /* Duplicate condition name. */
  OnnxExpectConfigParseFailure(
      "E01 duplicate condition name",
      "{\"schema_version\":1,\"model\":\"model.onnx\","
      "\"conditions\":{\"initial\":{},\"initial\":{}},"
      "\"inputs\":[],\"outputs\":[]}",
      "Duplicate name 'initial'");

  /* Empty condition name. */
  OnnxExpectConfigParseFailure(
      "E01 empty condition name",
      "{\"schema_version\":1,\"model\":\"model.onnx\","
      "\"conditions\":{\"\":{}},\"inputs\":[],\"outputs\":[]}",
      "must be nonempty");

  /* Condition bodies must be objects. */
  OnnxExpectConfigParseFailure(
      "E01 condition must be an object",
      "{\"schema_version\":1,\"model\":\"model.onnx\","
      "\"conditions\":{\"initial\":1},\"inputs\":[],\"outputs\":[]}",
      "Condition 'initial' must be an object");

  /* Duplicate condition feature. */
  OnnxExpectConfigParseFailure(
      "E01 duplicate condition feature",
      "{\"schema_version\":1,\"model\":\"model.onnx\","
      "\"conditions\":{\"initial\":{\"f\":1,\"f\":2}},"
      "\"inputs\":[],\"outputs\":[]}",
      "Duplicate name 'f'");

  /* Empty condition feature. */
  OnnxExpectConfigParseFailure(
      "E01 empty condition feature",
      "{\"schema_version\":1,\"model\":\"model.onnx\","
      "\"conditions\":{\"initial\":{\"\":1}},"
      "\"inputs\":[],\"outputs\":[]}",
      "must be nonempty");

  /* Condition values must be numeric. */
  OnnxExpectConfigParseFailure(
      "E01 condition feature must be numeric",
      "{\"schema_version\":1,\"model\":\"model.onnx\","
      "\"conditions\":{\"initial\":{\"f\":\"1\"}},"
      "\"inputs\":[],\"outputs\":[]}",
      "must be a finite number");

  /* Condition values must be finite. */
  OnnxExpectConfigParseFailure(
      "E01 condition feature must be finite",
      "{\"schema_version\":1,\"model\":\"model.onnx\","
      "\"conditions\":{\"initial\":{\"f\":1e999}},"
      "\"inputs\":[],\"outputs\":[]}",
      "must be a finite number");

  /* Conditions must cover every configured input feature. */
  OnnxExpectConfigParseFailure(
      "E01 condition must cover every input feature",
      "{\"schema_version\":1,\"model\":\"model.onnx\","
      "\"conditions\":{\"initial\":{\"H\":1e-5}},"
      "\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "," ONNX_TEST_VALID_INPUT_1 "],"
      "\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}",
      "Condition 'initial' is missing input feature 'Zn'");
}

/**
 * @brief Verifies strict ONNX JSON mapping-contract failures.
 */
static void TestE02ConfigContractFailures(void)
{
  AlquimiaEngineStatus status;
  AlquimiaInterface interface;

  printf("Running strict ONNX config contract cases.\n");
  ONNX_TEST_REQUIRE(OnnxCreateInterface(&interface, &status));

  /* | E02 | Schema version is missing, malformed, or unsupported | Setup error naming `schema_version` | */
  OnnxExpectConfigSetupFailure(&interface, &status, "E02 missing schema version",
      "{\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[],\"outputs\":[]}", "schema_version");

  OnnxExpectConfigSetupFailure(&interface, &status, "E02 malformed schema version",
      "{\"schema_version\":\"1\",\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[],\"outputs\":[]}", "schema_version");

  OnnxExpectConfigSetupFailure(&interface, &status, "E02 unsupported schema version",
      "{\"schema_version\":2,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[],\"outputs\":[]}", "schema_version");

  /* | E03 | Required top-level object or array is missing | Setup error naming the property | */
  OnnxExpectConfigSetupFailure(&interface, &status, "E03 missing model",
      "{\"schema_version\":1,\"inputs\":[],\"outputs\":[]}", "model");

  OnnxExpectConfigSetupFailure(&interface, &status, "E03 missing inputs",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"outputs\":[]}", "inputs and outputs");

  OnnxExpectConfigSetupFailure(&interface, &status, "E03 missing outputs",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[]}", "inputs and outputs");

  /* | E04 | Input mapping property is missing | Setup error naming the property | */
  OnnxExpectConfigSetupFailure(&interface, &status, "E04 missing input tensor",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[{\"tensor_element_index\":0,\"feature\":\"f\","
      "\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0}],"
      "\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}", "tensor");

  OnnxExpectConfigSetupFailure(&interface, &status,
      "E04 missing input tensor element index",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\","
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0}],"
      "\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}", "tensor_element_index");

  OnnxExpectConfigSetupFailure(&interface, &status, "E04 missing input feature",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0,"
      "\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0}],"
      "\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}", "feature");

  OnnxExpectConfigSetupFailure(&interface, &status,
      "E04 missing input Alquimia state variable",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"f\",\"alquimia_state_index\":0}],"
      "\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}", "alquimia_state");

  OnnxExpectConfigSetupFailure(&interface, &status,
      "E04 missing input Alquimia state index",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\"}],"
      "\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}", "alquimia_state_index");

  /* | E05 | Output mapping property is missing | Setup error naming the property | */
  OnnxExpectConfigSetupFailure(&interface, &status, "E05 missing output tensor",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "," ONNX_TEST_VALID_INPUT_1
      "],\"outputs\":[{\"tensor_element_index\":0,\"feature\":\"H\","
      "\"alquimia_state\":\"total_mobile\","
      "\"alquimia_state_index\":0}]}", "tensor");

  OnnxExpectConfigSetupFailure(&interface, &status,
      "E05 missing output tensor element index",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "," ONNX_TEST_VALID_INPUT_1
      "],\"outputs\":[{\"tensor\":\"sorbed_output_raw\","
      "\"feature\":\"H\",\"alquimia_state\":\"total_mobile\","
      "\"alquimia_state_index\":0}]}", "tensor_element_index");

  OnnxExpectConfigSetupFailure(&interface, &status, "E05 missing output feature",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "," ONNX_TEST_VALID_INPUT_1
      "],\"outputs\":[{\"tensor\":\"sorbed_output_raw\","
      "\"tensor_element_index\":0,\"alquimia_state\":\"total_mobile\","
      "\"alquimia_state_index\":0}]}", "feature");

  OnnxExpectConfigSetupFailure(&interface, &status,
      "E05 missing output Alquimia state variable",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "," ONNX_TEST_VALID_INPUT_1
      "],\"outputs\":[{\"tensor\":\"sorbed_output_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"H\",\"alquimia_state_index\":0}]}",
      "alquimia_state");

  OnnxExpectConfigSetupFailure(&interface, &status,
      "E05 missing output Alquimia state index",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "," ONNX_TEST_VALID_INPUT_1
      "],\"outputs\":[{\"tensor\":\"sorbed_output_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"H\",\"alquimia_state\":\"total_mobile\"}]}",
      "alquimia_state_index");

  /* | E06 | Unknown or duplicate JSON property | Setup error | */
  OnnxExpectConfigSetupFailure(&interface, &status, "E06 unknown property",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[],\"outputs\":[],\"models\":\"extra\"}",
      "Unknown property 'models'");

  OnnxExpectConfigSetupFailure(&interface, &status, "E06 duplicate property",
      "{\"schema_version\":1,\"schema_version\":1,\"model\":\""
      ONNX_TEST_ALSURF_MODEL_PATH "\",\"inputs\":[],\"outputs\":[]}",
      "Duplicate property 'schema_version'");

  OnnxExpectConfigSetupFailure(&interface, &status, "E06 unknown input property",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0,"
      "\"unit\":\"molar\"}],\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}",
      "Unknown property 'unit'");

  OnnxExpectConfigSetupFailure(&interface, &status, "E06 duplicate output property",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "," ONNX_TEST_VALID_INPUT_1
      "],\"outputs\":[{\"tensor\":\"sorbed_output_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"H\",\"alquimia_state\":\"total_mobile\","
      "\"alquimia_state_index\":0,\"alquimia_state_index\":0}]}",
      "Duplicate property 'alquimia_state_index'");

  /* | E07 | `alquimia_state` is unsupported | Setup error | */
  OnnxExpectConfigSetupFailure(&interface, &status,
      "E07 unsupported Alquimia state variable",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"H\",\"alquimia_state\":\"total_mobiles\","
      "\"alquimia_state_index\":0}," ONNX_TEST_VALID_INPUT_1 "],\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}",
      "Unsupported AlquimiaState variable 'total_mobiles'");

  /* | E08 | `alquimia_state_index` or `tensor_element_index` is invalid | Setup error | */
  OnnxExpectConfigSetupFailure(&interface, &status,
      "E08 negative tensor element index",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":-1,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0}],"
      "\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}", "tensor_element_index");

  OnnxExpectConfigSetupFailure(&interface, &status,
      "E08 fractional Alquimia state index",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0.5}],"
      "\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}", "alquimia_state_index");

  OnnxExpectConfigSetupFailure(&interface, &status,
      "E08 negative Alquimia state index",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":-1}],"
      "\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}", "alquimia_state_index");

  OnnxExpectConfigSetupFailure(&interface, &status,
      "E08 fractional tensor element index",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0.5,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0}],"
      "\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}", "tensor_element_index");

  /* tensor_element_index = 2147483648 > INT_MAX, in the onnx_alquimia_config.c */
  OnnxExpectConfigSetupFailure(&interface, &status,
      "E08 tensor element index exceeds C int range",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\","
      "\"tensor_element_index\":2147483648,\"feature\":\"f\","
      "\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0}],"
      "\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}", "tensor_element_index");

  OnnxExpectConfigSetupFailure(&interface, &status,
      "E08 Alquimia state index exceeds C int range",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\","
      "\"alquimia_state_index\":2147483648}],\"outputs\":[" ONNX_TEST_VALID_OUTPUT
      "]}", "alquimia_state_index");

  /* | E09 | Scalar mapping uses a nonzero `alquimia_state_index` | Setup error | */
  OnnxExpectConfigSetupFailure(&interface, &status,
      "E09 scalar has nonzero Alquimia state index",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"temperature\",\"alquimia_state\":\"temperature\","
      "\"alquimia_state_index\":1}," ONNX_TEST_VALID_INPUT_1 "],\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}",
      "incompatible with variable 'temperature'");

  /* | E10 | Tensor name is unknown | Setup error naming the tensor | */
  OnnxExpectConfigSetupFailure(&interface, &status, "E10 unknown tensor",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"invalid_input\",\"tensor_element_index\":0,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0},"
      ONNX_TEST_VALID_INPUT_1 "],\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}",
      "unknown tensor 'invalid_input'");

  /* | E11 | `tensor_element_index` exceeds its flattened extent | Setup error | */
  OnnxExpectConfigSetupFailure(&interface, &status,
      "E11 tensor element index exceeds tensor extent",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":2,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0},"
      ONNX_TEST_VALID_INPUT_1 "],\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}", "out of range");

  OnnxExpectConfigSetupFailure(&interface, &status, "E11 output exceeds tensor extent",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "," ONNX_TEST_VALID_INPUT_1
      "],\"outputs\":[{\"tensor\":\"sorbed_output_raw\",\"tensor_element_index\":2,"
      "\"feature\":\"H\",\"alquimia_state\":\"total_mobile\","
      "\"alquimia_state_index\":0}]}", "out of range");

  /* | E12 | A tensor element index is mapped more than once | Setup error | */
  OnnxExpectConfigSetupFailure(&interface, &status,
      "E12 duplicate tensor element index",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "," ONNX_TEST_VALID_INPUT_0
      "],\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}", "Duplicate ONNX input mapping");

  OnnxExpectConfigSetupFailure(&interface, &status,
      "E12 duplicate output tensor element index",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "," ONNX_TEST_VALID_INPUT_1
      "],\"outputs\":[" ONNX_TEST_VALID_OUTPUT "," ONNX_TEST_VALID_OUTPUT "]}",
      "Duplicate ONNX output mapping");

  /* | E13 | A required model tensor element index is unmapped | Setup error | */
  OnnxExpectConfigSetupFailure(&interface, &status,
      "E13 tensor element index is unmapped",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "],\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}",
      "does not map every input tensor element");

  OnnxExpectConfigSetupFailure(&interface, &status,
      "E13 output tensor element index is unmapped",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "," ONNX_TEST_VALID_INPUT_1
      "],\"outputs\":[]}", "does not map every output tensor element");

  /* | E14 | Mappings derive unsafe or overflowing Alquimia sizes | Setup error | */
  OnnxExpectConfigSetupFailure(&interface, &status, "E14 derived size overflows int",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\","
      "\"alquimia_state_index\":2147483647}," ONNX_TEST_VALID_INPUT_1 "],"
      "\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}", "incompatible with variable");

  /* | E15 | Two names target the same metadata destination | Setup rejects the ambiguity | */
  OnnxExpectConfigSetupFailure(&interface, &status, "E15 conflicting feature names",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"first_name\",\"alquimia_state\":\"total_mobile\","
      "\"alquimia_state_index\":0},{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":1,"
      "\"feature\":\"second_name\",\"alquimia_state\":\"total_mobile\","
      "\"alquimia_state_index\":0}],\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}",
      "Conflicting ONNX feature names");

  OnnxExpectConfigSetupFailure(&interface, &status,
      "E15 conflicting input and output feature names",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "," ONNX_TEST_VALID_INPUT_1
      "],\"outputs\":[{\"tensor\":\"sorbed_output_raw\","
      "\"tensor_element_index\":0,\"feature\":\"different_name\","
      "\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0},"
      ONNX_TEST_VALID_OUTPUT_1 "]}",
      "Conflicting ONNX feature names");

  OnnxExpectConfigSetupFailure(&interface, &status,
      "E15 conflicting mobile and immobile feature names",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "," ONNX_TEST_VALID_INPUT_1
      "],\"outputs\":[{\"tensor\":\"sorbed_output_raw\","
      "\"tensor_element_index\":0,\"feature\":\"different_name\","
      "\"alquimia_state\":\"total_immobile\",\"alquimia_state_index\":0},"
      ONNX_TEST_VALID_OUTPUT_1 "]}",
      "Conflicting ONNX feature names");

  /* | E16 | Similar but invalid property has trailing text | Property is rejected rather than partially matched | */
  OnnxExpectConfigSetupFailure(&interface, &status, "E16 property trailing text",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0,"
      "\"alquimia_state_index_extra\":0}],\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}",
      "Unknown property 'alquimia_state_index_extra'");

  /* | E17 | Two input mappings use the same feature name | Setup rejects the duplicate lookup key | */
  OnnxExpectConfigSetupFailure(&interface, &status,
      "E17 duplicate input feature name",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_ALSURF_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"duplicate_feature\",\"alquimia_state\":\"total_mobile\","
      "\"alquimia_state_index\":0},{\"tensor\":\"chemical_input_raw\","
      "\"tensor_element_index\":1,\"feature\":\"duplicate_feature\","
      "\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":1}],"
      "\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}",
      "Duplicate ONNX input feature name 'duplicate_feature'.");

  FreeAlquimiaEngineStatus(&status);
}

/* ---------- Runners ---------- */

/**
 * @brief Runs ONNX mapping/config cases.
 */
static void RunMappingTests(void)
{
  TestM01ConditionConfigSuccess();
  TestM02SuccessfulMappingMetadata();
  TestM03RelativeModelPath();
  TestE01ConditionConfigFailures();
  TestE02ConfigContractFailures();
}

#endif

/**
 * @brief Runs ONNX mapping/config tests.
 */
int main(int argc, char **argv)
{
#if ALQUIMIA_HAVE_ONNX
  (void)argc;
  (void)argv;

  setvbuf(stdout, NULL, _IONBF, 0);
  setvbuf(stderr, NULL, _IONBF, 0);

  RunMappingTests();
#else
  (void)argc;
  (void)argv;
  printf("ONNX not enabled. Skipping ONNX engine unit test.\n");
#endif
  return EXIT_SUCCESS;
}
