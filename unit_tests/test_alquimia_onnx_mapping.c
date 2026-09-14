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

static int num_failures = 0;

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

  bool config_written = false;

  printf("  M01 conditions cover inputs and allow extra features\n");
  ONNX_TEST_REQUIRE(&num_failures, OnnxWriteTemporaryConfig(valid_config));
  config_written = true;

  if (!OnnxAlquimiaLoadConfig(
      ONNX_TEST_TEMP_CONFIG, &config, error_message, sizeof(error_message)))
  {
    fprintf(stderr, "M01 could not parse valid conditions: %s\n",
            error_message);
    ONNX_TEST_REQUIRE(&num_failures, false);
  }

  ONNX_TEST_REQUIRE(&num_failures, config.num_conditions == 1);
  ONNX_TEST_REQUIRE(&num_failures, strcmp(config.conditions[0].name, "initial") == 0);
  ONNX_TEST_REQUIRE(&num_failures, config.conditions[0].num_items == 3);
  ONNX_TEST_REQUIRE(&num_failures,
      strcmp(config.conditions[0].items[0].feature, "H") == 0);
  ONNX_TEST_REQUIRE(&num_failures, config.conditions[0].items[0].value == 1.0e-5);
  ONNX_TEST_REQUIRE(&num_failures,
      strcmp(config.conditions[0].items[2].feature, "unused_feature") == 0);
  ONNX_TEST_REQUIRE(&num_failures, config.conditions[0].items[2].value == 7.0);
cleanup:
  OnnxAlquimiaFreeConfig(&config);
  ONNX_TEST_EXPECT(&num_failures, "M01", "Config cleanup failed",
      config.conditions == NULL, NULL);
  ONNX_TEST_EXPECT(&num_failures, "M01", "Config cleanup failed",
      config.num_conditions == 0, NULL);

  if (config_written)
  {
    ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxRemoveTemporaryConfig failed",
        OnnxRemoveTemporaryConfig("M01"), NULL);
  }
}

/**
 * @brief Verifies mapping metadata names are published by state category.
 *
 * | M02 | Feature names target different state categories | Correct problem-metadata vectors are populated |
 */
static void TestM02SuccessfulMappingMetadata(void)
{
  static const char category_config[] =
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
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

  OnnxTestEngine engine = {0};
  AlquimiaProblemMetaData meta_data = {0};

  bool config_written = false;
  ONNX_TEST_REQUIRE(&num_failures, OnnxWriteTemporaryConfig(category_config));
  config_written = true;

  if (!OnnxSetupEngineAtPath(ONNX_TEST_TEMP_CONFIG, false, &engine))
  {
    fprintf(stderr, "M02 setup failed: %s\n", engine.status.message);
  }

  ONNX_TEST_REQUIRE(&num_failures, engine.status.error == kAlquimiaNoError);
  ONNX_TEST_REQUIRE(&num_failures, engine.engine_state != NULL);
  ONNX_TEST_REQUIRE(&num_failures, engine.sizes.num_primary == 2);
  ONNX_TEST_REQUIRE(&num_failures, engine.sizes.num_gases == 2);

  AllocateAlquimiaProblemMetaData(&engine.sizes, &meta_data);
  engine.interface.GetProblemMetaData(
      &engine.engine_state, &meta_data, &engine.status);

  ONNX_TEST_REQUIRE(&num_failures, engine.status.error == kAlquimiaNoError);
  ONNX_TEST_REQUIRE(&num_failures, strcmp(meta_data.primary_names.data[0],
                         "aqueous_feature") == 0);
  ONNX_TEST_REQUIRE(&num_failures, strcmp(meta_data.primary_names.data[1],
                         "output_feature") == 0);
  ONNX_TEST_REQUIRE(&num_failures, strcmp(meta_data.gas_names.data[0], "gas_feature") == 0);
  ONNX_TEST_REQUIRE(&num_failures, strcmp(meta_data.gas_names.data[1],
                         "output_gas_feature") == 0);

cleanup:
  FreeAlquimiaProblemMetaData(&meta_data);

  ONNX_TEST_EXPECT(&num_failures, __func__, "Shutdown failed",
      OnnxShutdownEngine(&engine), NULL);

  if (config_written)
  {
    ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxRemoveTemporaryConfig failed",
        OnnxRemoveTemporaryConfig("M02"), NULL);
  }
}

/**
 * @brief Verifies relative ONNX model paths resolve from the config directory.
 *
 * | M03 | Model path is relative to the config | Correct path is resolved from the config directory |
 */
static void TestM03RelativeModelPath(void)
{
  OnnxTestEngine engine = {0};

  if (!OnnxSetupEngineAtPath(ONNX_TEST_EX8_RELATIVE_CONFIG, false, &engine))
  {
    fprintf(stderr, "M03 setup failed: %s\n", engine.status.message);
  }

  ONNX_TEST_REQUIRE(&num_failures, engine.status.error == kAlquimiaNoError);
  ONNX_TEST_REQUIRE(&num_failures, engine.engine_state != NULL);
cleanup:
  ONNX_TEST_EXPECT(&num_failures, __func__, "Shutdown failed",
      OnnxShutdownEngine(&engine), NULL);
}

/**
 * @brief Verifies behavior of paired input fields sharing metadata names.
 *
 * | Inputs                       | Expected sum |
 * | Mobile + immobile             | 0.0012       |
 * | Total molar + mobile          | 0.0022       |
 * | Total molar + immobile        | 0.0014       |
 * | Mineral fraction + area      | 100.05       |
 *
 * Each case also checks metadata, partial initialization, and rejection of
 * ambiguous driver constraints without changing either field.
 */
static void TestM04PairedInputFields(void)
{
  typedef struct {
    const char *name;
    const char *feature;
    const char *left_field;
    const char *right_field;
    bool is_mineral;
    double expected_sum;
  } PairedInputCase;

  const PairedInputCase cases[] = {
      {"M04 mobile + immobile", "Zn",
       "total_mobile", "total_immobile", false, 0.0012},
      {"M04 total molar + mobile", "Zn",
       "total_molar", "total_mobile", false, 0.0022},
      {"M04 total molar + immobile", "Zn",
       "total_molar", "total_immobile", false, 0.0014},
      {"M04 mineral fraction + area", "Calcite",
       "mineral_volume_fraction", "mineral_specific_surface_area", true, 100.05}
  };

  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
  {
    const PairedInputCase *test_case = &cases[i];
    char json[4096];
    const char *fields = test_case->is_mineral
        ? "        \"mineral_volume_fraction\": 0.05,\n"
          "        \"mineral_specific_surface_area\": 100.0\n"
        : "        \"total_mobile\": 0.001,\n"
          "        \"total_immobile\": 0.2\n";
    const char *partial_field = test_case->is_mineral
        ? "mineral_volume_fraction" : "total_mobile";
    bool config_written = false;
    OnnxTestEngine engine = {0};
    AlquimiaState state = {0};
    AlquimiaProblemMetaData metadata = {0};
    AlquimiaGeochemicalCondition condition = {0};

    printf("  %s\n", test_case->name);

    /* Arrange: map two tensor inputs to distinct fields of the same identity. */
    int length = snprintf(json, sizeof(json),
        "{\n"
        "  \"schema_version\": 1,\n"
        "  \"model\": \"%s/onnx_test_cases/deterministic/add_two_inputs.onnx\",\n"
        "  \"inputs\": [\n"
        "    {\n"
        "      \"tensor\": \"left\", \"tensor_element_index\": 0,\n"
        "      \"feature\": \"%s\",\n"
        "      \"alquimia_state\": \"%s\", \"alquimia_state_index\": 2\n"
        "    },\n"
        "    {\n"
        "      \"tensor\": \"right\", \"tensor_element_index\": 0,\n"
        "      \"feature\": \"%s\",\n"
        "      \"alquimia_state\": \"%s\", \"alquimia_state_index\": 2\n"
        "    }\n"
        "  ],\n"
        "  \"outputs\": [{\n"
        "    \"tensor\": \"sum\", \"tensor_element_index\": 0,\n"
        "    \"feature\": \"temperature\",\n"
        "    \"alquimia_state\": \"temperature\", \"alquimia_state_index\": 0\n"
        "  }],\n"
        "  \"conditions\": {\n"
        "    \"initial\": {\n"
        "      \"%s\": {\n%s      }\n"
        "    },\n"
        "    \"partial\": {\"%s\": {\"%s\": -0.002}}\n"
        "  }\n"
        "}\n",
        CMAKE_CURRENT_SOURCE_DIR,
        test_case->feature, test_case->left_field,
        test_case->feature, test_case->right_field,
        test_case->feature, fields, test_case->feature, partial_field);
    ONNX_TEST_REQUIRE(&num_failures, length >= 0 && (size_t)length < sizeof(json));
    ONNX_TEST_REQUIRE(&num_failures, OnnxWriteTemporaryConfig(json));
    config_written = true;
    /* JSON provides the data */
    ONNX_TEST_REQUIRE(&num_failures, OnnxSetupEngineAtPath(ONNX_TEST_TEMP_CONFIG, true, &engine));

    /* Both fields must share an allocated metadata identity at index 2. */
    if (test_case->is_mineral)
    {
      ONNX_TEST_REQUIRE(&num_failures, engine.sizes.num_minerals == 3);
    }
    else
    {
      ONNX_TEST_REQUIRE(&num_failures, engine.sizes.num_primary == 3);
      ONNX_TEST_REQUIRE(&num_failures, engine.sizes.num_sorbed == 3);
    }
    
    /* The generic metadata allocator requires at least one primary species. */
    AllocateAlquimiaVectorString(engine.sizes.num_primary, &metadata.primary_names);
    AllocateAlquimiaVectorString(engine.sizes.num_minerals, &metadata.mineral_names);
    engine.interface.GetProblemMetaData(&engine.engine_state, &metadata, &engine.status);
    ONNX_TEST_REQUIRE(&num_failures, engine.status.error == kAlquimiaNoError);
    const char *metadata_name = test_case->is_mineral
        ? metadata.mineral_names.data[2] : metadata.primary_names.data[2];
    ONNX_TEST_REQUIRE(&num_failures, strcmp(metadata_name, test_case->feature) == 0);

    if (test_case->is_mineral)
    {
      /* The total_mobile = NULL, we can't use OnnxAllocateState to initialize the state */
      AllocateAlquimiaVectorDouble(3, &state.mineral_volume_fraction);
      AllocateAlquimiaVectorDouble(3, &state.mineral_specific_surface_area);
    }
    else
    {
      OnnxAllocateState(&engine, &state);
    }
    
    /* Full initialization supplies both fields in their native units. 
    */
    ONNX_TEST_REQUIRE(&num_failures, OnnxApplyNamedCondition(&engine, "initial", &state));
    if (test_case->is_mineral)
    {
      ONNX_TEST_REQUIRE(&num_failures, state.mineral_volume_fraction.data[2] == 0.05);
      ONNX_TEST_REQUIRE(&num_failures, state.mineral_specific_surface_area.data[2] == 100.0);
    }
    else
    {
      ONNX_TEST_REQUIRE(&num_failures, state.total_mobile.data[2] == 0.001);
      ONNX_TEST_REQUIRE(&num_failures, state.total_immobile.data[2] == 0.2);
    }

    /* The add graph exposes exactly what the two model inputs received.
     * With porosity = saturation = 1, immobile input is 0.2 / 1000 mol/L
     * water and total input is 0.001 + 0.2 / 1000 mol/L water. */
    ONNX_TEST_REQUIRE(&num_failures, OnnxRunInference(&engine, &state));
    ONNX_TEST_REQUIRE(&num_failures, OnnxCloseEnough(
        state.temperature, test_case->expected_sum, 1.0e-12));

    /* Partial initialization must leave the second field untouched. */
    ONNX_TEST_REQUIRE(&num_failures, OnnxApplyNamedCondition(&engine, "partial", &state));
    if (test_case->is_mineral)
    {
      ONNX_TEST_REQUIRE(&num_failures, state.mineral_volume_fraction.data[2] == -0.002);
      ONNX_TEST_REQUIRE(&num_failures, state.mineral_specific_surface_area.data[2] == 100.0);
    }
    else
    {
      ONNX_TEST_REQUIRE(&num_failures, state.total_mobile.data[2] == -0.002);
      ONNX_TEST_REQUIRE(&num_failures, state.total_immobile.data[2] == 0.2);
    }

    /* A generic driver value has no field label, so neither field may change. */
    ONNX_TEST_EXPECT(&num_failures, __func__, "Shutdown failed",
        OnnxShutdownEngine(&engine), NULL);

    /* .cfg provides the data */
    ONNX_TEST_REQUIRE(&num_failures, OnnxSetupEngineAtPath(ONNX_TEST_TEMP_CONFIG, false, &engine));
    AllocateAlquimiaGeochemicalCondition(8, 1, 0, &condition);
    OnnxInitializeConstraint(&condition, 0, test_case->feature, 123.0);
    engine.interface.ProcessCondition(
        &engine.engine_state, &condition, NULL, &state, NULL, &engine.status);
    ONNX_TEST_REQUIRE(&num_failures, engine.status.error == kAlquimiaErrorEngineIntegrity);
    ONNX_TEST_REQUIRE(&num_failures,
        strstr(engine.status.message, "requires explicit state fields") != NULL);
    if (test_case->is_mineral)
    {
      ONNX_TEST_REQUIRE(&num_failures, state.mineral_volume_fraction.data[2] == -0.002);
      ONNX_TEST_REQUIRE(&num_failures, state.mineral_specific_surface_area.data[2] == 100.0);
    }
    else
    {
      ONNX_TEST_REQUIRE(&num_failures, state.total_mobile.data[2] == -0.002);
      ONNX_TEST_REQUIRE(&num_failures, state.total_immobile.data[2] == 0.2);
    }

cleanup:
    FreeAlquimiaGeochemicalCondition(&condition);
    FreeAlquimiaState(&state);
    FreeAlquimiaProblemMetaData(&metadata);
    ONNX_TEST_EXPECT(&num_failures, __func__, "Shutdown failed",
        OnnxShutdownEngine(&engine), NULL);
    if (config_written)
    {
      ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxRemoveTemporaryConfig failed",
          OnnxRemoveTemporaryConfig(test_case->name), NULL);
    }
  }
}

/**
 * @brief Verifies condition-only fields are allocated and initialized correctly.
 * When a geochemical condition provides paired fields (e.g., both mobile and 
 * immobile concentrations), but the ONNX model signature only requires one 
 * of them, the engine must still allocate and initialize the unused field in 
 * the Alquimia state to ensure state integrity.
 *
 * | ONNX Model Requires | Condition JSON Provides | Engine Must Allocate & Set |
 * | total_mobile | total_mobile + total_immobile | total_immobile |
 * | total_immobile | total_immobile + total_mobile | total_mobile (plus primary metadata) |
 * | mineral_volume_fraction | fraction + specific_surface_area | mineral_specific_surface_area |
 *
 * | M05 | Unrelated features defined in the condition (e.g., an entirely "unused" block) must be safely ignored without throwing errors. |
 */
static void TestM05ConditionOnlyFields(void)
{
  typedef struct {
    const char *name;
    const char *input_field;
    bool is_mineral;
  } ConditionOnlyCase;

  const ConditionOnlyCase cases[] = {
      {"M05 mobile input, condition adds immobile", "total_mobile", false},
      {"M05 immobile input, condition adds mobile", "total_immobile", false},
      {"M05 mineral fraction input, condition adds area",
       "mineral_volume_fraction", true}
  };

  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
  {
    const ConditionOnlyCase *test_case = &cases[i];
    char json[4096];
    const char *fields = test_case->is_mineral
        ? "        \"mineral_volume_fraction\": 0.05,\n"
          "        \"mineral_specific_surface_area\": 100.0\n"
        : "        \"total_mobile\": 0.001,\n"
          "        \"total_immobile\": 0.2\n";
    bool config_written = false;
    OnnxTestEngine engine = {0};
    AlquimiaState state = {0};
    AlquimiaProblemMetaData metadata = {0};

    printf("  %s\n", test_case->name);

    /* Only one field is mapped, but the condition explicitly supplies both. */
    int length = snprintf(json, sizeof(json),
        "{\n"
        "  \"schema_version\": 1,\n"
        "  \"model\": \"%s/onnx_test_cases/deterministic/identity_double.onnx\",\n"
        "  \"inputs\": [{\n"
        "    \"tensor\": \"input\", \"tensor_element_index\": 0,\n"
        "    \"feature\": \"feature\",\n"
        "    \"alquimia_state\": \"%s\", \"alquimia_state_index\": 2\n"
        "  }],\n"
        "  \"outputs\": [{\n"
        "    \"tensor\": \"output\", \"tensor_element_index\": 0,\n"
        "    \"feature\": \"temperature\",\n"
        "    \"alquimia_state\": \"temperature\", \"alquimia_state_index\": 0\n"
        "  }],\n"
        "  \"conditions\": {\n"
        "    \"initial\": {\n"
        "      \"feature\": {\n%s      },\n"
        "      \"unused\": {\"total_mobile\": 42}\n"
        "    }\n"
        "  }\n"
        "}\n",
        CMAKE_CURRENT_SOURCE_DIR, test_case->input_field, fields);
    ONNX_TEST_REQUIRE(&num_failures, length >= 0 && (size_t)length < sizeof(json));
    ONNX_TEST_REQUIRE(&num_failures, OnnxWriteTemporaryConfig(json));
    config_written = true;
    ONNX_TEST_REQUIRE(&num_failures, OnnxSetupEngineAtPath(ONNX_TEST_TEMP_CONFIG, true, &engine));

    /* Condition-only storage still needs the same component/mineral index. */
    if (test_case->is_mineral)
    {
      ONNX_TEST_REQUIRE(&num_failures, engine.sizes.num_minerals == 3);
    }
    else
    {
      ONNX_TEST_REQUIRE(&num_failures, engine.sizes.num_primary == 3);
      ONNX_TEST_REQUIRE(&num_failures, engine.sizes.num_sorbed == 3);
    }
    AllocateAlquimiaVectorString(engine.sizes.num_primary, &metadata.primary_names);
    AllocateAlquimiaVectorString(engine.sizes.num_minerals, &metadata.mineral_names);
    engine.interface.GetProblemMetaData(&engine.engine_state, &metadata, &engine.status);
    ONNX_TEST_REQUIRE(&num_failures, engine.status.error == kAlquimiaNoError);
    const char *metadata_name = test_case->is_mineral
        ? metadata.mineral_names.data[2] : metadata.primary_names.data[2];
    ONNX_TEST_REQUIRE(&num_failures, strcmp(metadata_name, "feature") == 0);
    if (test_case->is_mineral)
    {
      AllocateAlquimiaVectorDouble(3, &state.mineral_volume_fraction);
      AllocateAlquimiaVectorDouble(3, &state.mineral_specific_surface_area);
    }
    else
    {
      OnnxAllocateState(&engine, &state);
    }
    /* ProcessCondition */
    ONNX_TEST_REQUIRE(&num_failures, OnnxApplyNamedCondition(&engine, "initial", &state));
    if (test_case->is_mineral)
    {
      ONNX_TEST_REQUIRE(&num_failures, state.mineral_volume_fraction.data[2] == 0.05);
      ONNX_TEST_REQUIRE(&num_failures, state.mineral_specific_surface_area.data[2] == 100.0);
    }
    else
    {
      ONNX_TEST_REQUIRE(&num_failures, state.total_mobile.data[2] == 0.001);
      ONNX_TEST_REQUIRE(&num_failures, state.total_immobile.data[2] == 0.2);
    }

cleanup:
    FreeAlquimiaState(&state);
    FreeAlquimiaProblemMetaData(&metadata);
    ONNX_TEST_EXPECT(&num_failures, __func__, "Shutdown failed",
        OnnxShutdownEngine(&engine), NULL);
    if (config_written)
    {
      ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxRemoveTemporaryConfig failed",
          OnnxRemoveTemporaryConfig(test_case->name), NULL);
    }
  }
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
  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigParseFailure failed",
      OnnxExpectConfigParseFailure(
      "E01 conditions must be an object",
      "{\"schema_version\":1,\"model\":\"model.onnx\","
      "\"conditions\":[],\"inputs\":[],\"outputs\":[]}",
      "conditions must be an object"), NULL);

  /* Duplicate top-level conditions property. */
  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigParseFailure failed",
      OnnxExpectConfigParseFailure(
      "E01 duplicate conditions property",
      "{\"schema_version\":1,\"model\":\"model.onnx\","
      "\"conditions\":{},\"conditions\":{},\"inputs\":[],\"outputs\":[]}",
      "Duplicate property 'conditions'"), NULL);

  /* Duplicate condition name. */
  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigParseFailure failed",
      OnnxExpectConfigParseFailure(
      "E01 duplicate condition name",
      "{\"schema_version\":1,\"model\":\"model.onnx\","
      "\"conditions\":{\"initial\":{},\"initial\":{}},"
      "\"inputs\":[],\"outputs\":[]}",
      "Duplicate name 'initial'"), NULL);

  /* Empty condition name. */
  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigParseFailure failed",
      OnnxExpectConfigParseFailure(
      "E01 empty condition name",
      "{\"schema_version\":1,\"model\":\"model.onnx\","
      "\"conditions\":{\"\":{}},\"inputs\":[],\"outputs\":[]}",
      "must be nonempty"), NULL);

  /* Condition bodies must be objects. */
  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigParseFailure failed",
      OnnxExpectConfigParseFailure(
      "E01 condition must be an object",
      "{\"schema_version\":1,\"model\":\"model.onnx\","
      "\"conditions\":{\"initial\":1},\"inputs\":[],\"outputs\":[]}",
      "Condition 'initial' must be an object"), NULL);

  /* Duplicate condition feature. */
  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigParseFailure failed",
      OnnxExpectConfigParseFailure(
      "E01 duplicate condition feature",
      "{\"schema_version\":1,\"model\":\"model.onnx\","
      "\"conditions\":{\"initial\":{\"f\":1,\"f\":2}},"
      "\"inputs\":[],\"outputs\":[]}",
      "Duplicate name 'f'"), NULL);

  /* Empty condition feature. */
  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigParseFailure failed",
      OnnxExpectConfigParseFailure(
      "E01 empty condition feature",
      "{\"schema_version\":1,\"model\":\"model.onnx\","
      "\"conditions\":{\"initial\":{\"\":1}},"
      "\"inputs\":[],\"outputs\":[]}",
      "must be nonempty"), NULL);

  /* Condition values must be numeric. */
  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigParseFailure failed",
      OnnxExpectConfigParseFailure(
      "E01 condition feature must be numeric",
      "{\"schema_version\":1,\"model\":\"model.onnx\","
      "\"conditions\":{\"initial\":{\"f\":\"1\"}},"
      "\"inputs\":[],\"outputs\":[]}",
      "must be a finite number"), NULL);

  /* Condition values must be finite. */
  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigParseFailure failed",
      OnnxExpectConfigParseFailure(
      "E01 condition feature must be finite",
      "{\"schema_version\":1,\"model\":\"model.onnx\","
      "\"conditions\":{\"initial\":{\"f\":1e999}},"
      "\"inputs\":[],\"outputs\":[]}",
      "must be a finite number"), NULL);

  /* Conditions must cover every configured input feature. */
  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigParseFailure failed",
      OnnxExpectConfigParseFailure(
      "E01 condition must cover every input feature",
      "{\"schema_version\":1,\"model\":\"model.onnx\","
      "\"conditions\":{\"initial\":{\"H\":1e-5}},"
      "\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "," ONNX_TEST_VALID_INPUT_1 "],"
      "\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}",
      "Condition 'initial' is missing input feature 'Zn'"), NULL);

  /* Keep malformed JSON literal: duplicate keys and overflow are intentional. */
  const struct {
    const char *name;
    const char *fields_json;
    const char *expected_message;
  } nested_cases[] = {
      {"E01 empty nested fields",
       "{}", "has no fields"},
      {"E01 duplicate nested field",
       "{\"total_mobile\": 1, \"total_mobile\": 2}", "Duplicate name 'total_mobile'"},
      {"E01 empty nested field name",
       "{\"\": 1}", "must be nonempty"},
      {"E01 non-finite nested value",
       "{\"total_mobile\": 1e999}", "must be a finite number"},
      {"E01 boolean nested value",
       "{\"total_mobile\": true}", "must be a finite number"},
      {"E01 object instead of numeric value",
       "{\"total_mobile\": {\"value\": 1}}", "must be a finite number"}
  };

  for (size_t i = 0; i < sizeof(nested_cases) / sizeof(nested_cases[0]); ++i)
  {
    char json[1024];
    printf("  %s\n", nested_cases[i].name);
    int length = snprintf(json, sizeof(json),
        "{\n"
        "  \"schema_version\": 1,\n"
        "  \"model\": \"model.onnx\",\n"
        "  \"inputs\": [],\n"
        "  \"outputs\": [],\n"
        "  \"conditions\": {\n"
        "    \"initial\": {\"Zn\": %s}\n"
        "  }\n"
        "}\n", nested_cases[i].fields_json);
    ONNX_TEST_REQUIRE(&num_failures, length >= 0 && (size_t)length < sizeof(json));
    ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigParseFailure failed",
        OnnxExpectConfigParseFailure(
        nested_cases[i].name, json, nested_cases[i].expected_message), NULL);
cleanup:
    continue;
  }
}

/**
 * @brief Verifies strict ONNX JSON mapping-contract failures.
 */
static void TestE02ConfigContractFailures(void)
{
  AlquimiaEngineStatus status = {0};
  AlquimiaInterface interface;

  printf("Running strict ONNX config contract cases.\n");
  ONNX_TEST_REQUIRE(&num_failures, OnnxCreateInterface(&interface, &status));

  /* | E02 | Schema version is missing, malformed, or unsupported | Setup error naming `schema_version` | */
  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status, "E02 missing schema version",
      "{\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[],\"outputs\":[]}", "schema_version"), NULL);

  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status, "E02 malformed schema version",
      "{\"schema_version\":\"1\",\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[],\"outputs\":[]}", "schema_version"), NULL);

  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status, "E02 unsupported schema version",
      "{\"schema_version\":2,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[],\"outputs\":[]}", "schema_version"), NULL);

  /* | E03 | Required top-level object or array is missing | Setup error naming the property | */
  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status, "E03 missing model",
      "{\"schema_version\":1,\"inputs\":[],\"outputs\":[]}", "model"), NULL);

  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status, "E03 missing inputs",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"outputs\":[]}", "inputs and outputs"), NULL);

  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status, "E03 missing outputs",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[]}", "inputs and outputs"), NULL);

  /* | E04 | Input mapping property is missing | Setup error naming the property | */
  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status, "E04 missing input tensor",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[{\"tensor_element_index\":0,\"feature\":\"f\","
      "\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0}],"
      "\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}", "tensor"), NULL);

  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status,
      "E04 missing input tensor element index",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\","
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0}],"
      "\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}", "tensor_element_index"), NULL);

  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status, "E04 missing input feature",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0,"
      "\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0}],"
      "\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}", "feature"), NULL);

  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status,
      "E04 missing input Alquimia state variable",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"f\",\"alquimia_state_index\":0}],"
      "\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}", "alquimia_state"), NULL);

  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status,
      "E04 missing input Alquimia state index",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\"}],"
      "\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}", "alquimia_state_index"), NULL);

  /* | E05 | Output mapping property is missing | Setup error naming the property | */
  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status, "E05 missing output tensor",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "," ONNX_TEST_VALID_INPUT_1
      "],\"outputs\":[{\"tensor_element_index\":0,\"feature\":\"H\","
      "\"alquimia_state\":\"total_mobile\","
      "\"alquimia_state_index\":0}]}", "tensor"), NULL);

  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status,
      "E05 missing output tensor element index",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "," ONNX_TEST_VALID_INPUT_1
      "],\"outputs\":[{\"tensor\":\"sorbed_output_raw\","
      "\"feature\":\"H\",\"alquimia_state\":\"total_mobile\","
      "\"alquimia_state_index\":0}]}", "tensor_element_index"), NULL);

  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status, "E05 missing output feature",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "," ONNX_TEST_VALID_INPUT_1
      "],\"outputs\":[{\"tensor\":\"sorbed_output_raw\","
      "\"tensor_element_index\":0,\"alquimia_state\":\"total_mobile\","
      "\"alquimia_state_index\":0}]}", "feature"), NULL);

  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status,
      "E05 missing output Alquimia state variable",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "," ONNX_TEST_VALID_INPUT_1
      "],\"outputs\":[{\"tensor\":\"sorbed_output_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"H\",\"alquimia_state_index\":0}]}",
      "alquimia_state"), NULL);

  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status,
      "E05 missing output Alquimia state index",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "," ONNX_TEST_VALID_INPUT_1
      "],\"outputs\":[{\"tensor\":\"sorbed_output_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"H\",\"alquimia_state\":\"total_mobile\"}]}",
      "alquimia_state_index"), NULL);

  /* | E06 | Unknown or duplicate JSON property | Setup error | */
  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status, "E06 unknown property",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[],\"outputs\":[],\"models\":\"extra\"}",
      "Unknown property 'models'"), NULL);

  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status, "E06 duplicate property",
      "{\"schema_version\":1,\"schema_version\":1,\"model\":\""
      ONNX_TEST_EX8_MODEL_PATH "\",\"inputs\":[],\"outputs\":[]}",
      "Duplicate property 'schema_version'"), NULL);

  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status, "E06 unknown input property",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0,"
      "\"unit\":\"molar\"}],\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}",
      "Unknown property 'unit'"), NULL);

  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status, "E06 duplicate output property",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "," ONNX_TEST_VALID_INPUT_1
      "],\"outputs\":[{\"tensor\":\"sorbed_output_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"H\",\"alquimia_state\":\"total_mobile\","
      "\"alquimia_state_index\":0,\"alquimia_state_index\":0}]}",
      "Duplicate property 'alquimia_state_index'"), NULL);

  /* | E07 | `alquimia_state` is unsupported | Setup error | */
  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status,
      "E07 unsupported Alquimia state variable",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"H\",\"alquimia_state\":\"total_mobiles\","
      "\"alquimia_state_index\":0}," ONNX_TEST_VALID_INPUT_1 "],\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}",
      "Unsupported AlquimiaState variable 'total_mobiles'"), NULL);

      ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
          OnnxExpectConfigSetupFailure(&interface, &status,
      "E07 total concentration cannot be an output",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "," ONNX_TEST_VALID_INPUT_1
      "],\"outputs\":[{\"tensor\":\"sorbed_output_raw\","
      "\"tensor_element_index\":0,\"feature\":\"H\","
      "\"alquimia_state\":\"total_molar\",\"alquimia_state_index\":0},"
      ONNX_TEST_VALID_OUTPUT_1 "]}", "input-only mapping"), NULL);

  /* | E08 | `alquimia_state_index` or `tensor_element_index` is invalid | Setup error | */
  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status,
      "E08 negative tensor element index",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":-1,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0}],"
      "\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}", "tensor_element_index"), NULL);

  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status,
      "E08 fractional Alquimia state index",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0.5}],"
      "\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}", "alquimia_state_index"), NULL);

  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status,
      "E08 negative Alquimia state index",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":-1}],"
      "\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}", "alquimia_state_index"), NULL);

  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status,
      "E08 fractional tensor element index",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0.5,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0}],"
      "\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}", "tensor_element_index"), NULL);

  /* tensor_element_index = 2147483648 > INT_MAX, in the onnx_alquimia_config.c */
  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status,
      "E08 tensor element index exceeds C int range",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\","
      "\"tensor_element_index\":2147483648,\"feature\":\"f\","
      "\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0}],"
      "\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}", "tensor_element_index"), NULL);

  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status,
      "E08 Alquimia state index exceeds C int range",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\","
      "\"alquimia_state_index\":2147483648}],\"outputs\":[" ONNX_TEST_VALID_OUTPUT
      "]}", "alquimia_state_index"), NULL);

  /* | E09 | Scalar mapping uses a nonzero `alquimia_state_index` | Setup error | */
  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status,
      "E09 scalar has nonzero Alquimia state index",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"temperature\",\"alquimia_state\":\"temperature\","
      "\"alquimia_state_index\":1}," ONNX_TEST_VALID_INPUT_1 "],\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}",
      "incompatible with variable 'temperature'"), NULL);

  /* | E10 | Tensor name is unknown | Setup error naming the tensor | */
  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status, "E10 unknown tensor",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"invalid_input\",\"tensor_element_index\":0,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0},"
      ONNX_TEST_VALID_INPUT_1 "],\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}",
      "unknown tensor 'invalid_input'"), NULL);

  /* | E11 | `tensor_element_index` exceeds its flattened extent | Setup error | */
  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status,
      "E11 tensor element index exceeds tensor extent",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":2,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0},"
      ONNX_TEST_VALID_INPUT_1 "],\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}", "out of range"), NULL);

  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status, "E11 output exceeds tensor extent",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "," ONNX_TEST_VALID_INPUT_1
      "],\"outputs\":[{\"tensor\":\"sorbed_output_raw\",\"tensor_element_index\":2,"
      "\"feature\":\"H\",\"alquimia_state\":\"total_mobile\","
      "\"alquimia_state_index\":0}]}", "out of range"), NULL);

  /* | E12 | A tensor element index is mapped more than once | Setup error | */
  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status,
      "E12 duplicate tensor element index",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "," ONNX_TEST_VALID_INPUT_0
      "],\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}", "Duplicate ONNX input mapping"), NULL);

  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status,
      "E12 duplicate output tensor element index",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "," ONNX_TEST_VALID_INPUT_1
      "],\"outputs\":[" ONNX_TEST_VALID_OUTPUT "," ONNX_TEST_VALID_OUTPUT "]}",
      "Duplicate ONNX output mapping"), NULL);

  /* | E13 | A required model tensor element index is unmapped | Setup error | */
  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status,
      "E13 tensor element index is unmapped",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "],\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}",
      "does not map every input tensor element"), NULL);

  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status,
      "E13 output tensor element index is unmapped",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "," ONNX_TEST_VALID_INPUT_1
      "],\"outputs\":[]}", "does not map every output tensor element"), NULL);

  /* | E14 | Mappings derive unsafe or overflowing Alquimia sizes | Setup error | */
  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status, "E14 derived size overflows int",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\","
      "\"alquimia_state_index\":2147483647}," ONNX_TEST_VALID_INPUT_1 "],"
      "\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}", "incompatible with variable"), NULL);

  /* | E15 | Two names target the same metadata destination | Setup rejects the ambiguity | */
  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status, "E15 conflicting feature names",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"first_name\",\"alquimia_state\":\"total_mobile\","
      "\"alquimia_state_index\":0},{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":1,"
      "\"feature\":\"second_name\",\"alquimia_state\":\"total_mobile\","
      "\"alquimia_state_index\":0}],\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}",
      "Conflicting ONNX feature names"), NULL);

  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status,
      "E15 conflicting total and immobile feature names",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"different_name\",\"alquimia_state\":\"total_molar\","
      "\"alquimia_state_index\":0}," ONNX_TEST_VALID_INPUT_1 "],"
      "\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}", "Conflicting ONNX feature names"), NULL);

  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status,
      "E15 conflicting input and output feature names",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "," ONNX_TEST_VALID_INPUT_1
      "],\"outputs\":[{\"tensor\":\"sorbed_output_raw\","
      "\"tensor_element_index\":0,\"feature\":\"different_name\","
      "\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0},"
      ONNX_TEST_VALID_OUTPUT_1 "]}",
      "Conflicting ONNX feature names"), NULL);

  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status,
      "E15 conflicting mobile and immobile feature names",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[" ONNX_TEST_VALID_INPUT_0 "," ONNX_TEST_VALID_INPUT_1
      "],\"outputs\":[{\"tensor\":\"sorbed_output_raw\","
      "\"tensor_element_index\":0,\"feature\":\"different_name\","
      "\"alquimia_state\":\"total_immobile\",\"alquimia_state_index\":0},"
      ONNX_TEST_VALID_OUTPUT_1 "]}",
      "Conflicting ONNX feature names"), NULL);

  /* | E16 | Similar but invalid property has trailing text | Property is rejected rather than partially matched | */
  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status, "E16 property trailing text",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0,"
      "\"alquimia_state_index_extra\":0}],\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}",
      "Unknown property 'alquimia_state_index_extra'"), NULL);

  /* | E17 | Two input mappings use the same feature name | Setup rejects the duplicate lookup key | */
  ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
      OnnxExpectConfigSetupFailure(&interface, &status,
      "E17 duplicate input feature name",
      "{\"schema_version\":1,\"model\":\"" ONNX_TEST_EX8_MODEL_PATH
      "\",\"inputs\":[{\"tensor\":\"chemical_input_raw\",\"tensor_element_index\":0,"
      "\"feature\":\"duplicate_feature\",\"alquimia_state\":\"total_mobile\","
      "\"alquimia_state_index\":0},{\"tensor\":\"chemical_input_raw\","
      "\"tensor_element_index\":1,\"feature\":\"duplicate_feature\","
      "\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":1}],"
      "\"outputs\":[" ONNX_TEST_VALID_OUTPUT "]}",
      "Duplicate ONNX input feature name 'duplicate_feature'."), NULL);

cleanup:
  FreeAlquimiaEngineStatus(&status);
}

/**
 * @brief Rejects invalid condition fields and shared-name input mappings.
 *
 * Cases cover unknown fields, derived totals, incompatible categories, scalar
 * ambiguity, repeated input fields, and reuse across categories or indices.
 * Each record keeps the invalid condition beside its expected diagnostic.
 */
static void TestE18ConditionFieldFailures(void)
{
  typedef struct {
    const char *name;
    const char *left_field;
    const char *right_field;
    int right_index;
    const char *condition_json;
    const char *expected_message;
  } ConditionFailureCase;

  const ConditionFailureCase cases[] = {
      {
        .name = "E18 unknown condition field",
        .left_field = "total_mobile",
        .right_field = "total_immobile",
        .right_index = 0,
        .condition_json = "{\"total_mobile_typo\": 1}",
        .expected_message = "Unsupported AlquimiaState variable"
      },
      {
        .name = "E18 explicit assignment to derived total",
        .left_field = "total_mobile",
        .right_field = "total_immobile",
        .right_index = 0,
        .condition_json = "{\"total_molar\": 1}",
        .expected_message = "total_molar is derived"
      },
      {
        .name = "E18 mineral field assigned to a component",
        .left_field = "total_mobile",
        .right_field = "total_immobile",
        .right_index = 0,
        .condition_json = "{\"mineral_volume_fraction\": 1}",
        .expected_message = "incompatible with feature"
      },
      {
        .name = "E18 scalar condition for paired inputs",
        .left_field = "total_mobile",
        .right_field = "total_immobile",
        .right_index = 0,
        .condition_json = "1",
        .expected_message = "requires explicit state fields"
      },
      {
        .name = "E18 scalar condition for a total input",
        .left_field = "total_molar",
        .right_field = "total_mobile",
        .right_index = 0,
        .condition_json = "1",
        .expected_message = "requires explicit state fields"
      },
      {
        .name = "E18 repeated input field",
        .left_field = "total_mobile",
        .right_field = "total_mobile",
        .right_index = 0,
        .condition_json = "{\"total_mobile\": 1}",
        .expected_message = "Duplicate ONNX input feature name"
      },
      {
        .name = "E18 shared input name across categories",
        .left_field = "total_mobile",
        .right_field = "mineral_volume_fraction",
        .right_index = 0,
        .condition_json = "{\"total_mobile\": 1}",
        .expected_message = "Duplicate ONNX input feature name"
      },
      {
        .name = "E18 shared input name across indices",
        .left_field = "total_mobile",
        .right_field = "total_immobile",
        .right_index = 1,
        .condition_json = "{\"total_mobile\": 1}",
        .expected_message = "Duplicate ONNX input feature name"
      }
  };
  AlquimiaInterface interface;
  AlquimiaEngineStatus status = {0};

  ONNX_TEST_REQUIRE(&num_failures, OnnxCreateInterface(&interface, &status));
  for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
  {
    const ConditionFailureCase *test_case = &cases[i];
    char json[4096];

    printf("  %s\n", test_case->name);
    int length = snprintf(json, sizeof(json),
        "{\n"
        "  \"schema_version\": 1,\n"
        "  \"model\": \"%s/onnx_test_cases/deterministic/add_two_inputs.onnx\",\n"
        "  \"inputs\": [\n"
        "    {\n"
        "      \"tensor\": \"left\", \"tensor_element_index\": 0,\n"
        "      \"feature\": \"Zn\",\n"
        "      \"alquimia_state\": \"%s\", \"alquimia_state_index\": 0\n"
        "    },\n"
        "    {\n"
        "      \"tensor\": \"right\", \"tensor_element_index\": 0,\n"
        "      \"feature\": \"Zn\",\n"
        "      \"alquimia_state\": \"%s\", \"alquimia_state_index\": %d\n"
        "    }\n"
        "  ],\n"
        "  \"outputs\": [{\n"
        "    \"tensor\": \"sum\", \"tensor_element_index\": 0,\n"
        "    \"feature\": \"temperature\",\n"
        "    \"alquimia_state\": \"temperature\", \"alquimia_state_index\": 0\n"
        "  }],\n"
        "  \"conditions\": {\n"
        "    \"initial\": {\"Zn\": %s}\n"
        "  }\n"
        "}\n",
        CMAKE_CURRENT_SOURCE_DIR,
        test_case->left_field, test_case->right_field, test_case->right_index,
        test_case->condition_json);
    ONNX_TEST_REQUIRE(&num_failures, length >= 0 && (size_t)length < sizeof(json));
    ONNX_TEST_EXPECT(&num_failures, __func__, "OnnxExpectConfigSetupFailure failed",
        OnnxExpectConfigSetupFailure(
        &interface, &status, test_case->name, json, test_case->expected_message), NULL);
  }
cleanup:
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
  TestM04PairedInputFields();
  TestM05ConditionOnlyFields();
  TestE01ConditionConfigFailures();
  TestE02ConfigContractFailures();
  TestE18ConditionFieldFailures();
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
  return num_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
#else
  (void)argc;
  (void)argv;
  printf("ONNX not enabled. Skipping ONNX engine unit test.\n");
#endif
  return EXIT_SUCCESS;
}
