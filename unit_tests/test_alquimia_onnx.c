/* -*-  mode: c; c-default-style: "google"; indent-tabs-mode: nil -*- 
** This file is used to test the JSON Mapping contract
** The related test case is below:
** | ID | Scenario | Expected result |
** |---|---|---|
** | M01 | Schema version is missing, malformed, or unsupported | Setup error naming `schema_version` |
** | M02 | Required top-level object or array is missing | Setup error naming the property |
** | M03 | Input mapping property is missing | Setup error naming the property |
** | M04 | Output mapping property is missing | Setup error naming the property |
** | M05 | Unknown or duplicate JSON property | Setup error |
** | M06 | `alquimia_state` is unsupported | Setup error |
** | M07 | `alquimia_state_index` or `tensor_element_index` is invalid | Setup error |
** | M08 | Scalar mapping uses a nonzero `alquimia_state_index` | Setup error |
** | M09 | Tensor name is unknown | Setup error naming the tensor |
** | M10 | `tensor_element_index` exceeds its flattened extent | Setup error |
** | M11 | A tensor element index is mapped more than once | Setup error |
** | M12 | A required model tensor element index is unmapped | Setup error |
** | M13 | Mappings derive unsafe or overflowing Alquimia sizes | Setup error |
** | M14 | Feature names target different state categories | Correct problem-metadata vectors are populated |
** | M15 | Two names target the same metadata destination | Setup rejects the ambiguity |
** | M16 | Similar but invalid property has trailing text | Property is rejected rather than partially matched |
** | M17 | Model path is relative to the manifest | Correct path is resolved from the manifest directory |
** | M18 | Two input mappings use the same feature name | Setup rejects the duplicate lookup key |
**
** Here we use the LSURF model for testing
** This model has 1 input/output tensor
** input tensor: 2 inputs
** output tensor: 1 output
** input tensor name: "double_input_2"
** output tensor name: "double_output_2"
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "alquimia/alquimia_constants.h"
#include "alquimia/alquimia_interface.h"
#include "alquimia/alquimia_memory.h"
#include "alquimia/alquimia_util.h"

#if ALQUIMIA_HAVE_ONNX

/* CMAKE_CURRENT_SOURCE_DIR "/../model/lsurf_model_2_float_64.onnx"  
** Assume CMAKE_CURRENT_SOURCE_DIR = /home/user/project/alquimia
** "/home/user/project/models/lsurf_model_2_float_64.onnx"
*/
#define MODEL_2_PATH \
  CMAKE_CURRENT_SOURCE_DIR "/../models/lsurf_model_2_float_64.onnx"
#define MODEL_2_MANIFEST \
  CMAKE_CURRENT_SOURCE_DIR "/../models/lsurf_model_2_test.json"
#define MODEL_2_RELATIVE_MANIFEST                                      \
  CMAKE_CURRENT_SOURCE_DIR                                             \
  "/onnx_fixtures/manifests/model_2_relative.json"
#define TEMP_MANIFEST "test_alquimia_onnx_manifest_case.json"

// Match the first input of the model
#define VALID_INPUT_0                                                  \
  "{\"tensor\":\"double_input_2\",\"tensor_element_index\":0,"        \
  "\"feature\":\"uranium_total\",\"alquimia_state\":\"total_mobile\"," \
  "\"alquimia_state_index\":0}"

// Match the second input of the model
#define VALID_INPUT_1                                                  \
  "{\"tensor\":\"double_input_2\",\"tensor_element_index\":1,"        \
  "\"feature\":\"U_species14\",\"alquimia_state\":\"total_mobile\","   \
  "\"alquimia_state_index\":1}"

// Match the only output of the model
#define VALID_OUTPUT                                                   \
  "{\"tensor\":\"double_output_2\",\"tensor_element_index\":0,"       \
  "\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0}"

static void CreateOnnxInterface(
    AlquimiaInterface *interface,
    AlquimiaEngineStatus *status)
{
  CreateAlquimiaInterface("ONNX", interface, status);
  ALQUIMIA_ASSERT(status->error == kAlquimiaNoError);
}
/* Write the contents: manifest to the temporary file 
** manifest: JSON 
** If used, needed to remove() manually
*/
static void WriteTemporaryManifest(const char *contents)
{
  /* TEMP_MANIFEST = "test_alquimia_onnx_manifest_case.json" */
  FILE *file = fopen(TEMP_MANIFEST, "wb");
  size_t length = strlen(contents);

  ALQUIMIA_ASSERT(file != NULL);
  ALQUIMIA_ASSERT(fwrite(contents, 1, length, file) == length);
  ALQUIMIA_ASSERT(fclose(file) == 0);
}
/* Expected: fail */
static void ExpectSetupFailure(
    AlquimiaInterface *interface,
    AlquimiaEngineStatus *status,
    const char *test_id,
    const char *manifest,
    const char *expected_message)
{
  AlquimiaEngineFunctionality functionality = {0};
  AlquimiaSizes sizes = {0};
  void *engine_state = NULL;

  printf("  %s\n", test_id);
  /* Write the contents: manifest to the temporary file */
  /* manifest: JSON 
  */
  WriteTemporaryManifest(manifest);
  /* Setup by the temporary JSON file */
  interface->Setup(TEMP_MANIFEST, false, &engine_state, &sizes,
                   &functionality, status);
  if (status->error == kAlquimiaNoError || engine_state != NULL ||
    /* expected_message is the substring of the status->message */
      strstr(status->message, expected_message) == NULL)
  {
    fprintf(stderr,
            "%s returned error %d and message '%s' (expected '%s').\n",
            test_id, status->error, status->message, expected_message);
  }
  ALQUIMIA_ASSERT(status->error == kAlquimiaErrorEngineIntegrity);
  ALQUIMIA_ASSERT(engine_state == NULL);
  ALQUIMIA_ASSERT(strstr(status->message, expected_message) != NULL);
  ALQUIMIA_ASSERT(remove(TEMP_MANIFEST) == 0);
}

/* Test the whole lifecycle */
static void TestModel2Lifecycle(void)
{
  static const char *const features[] = {"uranium_total", "U_species14"};
  static const double inputs[] = {
      -6.677780705266080, -37.488999999999997};
  AlquimiaAuxiliaryData aux_data = {0};
  AlquimiaEngineFunctionality functionality = {0};
  AlquimiaEngineStatus status;
  AlquimiaGeochemicalCondition condition;
  AlquimiaInterface interface;
  AlquimiaProblemMetaData meta_data;
  AlquimiaProperties properties = {0};
  AlquimiaSizes sizes = {0};
  AlquimiaState state = {0};
  void *engine_state = NULL;
  int i;

  printf("Running successful model-2 ONNX lifecycle.\n");
  AllocateAlquimiaEngineStatus(&status);
  CreateOnnxInterface(&interface, &status);

  interface.Setup(MODEL_2_MANIFEST, false, &engine_state, &sizes,
                  &functionality, &status);
  if (status.error != kAlquimiaNoError)
  {
    fprintf(stderr, "Model-2 setup failed: %s\n", status.message);
  }
  ALQUIMIA_ASSERT(status.error == kAlquimiaNoError);
  ALQUIMIA_ASSERT(engine_state != NULL);
  ALQUIMIA_ASSERT(sizes.num_primary == 2);
  ALQUIMIA_ASSERT(sizes.num_sorbed == 0);
  ALQUIMIA_ASSERT(sizes.num_minerals == 0);
  ALQUIMIA_ASSERT(sizes.num_surface_sites == 0);
  ALQUIMIA_ASSERT(sizes.num_ion_exchange_sites == 0);
  ALQUIMIA_ASSERT(sizes.num_gases == 0);
  ALQUIMIA_ASSERT(functionality.operator_splitting);
  ALQUIMIA_ASSERT(!functionality.thread_safe);

  AllocateAlquimiaProblemMetaData(&sizes, &meta_data);
  interface.GetProblemMetaData(&engine_state, &meta_data, &status);
  ALQUIMIA_ASSERT(status.error == kAlquimiaNoError);
  for (i = 0; i < 2; ++i)
  {
    ALQUIMIA_ASSERT(strcmp(meta_data.primary_names.data[i], features[i]) == 0);
  }

  AllocateAlquimiaState(&sizes, &state);
  AllocateAlquimiaGeochemicalCondition(8, 2, 0, &condition);
  strcpy(condition.name, "initial");
  for (i = 0; i < 2; ++i)
  {
    AllocateAlquimiaAqueousConstraint(&condition.aqueous_constraints.data[i]);
    strcpy(condition.aqueous_constraints.data[i].primary_species_name,
           features[i]);
    strcpy(condition.aqueous_constraints.data[i].constraint_type, "total");
    condition.aqueous_constraints.data[i].value = inputs[i];
  }

  interface.ProcessCondition(&engine_state, &condition, &properties, &state,
                             &aux_data, &status);
  ALQUIMIA_ASSERT(status.error == kAlquimiaNoError);
  for (i = 0; i < 2; ++i)
  {
    ALQUIMIA_ASSERT(state.total_mobile.data[i] == inputs[i]);
  }

  interface.ReactionStepOperatorSplit(
      &engine_state, 1.0, &properties, &state, &aux_data, 1, &status);
  if (status.error != kAlquimiaNoError)
  {
    fprintf(stderr, "Model-2 inference failed: %s\n", status.message);
  }
  ALQUIMIA_ASSERT(status.error == kAlquimiaNoError);
  ALQUIMIA_ASSERT(isfinite(state.total_mobile.data[0]));
  ALQUIMIA_ASSERT(state.total_mobile.data[0] == -6.694877268231629);

  FreeAlquimiaGeochemicalCondition(&condition);
  FreeAlquimiaState(&state);
  FreeAlquimiaProblemMetaData(&meta_data);
  interface.Shutdown(&engine_state, &status);
  ALQUIMIA_ASSERT(status.error == kAlquimiaNoError);
  ALQUIMIA_ASSERT(engine_state == NULL);
  FreeAlquimiaEngineStatus(&status);
}

// | M14 | Feature names target different state categories | Correct problem-metadata vectors are populated |
// | M17 | Model path is relative to the manifest | Correct path is resolved from the manifest directory |
static void TestSuccessfulMappingCases(
    AlquimiaInterface *interface,
    AlquimiaEngineStatus *status)
{
  static const char category_manifest[] =
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      "\",\"inputs\":["
      "{\"tensor\":\"double_input_2\",\"tensor_element_index\":0,"
      "\"feature\":\"aqueous_feature\",\"alquimia_state\":\"total_mobile\","
      "\"alquimia_state_index\":0},"
      "{\"tensor\":\"double_input_2\",\"tensor_element_index\":1,"
      "\"feature\":\"gas_feature\",\"alquimia_state\":\"gas_concentration\","
      "\"alquimia_state_index\":0}],\"outputs\":[" VALID_OUTPUT "]}";
  AlquimiaEngineFunctionality functionality = {0};
  AlquimiaProblemMetaData meta_data;
  AlquimiaSizes sizes = {0};
  void *engine_state = NULL;

  printf("  M14 feature names populate separate metadata categories\n");
  /* Write the contents: manifest to the temporary file */
  /* manifest: JSON 
  */
  WriteTemporaryManifest(category_manifest);
  interface->Setup(TEMP_MANIFEST, false, &engine_state, &sizes,
                   &functionality, status);
  // Expected value:                  
  ALQUIMIA_ASSERT(status->error == kAlquimiaNoError);
  ALQUIMIA_ASSERT(engine_state != NULL);
  ALQUIMIA_ASSERT(sizes.num_primary == 1);
  ALQUIMIA_ASSERT(sizes.num_gases == 1);
  AllocateAlquimiaProblemMetaData(&sizes, &meta_data);
  interface->GetProblemMetaData(&engine_state, &meta_data, status);
  ALQUIMIA_ASSERT(status->error == kAlquimiaNoError);
  ALQUIMIA_ASSERT(strcmp(meta_data.primary_names.data[0],
                         "aqueous_feature") == 0);
  ALQUIMIA_ASSERT(strcmp(meta_data.gas_names.data[0], "gas_feature") == 0);
  FreeAlquimiaProblemMetaData(&meta_data);
  interface->Shutdown(&engine_state, status);
  ALQUIMIA_ASSERT(status->error == kAlquimiaNoError);
  ALQUIMIA_ASSERT(engine_state == NULL);
  ALQUIMIA_ASSERT(remove(TEMP_MANIFEST) == 0);

  // Test the ResolvePath in the onnx_alquimia_manifest.c 
  printf("  M17 model path resolves relative to the manifest\n");
  interface->Setup(MODEL_2_RELATIVE_MANIFEST, false, &engine_state, &sizes,
                   &functionality, status);
  if (status->error != kAlquimiaNoError)
  {
    fprintf(stderr, "M17 setup failed: %s\n", status->message);
  }
  ALQUIMIA_ASSERT(status->error == kAlquimiaNoError);
  ALQUIMIA_ASSERT(engine_state != NULL);
  interface->Shutdown(&engine_state, status);
  ALQUIMIA_ASSERT(status->error == kAlquimiaNoError);
  ALQUIMIA_ASSERT(engine_state == NULL);
}

/* Check the JSON mapping contract */
static void TestManifestContract(void)
{
  
  AlquimiaEngineStatus status;
  AlquimiaInterface interface;

  printf("Running strict ONNX manifest contract cases.\n");
  AllocateAlquimiaEngineStatus(&status);
  CreateOnnxInterface(&interface, &status);
  /* MODEL_2_PATH = absolute file .../models/lsurf_model_2_float_64.onnx */
  /* | M01 | Schema version is missing, malformed, or unsupported | Setup error naming `schema_version` | */
  ExpectSetupFailure(&interface, &status, "M01 missing schema version",
    /* Mock JSON file */
      "{\"model\":\"" MODEL_2_PATH
      "\",\"inputs\":[],\"outputs\":[]}", "schema_version");
  ExpectSetupFailure(&interface, &status, "M01 malformed schema version",
      // malformed schema_version: "1" (string)
      "{\"schema_version\":\"1\",\"model\":\"" MODEL_2_PATH
      "\",\"inputs\":[],\"outputs\":[]}", "schema_version");
  ExpectSetupFailure(&interface, &status, "M01 unsupported schema version",
      // Unsupported schema version
      "{\"schema_version\":2,\"model\":\"" MODEL_2_PATH
      "\",\"inputs\":[],\"outputs\":[]}", "schema_version");

  // | M02 | Required top-level object or array is missing | Setup error naming the property |
  ExpectSetupFailure(&interface, &status, "M02 missing model",
      "{\"schema_version\":1,\"inputs\":[],\"outputs\":[]}", "model");
  ExpectSetupFailure(&interface, &status, "M02 missing inputs",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      "\",\"outputs\":[]}", "inputs and outputs");
  ExpectSetupFailure(&interface, &status, "M02 missing outputs",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      "\",\"inputs\":[]}", "inputs and outputs");

  // | M03 | Input mapping property is missing | Setup error naming the property |
  ExpectSetupFailure(&interface, &status, "M03 missing input tensor",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      "\",\"inputs\":[{\"tensor_element_index\":0,\"feature\":\"f\","
      "\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0}],"
      "\"outputs\":[" VALID_OUTPUT "]}", "tensor");
  ExpectSetupFailure(&interface, &status,
      "M03 missing input tensor element index",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      "\",\"inputs\":[{\"tensor\":\"double_input_2\","
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0}],"
      "\"outputs\":[" VALID_OUTPUT "]}", "tensor_element_index");
  ExpectSetupFailure(&interface, &status, "M03 missing input feature",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      "\",\"inputs\":[{\"tensor\":\"double_input_2\",\"tensor_element_index\":0,"
      "\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0}],"
      "\"outputs\":[" VALID_OUTPUT "]}", "feature");
  ExpectSetupFailure(&interface, &status,
      "M03 missing input Alquimia state variable",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      "\",\"inputs\":[{\"tensor\":\"double_input_2\",\"tensor_element_index\":0,"
      "\"feature\":\"f\",\"alquimia_state_index\":0}],"
      "\"outputs\":[" VALID_OUTPUT "]}", "alquimia_state");
  ExpectSetupFailure(&interface, &status,
      "M03 missing input Alquimia state index",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      "\",\"inputs\":[{\"tensor\":\"double_input_2\",\"tensor_element_index\":0,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\"}],"
      "\"outputs\":[" VALID_OUTPUT "]}", "alquimia_state_index");

  // | M04 | Output mapping property is missing | Setup error naming the property |
  ExpectSetupFailure(&interface, &status, "M04 missing output tensor",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      "\",\"inputs\":[" VALID_INPUT_0 "," VALID_INPUT_1
      "],\"outputs\":[{\"tensor_element_index\":0,\"alquimia_state\":\"total_mobile\","
      "\"alquimia_state_index\":0}]}", "tensor");
  ExpectSetupFailure(&interface, &status,
      "M04 missing output tensor element index",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      "\",\"inputs\":[" VALID_INPUT_0 "," VALID_INPUT_1
      "],\"outputs\":[{\"tensor\":\"double_output_2\","
      "\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0}]}", "tensor_element_index");
  ExpectSetupFailure(&interface, &status,
      "M04 missing output Alquimia state variable",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      "\",\"inputs\":[" VALID_INPUT_0 "," VALID_INPUT_1
      "],\"outputs\":[{\"tensor\":\"double_output_2\",\"tensor_element_index\":0,"
      "\"alquimia_state_index\":0}]}", "alquimia_state");
  ExpectSetupFailure(&interface, &status,
      "M04 missing output Alquimia state index",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      "\",\"inputs\":[" VALID_INPUT_0 "," VALID_INPUT_1
      "],\"outputs\":[{\"tensor\":\"double_output_2\",\"tensor_element_index\":0,"
      "\"alquimia_state\":\"total_mobile\"}]}", "alquimia_state_index");

  // | M05 | Unknown or duplicate JSON property | Setup error |
  ExpectSetupFailure(&interface, &status, "M05 unknown property",
      // Unkown property "extra"
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      "\",\"inputs\":[],\"outputs\":[],\"models\":\"extra\"}",
      "Unknown property 'models'");
  ExpectSetupFailure(&interface, &status, "M05 duplicate property",
      // Duplicate "schema_version"
      "{\"schema_version\":1,\"schema_version\":1,\"model\":\""
      MODEL_2_PATH "\",\"inputs\":[],\"outputs\":[]}",
      "Duplicate property 'schema_version'");
  ExpectSetupFailure(&interface, &status, "M05 unknown input property",
      // Unknown "unit"
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      "\",\"inputs\":[{\"tensor\":\"double_input_2\",\"tensor_element_index\":0,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0,"
      "\"unit\":\"molar\"}],\"outputs\":[" VALID_OUTPUT "]}",
      "Unknown property 'unit'");
  ExpectSetupFailure(&interface, &status, "M05 duplicate output property",
      // Duplicate alquimia_state_index
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      "\",\"inputs\":[" VALID_INPUT_0 "," VALID_INPUT_1
      "],\"outputs\":[{\"tensor\":\"double_output_2\",\"tensor_element_index\":0,"
      "\"alquimia_state\":\"total_mobile\","
      "\"alquimia_state_index\":0,\"alquimia_state_index\":0}]}",
      "Duplicate property 'alquimia_state_index'");

  // | M06 | `alquimia_state` is unsupported | Setup error |
  ExpectSetupFailure(&interface, &status,
      "M06 unsupported Alquimia state variable",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      "\",\"inputs\":[{\"tensor\":\"double_input_2\",\"tensor_element_index\":0,"
      // Unsupported "total_mobiles"
      "\"feature\":\"uranium_total\",\"alquimia_state\":\"total_mobiles\","
      "\"alquimia_state_index\":0}," VALID_INPUT_1 "],\"outputs\":[" VALID_OUTPUT "]}",
      "Unsupported AlquimiaState variable 'total_mobiles'");

  // | M07 | `alquimia_state_index` or `tensor_element_index` is invalid | Setup error |
  ExpectSetupFailure(&interface, &status,
      "M07 negative tensor element index",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      // tensor_element_index < 0
      "\",\"inputs\":[{\"tensor\":\"double_input_2\",\"tensor_element_index\":-1,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0}],"
      "\"outputs\":[" VALID_OUTPUT "]}", "tensor_element_index");
  ExpectSetupFailure(&interface, &status,
      "M07 fractional Alquimia state index",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      "\",\"inputs\":[{\"tensor\":\"double_input_2\",\"tensor_element_index\":0,"
      // alquimia_state_index = 0.5
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0.5}],"
      "\"outputs\":[" VALID_OUTPUT "]}", "alquimia_state_index");
  ExpectSetupFailure(&interface, &status,
      "M07 negative Alquimia state index",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      "\",\"inputs\":[{\"tensor\":\"double_input_2\",\"tensor_element_index\":0,"
      // alquimia_state_index < 0
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":-1}],"
      "\"outputs\":[" VALID_OUTPUT "]}", "alquimia_state_index");
  ExpectSetupFailure(&interface, &status,
      "M07 fractional tensor element index",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      // tensor_element_index = 0.5
      "\",\"inputs\":[{\"tensor\":\"double_input_2\",\"tensor_element_index\":0.5,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0}],"
      "\"outputs\":[" VALID_OUTPUT "]}", "tensor_element_index");
  ExpectSetupFailure(&interface, &status,
      "M07 tensor element index exceeds C int range",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      "\",\"inputs\":[{\"tensor\":\"double_input_2\","
      // tensor_element_index = 2147483648 > INT_MAX, in the onnx_alquimia_manifest.c
      "\"tensor_element_index\":2147483648,\"feature\":\"f\","
      "\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0}],"
      "\"outputs\":[" VALID_OUTPUT "]}", "tensor_element_index");
  ExpectSetupFailure(&interface, &status,
      "M07 Alquimia state index exceeds C int range",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      "\",\"inputs\":[{\"tensor\":\"double_input_2\",\"tensor_element_index\":0,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\","
      // alquimia_state_index = 2147383648 > INT_MAX, in the onnx_alquimia_manifest.c
      "\"alquimia_state_index\":2147483648}],\"outputs\":[" VALID_OUTPUT
      "]}", "alquimia_state_index");

  // | M08 | Scalar mapping uses a nonzero `alquimia_state_index` | Setup error |
  ExpectSetupFailure(&interface, &status,
      "M08 scalar has nonzero Alquimia state index",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      "\",\"inputs\":[{\"tensor\":\"double_input_2\",\"tensor_element_index\":0,"
      // temperature is a scalar in AlquimiaState, the index should = 0 
      "\"feature\":\"temperature\",\"alquimia_state\":\"temperature\","
      "\"alquimia_state_index\":1}," VALID_INPUT_1 "],\"outputs\":[" VALID_OUTPUT "]}",
      "incompatible with variable 'temperature'");

  // | M09 | Tensor name is unknown | Setup error naming the tensor |
  ExpectSetupFailure(&interface, &status, "M09 unknown tensor",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      // Invalid "double_input", the valid input tensor name for this model is double_input_2
      "\",\"inputs\":[{\"tensor\":\"double_input\",\"tensor_element_index\":0,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0},"
      VALID_INPUT_1 "],\"outputs\":[" VALID_OUTPUT "]}",
      "unknown tensor 'double_input'");
      
  // | M10 | `tensor_element_index` exceeds its flattened extent | Setup error |
  ExpectSetupFailure(&interface, &status,
      "M10 tensor element index exceeds tensor extent",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      // This model only has 2 input, index = 2 means there are 3 inputs
      "\",\"inputs\":[{\"tensor\":\"double_input_2\",\"tensor_element_index\":2,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0},"
      VALID_INPUT_1 "],\"outputs\":[" VALID_OUTPUT "]}", "out of range");
      
  // | M11 | A tensor element index is mapped more than once | Setup error |
  ExpectSetupFailure(&interface, &status, "M10 output exceeds tensor extent",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      "\",\"inputs\":[" VALID_INPUT_0 "," VALID_INPUT_1
      // This model only has 1 output, index = 1 means there are 2 outputs
      "],\"outputs\":[{\"tensor\":\"double_output_2\",\"tensor_element_index\":1,"
      "\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0}]}", "out of range");
  ExpectSetupFailure(&interface, &status,
      "M11 duplicate tensor element index",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      // Duplicate input tensor
      "\",\"inputs\":[" VALID_INPUT_0 "," VALID_INPUT_0
      "],\"outputs\":[" VALID_OUTPUT "]}", "Duplicate ONNX input mapping");
  ExpectSetupFailure(&interface, &status,
      "M11 duplicate output tensor element index",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      "\",\"inputs\":[" VALID_INPUT_0 "," VALID_INPUT_1
      // Duplicate output tensor
      "],\"outputs\":[" VALID_OUTPUT "," VALID_OUTPUT "]}",
      "Duplicate ONNX output mapping");

  // | M12 | A required model tensor element index is unmapped | Setup error |
  ExpectSetupFailure(&interface, &status,
      "M12 tensor element index is unmapped",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      // This model expect two input, here it only accepts 1
      "\",\"inputs\":[" VALID_INPUT_0 "],\"outputs\":[" VALID_OUTPUT "]}",
      "does not map every input tensor element");
  ExpectSetupFailure(&interface, &status,
      "M12 output tensor element index is unmapped",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      "\",\"inputs\":[" VALID_INPUT_0 "," VALID_INPUT_1
      // outputs is empty
      "],\"outputs\":[]}", "does not map every output tensor element");

  // | M13 | Mappings derive unsafe or overflowing Alquimia sizes | Setup error |
  ExpectSetupFailure(&interface, &status, "M13 derived size overflows int",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      "\",\"inputs\":[{\"tensor\":\"double_input_2\",\"tensor_element_index\":0,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\","
      // Line 350 in onnx_alquimia_interface.c 
      // AlquimiaSize = index + 1
      "\"alquimia_state_index\":2147483647}," VALID_INPUT_1 "],"
      "\"outputs\":[" VALID_OUTPUT "]}", "incompatible with variable");

  // | M14 | Feature names target different state categories | Correct problem-metadata vectors are populated |
  // | M17 | Model path is relative to the manifest | Correct path is resolved from the manifest directory |
  TestSuccessfulMappingCases(&interface, &status);

  // | M15 | Two names target the same metadata destination | Setup rejects the ambiguity |
  ExpectSetupFailure(&interface, &status, "M15 conflicting feature names",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      // Test the case:
      // tensor, tensor element index are the same
      // But the feature name is different
      "\",\"inputs\":[{\"tensor\":\"double_input_2\",\"tensor_element_index\":0,"
      "\"feature\":\"first_name\",\"alquimia_state\":\"total_mobile\","
      "\"alquimia_state_index\":0},{\"tensor\":\"double_input_2\",\"tensor_element_index\":1,"
      "\"feature\":\"second_name\",\"alquimia_state\":\"total_immobile\","
      "\"alquimia_state_index\":0}],\"outputs\":[" VALID_OUTPUT "]}",
      "Conflicting ONNX feature names");

  // | M16 | Similar but invalid property has trailing text | Property is rejected rather than partially matched |
  ExpectSetupFailure(&interface, &status, "M16 property trailing text",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      "\",\"inputs\":[{\"tensor\":\"double_input_2\",\"tensor_element_index\":0,"
      "\"feature\":\"f\",\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":0,"
      // Extra "alquimia_state_index_extra"
      "\"alquimia_state_index_extra\":0}],\"outputs\":[" VALID_OUTPUT "]}",
      "Unknown property 'alquimia_state_index_extra'");

  // | M18 | Two input mappings use the same feature name | Setup rejects the duplicate lookup key |
  ExpectSetupFailure(&interface, &status,
      "M18 duplicate input feature name",
      "{\"schema_version\":1,\"model\":\"" MODEL_2_PATH
      // Duplicate input feature
      "\",\"inputs\":[{\"tensor\":\"double_input_2\",\"tensor_element_index\":0,"
      "\"feature\":\"duplicate_feature\",\"alquimia_state\":\"total_mobile\","
      "\"alquimia_state_index\":0},{\"tensor\":\"double_input_2\","
      "\"tensor_element_index\":1,\"feature\":\"duplicate_feature\","
      "\"alquimia_state\":\"total_mobile\",\"alquimia_state_index\":1}],"
      "\"outputs\":[" VALID_OUTPUT "]}",
      "Duplicate ONNX input feature name 'duplicate_feature'.");

  FreeAlquimiaEngineStatus(&status);
}

#endif

int main(int argc, char **argv)
{
  /* The test file couples with the error message in the related c file 
  ** Change the error message string could potentially fail the test
  */
#if ALQUIMIA_HAVE_ONNX
  /* Disable buffering for stdout and stderr to flush logs immediately */
  setvbuf(stdout, NULL, _IONBF, 0);
  setvbuf(stderr, NULL, _IONBF, 0);

  if (argc == 2 && strcmp(argv[1], "manifest") == 0)
  {
    // Enter: ./test_alquimia_onnx manifest to test the JSON parser, otherwise test the lifecycle
    TestManifestContract();
  }
  else
  {
    TestModel2Lifecycle();
  }
#else
  (void)argc;
  (void)argv;
  printf("ONNX not enabled. Skipping ONNX engine unit test.\n");
#endif
  return EXIT_SUCCESS;
}
