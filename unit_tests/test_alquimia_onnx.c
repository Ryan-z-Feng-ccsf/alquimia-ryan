/* -*-  mode: c; c-default-style: "google"; indent-tabs-mode: nil -*- */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "alquimia/alquimia_constants.h"
#include "alquimia/alquimia_memory.h"
#include "alquimia/alquimia_interface.h"
#include "alquimia/alquimia_util.h"

#if ALQUIMIA_HAVE_ONNX
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

#include "alquimia/onnx_alquimia_interface.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
#endif

#if ALQUIMIA_HAVE_ONNX

typedef struct {
  const char* filename;
  int num_features;
  const char* features[6];
  double inputs[6];
} ModelTestCase;

void test_SingleModelLifecycle(const ModelTestCase* test_case) {
  AlquimiaEngineStatus status;
  AlquimiaInterface interface;
  AlquimiaSizes sizes;
  AlquimiaEngineFunctionality functionality;
  void* engine_state = NULL;

  AllocateAlquimiaEngineStatus(&status);

  // 1. Create Interface
  CreateAlquimiaInterface("ONNX", &interface, &status);
  ALQUIMIA_ASSERT(status.error == kAlquimiaNoError);

  // 2. Setup (Load model)
  char model_path[2048];
  snprintf(model_path, sizeof(model_path), "%s/../models/%s", CMAKE_CURRENT_SOURCE_DIR, test_case->filename);

  // Note: Setup takes &engine_state (pointer-to-pointer)
  interface.Setup(model_path, false, &engine_state, &sizes, &functionality, &status);
  if (status.error != kAlquimiaNoError) {
    fprintf(stderr, "Setup failed for %s! Error: %d, Message: %s\n", test_case->filename, status.error, status.message);
    ALQUIMIA_ASSERT(strstr(status.message, "Missing required ONNX mapping metadata key") != NULL);
    ALQUIMIA_ASSERT(engine_state == NULL);
    FreeAlquimiaEngineStatus(&status);
    return;
  }

  ALQUIMIA_ASSERT(engine_state != NULL);
  ALQUIMIA_ASSERT(sizes.num_primary == test_case->num_features);

  // 3. Problem Metadata
  AlquimiaProblemMetaData meta_data;
  AllocateAlquimiaProblemMetaData(&sizes, &meta_data);
  interface.GetProblemMetaData(&engine_state, &meta_data, &status);
  ALQUIMIA_ASSERT(status.error == kAlquimiaNoError);

  // Verify feature names align perfectly with the model's custom metadata
  printf("Model: %s\n", test_case->filename);
  for (int i = 0; i < test_case->num_features; ++i) {
    printf("  Index %d: expected = '%s', actual = '%s'\n", i, test_case->features[i], meta_data.primary_names.data[i]);
  }

  for (int i = 0; i < test_case->num_features; ++i) {
    ALQUIMIA_ASSERT(strcmp(meta_data.primary_names.data[i], test_case->features[i]) == 0);
  }

  // 4. Process Condition (Loads dynamic state from geochemical condition)
  AlquimiaState state;
  AllocateAlquimiaState(&sizes, &state);

  AlquimiaGeochemicalCondition condition;
  AllocateAlquimiaGeochemicalCondition(7, test_case->num_features, 0, &condition);
  strcpy(condition.name, "initial");
  for (int i = 0; i < test_case->num_features; ++i) {
    AllocateAlquimiaAqueousConstraint(&condition.aqueous_constraints.data[i]);
    strcpy(condition.aqueous_constraints.data[i].primary_species_name, test_case->features[i]);
    strcpy(condition.aqueous_constraints.data[i].constraint_type, "total");
    condition.aqueous_constraints.data[i].value = test_case->inputs[i];
  }

  AlquimiaProperties properties = {0};
  AlquimiaAuxiliaryData aux_data = {0};

  interface.ProcessCondition(&engine_state, &condition, &properties, &state, &aux_data, &status);
  ALQUIMIA_ASSERT(status.error == kAlquimiaNoError);

  // Assert that values were dynamically assigned from condition -> state
  for (int i = 0; i < test_case->num_features; ++i) {
    ALQUIMIA_ASSERT(fabs(state.total_mobile.data[i] - test_case->inputs[i]) < 1e-9);
  }

  FreeAlquimiaGeochemicalCondition(&condition);

  // 5. Run ReactionStepOperatorSplit (Inference)
  interface.ReactionStepOperatorSplit(&engine_state, 1.0, &properties, &state, &aux_data, 1, &status);
  ALQUIMIA_ASSERT(status.error == kAlquimiaNoError);

  // Print output for verification
  printf("Model %s output (uranium_total): %.15f\n", test_case->filename, state.total_mobile.data[0]);

  // Strictly validate Model 5 output
  if (strcmp(test_case->filename, "lsurf_model_5_float_64.onnx") == 0) {
    ALQUIMIA_ASSERT(fabs(state.total_mobile.data[0] - (-6.691744980518146)) < 1e-4);
  }

  // 6. Cleanup
  FreeAlquimiaState(&state);
  FreeAlquimiaProblemMetaData(&meta_data);
  interface.Shutdown(&engine_state, &status);
  ALQUIMIA_ASSERT(status.error == kAlquimiaNoError);
  ALQUIMIA_ASSERT(engine_state == NULL); // verified pointer-to-pointer nullification!

  FreeAlquimiaEngineStatus(&status);
}

#endif

int main(int argc, char** argv) {
  (void) argc;
  (void) argv;

#if ALQUIMIA_HAVE_ONNX
  // Disable standard I/O buffering to capture real-time program diagnostic prints
  setvbuf(stdout, NULL, _IONBF, 0);
  setvbuf(stderr, NULL, _IONBF, 0);

  ModelTestCase test_cases[] = {
    {
      "lsurf_model_1_float_64.onnx", 1,
      {"uranium_total"},
      {-6.677780705266080}
    },
    {
      "lsurf_model_2_float_64.onnx", 2,
      {"uranium_total", "U_species14"},
      {-6.677780705266080, -37.488999999999997}
    },
    {
      "lsurf_model_5_float_64.onnx", 5,
      {"uranium_total", "Site_Density", "U_species1", "U_species8", "U_species14"},
      {-6.677780705266080, -4.54327863489071, -25.419000000000000, -22.514000000000000, -37.488999999999997}
    },
    {
      "lsurf_model_6_float_64.onnx", 6,
      {"Mineral_source", "uranium_total", "Site_Density", "U_species1", "U_species8", "U_species20"},
      {7.0, -6.677780705266080, -4.54327863489071, -25.419000000000000, -22.514000000000000, -20.510000000000002}
    }
  };

  for (int i = 0; i < 4; ++i) {
    printf("--- Running Test for %s ---\n", test_cases[i].filename);
    test_SingleModelLifecycle(&test_cases[i]);
  }

  printf("All 4 ONNX model unit tests completed.\n");
#else
  printf("ONNX not enabled. Skipping ONNX Engine unit test.\n");
#endif

  return EXIT_SUCCESS;
}
