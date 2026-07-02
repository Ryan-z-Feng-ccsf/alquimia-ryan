/* -*-  mode: c; c-default-style: "google"; indent-tabs-mode: nil -*- */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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

void test_OnnxEngineLifecycle(void) {
#if ALQUIMIA_HAVE_ONNX
  AlquimiaEngineStatus status;
  AlquimiaInterface interface;
  AlquimiaSizes sizes;
  AlquimiaEngineFunctionality functionality;
  void* engine_state = NULL;

  AllocateAlquimiaEngineStatus(&status);

  // 1. Create Interface
  CreateAlquimiaInterface("ONNX", &interface, &status);
  ALQUIMIA_ASSERT(status.error == kAlquimiaNoError);
  ALQUIMIA_ASSERT(interface.Setup == &onnx_alquimia_setup);

  // 2. Setup (Load model)
  char model_path[2048];
  snprintf(model_path, sizeof(model_path), "%s/../alquimia/mixed/lsurf_model_float_64.onnx", CMAKE_CURRENT_SOURCE_DIR);
  interface.Setup(model_path, false, &engine_state, &sizes, &functionality, &status);
  if (status.error != kAlquimiaNoError) {
    fprintf(stderr, "Setup failed! Error: %d, Message: %s\n", status.error, status.message);
  }
  ALQUIMIA_ASSERT(status.error == kAlquimiaNoError);
  ALQUIMIA_ASSERT(engine_state != NULL);
  printf("Sizes: num_primary=%d, num_sorbed=%d, num_minerals=%d\n", sizes.num_primary, sizes.num_sorbed, sizes.num_minerals);
  ALQUIMIA_ASSERT(sizes.num_primary == 5);

  // Allocate and test problem metadata
  AlquimiaProblemMetaData meta_data;
  AllocateAlquimiaProblemMetaData(&sizes, &meta_data);

  interface.GetProblemMetaData(engine_state, &meta_data, &status);
  if (status.error != kAlquimiaNoError) {
    fprintf(stderr, "GetProblemMetaData failed! Error: %d, Message: %s\n", status.error, status.message);
  }
  ALQUIMIA_ASSERT(status.error == kAlquimiaNoError);
  ALQUIMIA_ASSERT(meta_data.primary_names.size == 5);

  // Verify feature names align perfectly with ONNX model's metadata
  printf("Verifying retrieved problem metadata:\n");
  for (int i = 0; i < 5; ++i) {
    printf("  Index %d: %s\n", i, meta_data.primary_names.data[i]);
  }
  ALQUIMIA_ASSERT(strcmp(meta_data.primary_names.data[0], "uranium_total") == 0);
  ALQUIMIA_ASSERT(strcmp(meta_data.primary_names.data[1], "Site_Density") == 0);
  ALQUIMIA_ASSERT(strcmp(meta_data.primary_names.data[2], "U_species1") == 0);
  ALQUIMIA_ASSERT(strcmp(meta_data.primary_names.data[3], "U_species8") == 0);
  ALQUIMIA_ASSERT(strcmp(meta_data.primary_names.data[4], "U_species14") == 0);
  printf("Custom ONNX metadata validated successfully.\n");

  // Allocate state data
  AlquimiaState state;
  AllocateAlquimiaState(&sizes, &state);

  AlquimiaGeochemicalCondition condition; // dummy
  AlquimiaProperties properties; // dummy
  AlquimiaAuxiliaryData aux_data; // dummy

  // 3. Process Condition (Loads static input array into state)
  interface.ProcessCondition(engine_state, &condition, &properties, &state, &aux_data, &status);
  if (status.error != kAlquimiaNoError) {
    fprintf(stderr, "ProcessCondition failed! Error: %d, Message: %s\n", status.error, status.message);
  }
  ALQUIMIA_ASSERT(status.error == kAlquimiaNoError);
  printf("State total_mobile[0] after ProcessCondition: %.15f\n", state.total_mobile.data[0]);
  ALQUIMIA_ASSERT(fabs(state.total_mobile.data[0] - (-6.67778070526608)) < 1e-9);

  // 4. Run ReactionStepOperatorSplit (Inference)
  interface.ReactionStepOperatorSplit(engine_state, 1.0, &properties, &state, &aux_data, 1, &status);
  if (status.error != kAlquimiaNoError) {
    fprintf(stderr, "ReactionStepOperatorSplit failed! Error: %d, Message: %s\n", status.error, status.message);
  }
  ALQUIMIA_ASSERT(status.error == kAlquimiaNoError);

  // Verify inference result (The reference model outputs ~0.134110825313175)
  printf("Unit Test Inference Result: %.15f\n", state.total_mobile.data[0]);
  ALQUIMIA_ASSERT(fabs(state.total_mobile.data[0] - (-6.691744980518146)) < 1e-4);

  // 5. Cleanup
  FreeAlquimiaState(&state);
  FreeAlquimiaProblemMetaData(&meta_data);
  interface.Shutdown(engine_state, &status);
  ALQUIMIA_ASSERT(status.error == kAlquimiaNoError);

  FreeAlquimiaEngineStatus(&status);
  printf("ONNX Engine unit test passed successfully.\n");
#else
  printf("ONNX not enabled. Skipping ONNX Engine unit test.\n");
#endif
}

int main(int argc, char** argv) {
  (void) argc;
  (void) argv;
  test_OnnxEngineLifecycle();
  return EXIT_SUCCESS;
}
