/* -*-  mode: c; c-default-style: "google"; indent-tabs-mode: nil -*- 
** This file is to test the onnx inference and output routing
** | ID | Scenario | Expected result |
** |---|---|---|
** | R01 | Single input, single output | Exact expected state values |
** | R02 | Multiple inputs, single output | Tensor names and element positions route correctly |
** | R03 | Single input, multiple outputs | Every output mapping is applied |
** | R04 | Multiple inputs, multiple outputs | Tensor boundaries and mappings remain correct |
** | R05 | Mixed scalar and vector mappings | Values reach the correct state fields |
** | R06 | Outputs update different state categories | No cross-category overwrite |
** | R07 | Output vector is missing or undersized | Integrity error; no out-of-bounds write |
** | R08 | ONNX Runtime inference fails | Translated status; engine remains safe to shut down |
** | R09 | Repeated inference calls | Reusable buffers retain no incorrect prior values |
** | R10 | Two independent engine instances | No state or buffer sharing between instances |
** | R11 | Multiple rank-0 scalar inputs and outputs | Preserve scalar tensor rank during inference |
** | R12 | One or both phase outputs change paired components | Conserve only one-sided outputs |
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "alquimia/alquimia_constants.h"
#include "alquimia/alquimia_interface.h"
#include "alquimia/alquimia_memory.h"

#if ALQUIMIA_HAVE_ONNX

typedef struct {
  AlquimiaEngineStatus status;
  AlquimiaInterface interface;
  AlquimiaSizes sizes;
  AlquimiaEngineFunctionality functionality;
  void *engine_state;
} OnnxTestEngine;

/* Prevents trailing semicolon (`;`) compiler warnings/errors.
** 
** When using the macro inside a block:
**   if (condition) {
**     CHECK_CASE(test_id, expr);
**   }
** 
** Without do-while(0), an accidental trailing empty statement (e.g., `if (cond) { macro();; }`)
** can trigger strict compilation flags (like -Wpedantic) and halt the build.
** This do-while(0) pattern ensures safe block scoping and exact statement matching.
** 
** #define macro(x) if(){} => if(){}
** if (condition) 
**  CHECK_CASE(test_id, expr); => if(){}; 
** End the outer if statement
** Just in case of habit issue(adding ;)
** Expect: condition true
*/
#define CHECK(condition)                                                    \
  do {                                                                      \
    if (!(condition)) {                                                     \
      fprintf(stderr, "Check failed at %s:%d: %s\n", __FILE__, __LINE__, \
              #condition);                                                  \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  } while (0)

static void CheckClose(double actual, double expected) {
  CHECK(fabs(actual - expected) < 1.0e-12);
}

static void CheckCloseCase(
    const char *test_id,
    const char *value_name,
    double actual,
    double expected) {
  if (fabs(actual - expected) >= 1.0e-12) {
    fprintf(stderr, "%s: %s was %.17g; expected %.17g.\n",
            test_id, value_name, actual, expected);
    exit(EXIT_FAILURE);
  }
}

/**
 * @brief Initializes and sets up the ONNX engine instance for a given test fixture.
 * 
 * @param relative_path Relative path or filename of the ONNX config fixture.
 * @param[out] engine Pointer to the OnnxTestEngine instance to be configured.
 */
static void SetupEngine(const char *relative_path, OnnxTestEngine *engine) {
  char config_path[2048];

  // Initialize the test engine
  memset(engine, 0, sizeof(*engine));
  AllocateAlquimiaEngineStatus(&engine->status);
  CreateAlquimiaInterface("ONNX", &engine->interface, &engine->status);
  // Expect: engine->status.error == kAquimiaNoError
  CHECK(engine->status.error == kAlquimiaNoError);

  snprintf(config_path, sizeof(config_path), "%s/onnx_test_cases/%s",
           CMAKE_CURRENT_SOURCE_DIR, relative_path);
  engine->interface.Setup(config_path, false, &engine->engine_state,
                          &engine->sizes, &engine->functionality,
                          &engine->status);
  if (engine->status.error != kAlquimiaNoError) {
    fprintf(stderr, "Setup failed for %s: %s\n", relative_path,
            engine->status.message);
    exit(EXIT_FAILURE);
  }
  // Expect: engine->engine_state != NULL
  CHECK(engine->engine_state != NULL);
}

static void ShutdownEngine(OnnxTestEngine *engine) {
  engine->interface.Shutdown(&engine->engine_state, &engine->status);
  CHECK(engine->status.error == kAlquimiaNoError);
  CHECK(engine->engine_state == NULL);
  FreeAlquimiaEngineStatus(&engine->status);
} 

/* Zero-initializes and allocates memory for an AlquimiaState instance based on engine sizes. */
static void AllocateState(
    const OnnxTestEngine *engine,
    AlquimiaState *state) {
  memset(state, 0, sizeof(*state));
  AllocateAlquimiaState(&engine->sizes, state);
}

/* Run the ONNX inference */
static void RunInference(OnnxTestEngine *engine, AlquimiaState *state) {
  AlquimiaProperties properties = {0};
  AlquimiaAuxiliaryData auxiliary_data = {0};

  engine->interface.ReactionStepOperatorSplit(
      &engine->engine_state, 1.0, &properties, state, &auxiliary_data, 0,
      &engine->status);
  if (engine->status.error != kAlquimiaNoError) {
    fprintf(stderr, "Inference failed: %s\n", engine->status.message);
    exit(EXIT_FAILURE);
  }
}

// | R01 | Single input, single output | Exact expected state values |
static void TestR01SingleInputSingleOutput(void) {
  OnnxTestEngine engine;
  AlquimiaState state;

  // Set up the ONNX interface
  SetupEngine("deterministic/identity_double.json", &engine);

  // Expect: engine.sizes.num_primary == 1
  CHECK(engine.sizes.num_primary == 1);

  // Allocate AlquimiaState
  AllocateState(&engine, &state);
  state.total_mobile.data[0] = 4.25;

  // Run the inference
  RunInference(&engine, &state);

  // Check the inference result (Check the JSON and ONNX for mapping info)
  // Expect total_mobile.data[0] = 4.25
  CheckClose(state.total_mobile.data[0], 4.25);
  FreeAlquimiaState(&state);
  ShutdownEngine(&engine);
}

// | R02 | Multiple inputs, single output | Tensor names and element positions route correctly |
static void TestR02MultipleInputsSingleOutput(void) {
  OnnxTestEngine engine;
  AlquimiaState state;

  // Set up the ONNX interface (Two inputs, one output)
  SetupEngine("deterministic/add_two_inputs.json", &engine);

  // Expect: engine.sizes.num_primary = 3
  CHECK(engine.sizes.num_primary == 3);

  // Allocate AlquimiaState
  AllocateState(&engine, &state);

  // Set up the input tensor
  state.total_mobile.data[0] = 2.5;
  state.total_mobile.data[1] = -1.0;
  state.total_mobile.data[2] = 99.0;
  
  // Run the inference
  RunInference(&engine, &state);

  // Check the inference result (Check the JSON and ONNX for mapping info)
  // Expect: state.total_mobile.data[0] = 2.5
  CheckClose(state.total_mobile.data[0], 2.5);

  // Expect: state.total_mobile.data[1] = -1.0
  CheckClose(state.total_mobile.data[1], -1.0);

  // Expect: state.total_mobile.data[2] = 1.5
  CheckClose(state.total_mobile.data[2], 1.5);
  FreeAlquimiaState(&state);
  ShutdownEngine(&engine);
}

// | R03 | Single input, multiple outputs | Every output mapping is applied |
static void TestR03SingleInputMultipleOutputs(void) {
  OnnxTestEngine engine;
  AlquimiaState state;

  SetupEngine("deterministic/single_input_multiple_outputs.json", &engine);

  // Expect: engine.sizes.num_primary = 4
  CHECK(engine.sizes.num_primary == 4);

  // Expect: engine.sizes.num_sorbed = 2
  CHECK(engine.sizes.num_sorbed == 2);
  AllocateState(&engine, &state);

  // Set up the input tensor
  state.total_mobile.data[0] = 3.0;
  state.total_mobile.data[1] = -5.0;

  // Run the inference
  RunInference(&engine, &state);

  // Check the inference result (Check the JSON and ONNX for mapping info)
  CheckClose(state.total_mobile.data[2], 3.0);
  CheckClose(state.total_mobile.data[3], -5.0);
  CheckClose(state.total_immobile.data[0], 13.0);
  CheckClose(state.total_immobile.data[1], 15.0);
  FreeAlquimiaState(&state);
  ShutdownEngine(&engine);
}

// | R04 | Multiple inputs, multiple outputs | Tensor boundaries and mappings remain correct |
static void TestR04MultipleInputsMultipleOutputs(void) {
  OnnxTestEngine engine;
  AlquimiaState state;

  SetupEngine("deterministic/multiple_inputs_outputs.json", &engine);
  AllocateState(&engine, &state);

  // Input tensor
  state.porosity = 2.0;
  state.total_mobile.data[0] = 5.0;
  RunInference(&engine, &state);

  // Check the inference result (Check the JSON and ONNX for mapping info)
  // Output tensor
  CheckClose(state.temperature, 7.0);
  CheckClose(state.gas_concentration.data[0], 19.0);

  // Check the input tensor
  CheckClose(state.porosity, 2.0);
  CheckClose(state.total_mobile.data[0], 5.0);
  FreeAlquimiaState(&state);
  ShutdownEngine(&engine);
}

// | R05 | Mixed scalar and vector mappings | Values reach the correct state fields |
static void TestR05MixedScalarVectorMappings(void) {
  OnnxTestEngine engine;
  AlquimiaState state;

  SetupEngine("deterministic/mixed_scalar_vector.json", &engine);
  CHECK(engine.sizes.num_primary == 2);
  CHECK(engine.sizes.num_minerals == 2);
  CHECK(engine.sizes.num_surface_sites == 1);
  CHECK(engine.sizes.num_gases == 1);
  AllocateState(&engine, &state);
  state.porosity = 0.35;
  state.total_mobile.data[1] = 8.0;
  state.gas_concentration.data[0] = -2.0;
  RunInference(&engine, &state);

  // Check the inference result (Check the JSON and ONNX for mapping info)
  // Check the output tensor
  CheckClose(state.temperature, 0.35);
  CheckClose(state.mineral_volume_fraction.data[1], 8.0);
  CheckClose(state.surface_site_density.data[0], -2.0);

  // Check the input tensor
  CheckClose(state.porosity, 0.35);
  CheckClose(state.total_mobile.data[1], 8.0);
  CheckClose(state.gas_concentration.data[0], -2.0);
  FreeAlquimiaState(&state);
  ShutdownEngine(&engine);
}

// | R06 | Outputs update different state categories | No cross-category overwrite |
static void TestR06AllStateCategories(void) {
  OnnxTestEngine engine;
  AlquimiaState state;

  SetupEngine("deterministic/all_state_categories.json", &engine);
  CHECK(engine.sizes.num_primary == 1);
  CHECK(engine.sizes.num_sorbed == 2);
  CHECK(engine.sizes.num_minerals == 2);
  AllocateState(&engine, &state);
  
  // Setup input tensor
  state.water_density = 101.0;
  state.porosity = 102.0;
  state.temperature = 103.0;
  state.aqueous_pressure = 104.0;
  state.total_mobile.data[0] = 105.0;
  state.total_immobile.data[0] = -1.0;
  state.total_immobile.data[1] = 106.0;
  state.mineral_volume_fraction.data[0] = 107.0;
  state.mineral_volume_fraction.data[1] = -2.0;
  state.mineral_specific_surface_area.data[0] = -3.0;
  state.mineral_specific_surface_area.data[1] = 108.0;
  state.surface_site_density.data[0] = 109.0;
  state.cation_exchange_capacity.data[0] = 110.0;
  state.gas_concentration.data[0] = 111.0;
  
  RunInference(&engine, &state);

  // Check the inference result (Check the JSON and ONNX for mapping info)
  // Check the output tensor
  CheckClose(state.water_density, 101.0);
  CheckClose(state.porosity, 102.0);
  CheckClose(state.temperature, 103.0);
  CheckClose(state.aqueous_pressure, 104.0);
  CheckClose(state.total_mobile.data[0], 105.0);
  CheckClose(state.total_immobile.data[0], -1.0);
  CheckClose(state.total_immobile.data[1], 106.0);
  CheckClose(state.mineral_volume_fraction.data[0], 107.0);
  CheckClose(state.mineral_volume_fraction.data[1], -2.0);
  CheckClose(state.mineral_specific_surface_area.data[0], -3.0);
  CheckClose(state.mineral_specific_surface_area.data[1], 108.0);
  CheckClose(state.surface_site_density.data[0], 109.0);
  CheckClose(state.cation_exchange_capacity.data[0], 110.0);
  CheckClose(state.gas_concentration.data[0], 111.0);
  FreeAlquimiaState(&state);
  ShutdownEngine(&engine);
}

// | R07 | Output vector is missing or undersized | Integrity error; no out-of-bounds write |
static void TestR07UndersizedOutputVector(void) {
  OnnxTestEngine engine;
  AlquimiaState state;

  SetupEngine("deterministic/add_two_inputs.json", &engine);
  AllocateState(&engine, &state);
  state.total_mobile.data[0] = 4.0;
  state.total_mobile.data[1] = 6.0;
  state.total_mobile.data[2] = 12345.0;
  // Undersized
  state.total_mobile.size = 2;

  {
    AlquimiaProperties properties = {0};
    AlquimiaAuxiliaryData auxiliary_data = {0};
    engine.interface.ReactionStepOperatorSplit(
        &engine.engine_state, 1.0, &properties, &state, &auxiliary_data, 0,
        &engine.status);
  }
  CHECK(engine.status.error == kAlquimiaErrorEngineIntegrity);

  // Find the error substring, couples with the onnx_alquimia_interface.c
  // Check the inference result (Check the JSON and ONNX for mapping info)
  CHECK(strstr(engine.status.message,
               "Out-of-bounds total_mobile write") != NULL);
  CheckClose(state.total_mobile.data[2], 12345.0);
  state.total_mobile.size = 3;
  FreeAlquimiaState(&state);
  ShutdownEngine(&engine);
}

// | R08 | ONNX Runtime inference fails | Translated status; engine remains safe to shut down |
static void TestR08RuntimeInferenceFailure(void) {
  OnnxTestEngine engine;
  AlquimiaState state;
  AlquimiaProperties properties = {0};
  AlquimiaAuxiliaryData auxiliary_data = {0};

  SetupEngine("deterministic/runtime_failure.json", &engine);
  AllocateState(&engine, &state);
  state.total_mobile.data[0] = 1.0;
  state.total_mobile.data[1] = 2.0;
  
  // runtime_failure.onnx has an invalid opset
  engine.interface.ReactionStepOperatorSplit(
      &engine.engine_state, 1.0, &properties, &state, &auxiliary_data, 0,
      &engine.status);
  CHECK(engine.status.error == kAlquimiaErrorEngineIntegrity);

  // Find the error substring, couples with the onnx_alquimia_interface.c
  CHECK(strstr(engine.status.message, "ONNX Runtime Error") != NULL);
  CHECK(engine.engine_state != NULL);
  FreeAlquimiaState(&state);
  ShutdownEngine(&engine);
}

// | R09 | Repeated inference calls | Reusable buffers retain no incorrect prior values |
static void TestR09RepeatedInference(void) {
  OnnxTestEngine engine;
  AlquimiaState state;

  SetupEngine("deterministic/affine_double.json", &engine);
  AllocateState(&engine, &state);
  state.total_mobile.data[0] = 1.0;
  RunInference(&engine, &state);
  // Check the inference result (Check the JSON and ONNX for mapping info)
  CheckClose(state.total_mobile.data[0], 5.0);
  state.total_mobile.data[0] = 4.0;
  RunInference(&engine, &state);
  CheckClose(state.total_mobile.data[0], 11.0);
  state.total_mobile.data[0] = -3.0;
  RunInference(&engine, &state);
  CheckClose(state.total_mobile.data[0], -3.0);
  FreeAlquimiaState(&state);
  ShutdownEngine(&engine);
}

// | R10 | Two independent engine instances | No state or buffer sharing between instances |
static void TestR10IndependentInstances(void) {
  OnnxTestEngine first;
  OnnxTestEngine second;
  AlquimiaState first_state;
  AlquimiaState second_state;

  SetupEngine("deterministic/affine_double.json", &first);
  SetupEngine("deterministic/affine_double.json", &second);
  AllocateState(&first, &first_state);
  AllocateState(&second, &second_state);
  first_state.total_mobile.data[0] = 2.0;
  second_state.total_mobile.data[0] = 10.0;
  RunInference(&first, &first_state);
  RunInference(&second, &second_state);
  
  // Check the output tensor
  // Check the inference result (Check the JSON and ONNX for mapping info)
  CheckClose(first_state.total_mobile.data[0], 7.0);
  CheckClose(second_state.total_mobile.data[0], 23.0);
  first_state.total_mobile.data[0] = -1.0;

  RunInference(&first, &first_state);
  CheckClose(first_state.total_mobile.data[0], 1.0);
  CheckClose(second_state.total_mobile.data[0], 23.0);
  FreeAlquimiaState(&first_state);
  FreeAlquimiaState(&second_state);
  ShutdownEngine(&first);
  ShutdownEngine(&second);
}

// | R11 | Multiple rank-0 scalar inputs and outputs | Preserve scalar tensor rank during inference |
static void TestR11MultipleScalarInputsOutputs(void) {
  OnnxTestEngine engine;
  AlquimiaState state;

  SetupEngine("deterministic/multiple_scalar_inputs_outputs.json", &engine);
  CHECK(engine.sizes.num_primary == 1);
  AllocateState(&engine, &state);
  state.total_mobile.data[0] = 2.5;
  state.porosity = -1.0;

  RunInference(&engine, &state);

  CheckClose(state.water_density, 1.5);
  CheckClose(state.aqueous_pressure, 3.5);
  CheckClose(state.total_mobile.data[0], 2.5);
  CheckClose(state.porosity, -1.0);
  FreeAlquimiaState(&state);
  ShutdownEngine(&engine);
}

// | R12 | One or both phase outputs change paired components | Conserve only one-sided outputs |
static void TestR12MobileImmobileConservation(void) {
  OnnxTestEngine engine;
  AlquimiaState state;

  SetupEngine("deterministic/mobile_immobile_conservation.json", &engine);
  CHECK(engine.sizes.num_primary == 2);
  CHECK(engine.sizes.num_sorbed == 2);
  AllocateState(&engine, &state);

  state.total_mobile.data[0] = 3.0;
  state.total_mobile.data[1] = 4.0;
  state.total_immobile.data[0] = 17.0;
  state.total_immobile.data[1] = 6.0;

  RunInference(&engine, &state);

  /* shifted[0] changes mobile component 0 from 3 to 13. Its paired
  ** immobile value must decrease by 10 to preserve the total of 20. */
  CheckCloseCase("R12 mobile-immobile conservation", "total_mobile[0]",
                 state.total_mobile.data[0], 13.0);
  CheckCloseCase("R12 mobile-immobile conservation", "total_immobile[0]",
                 state.total_immobile.data[0], 7.0);
  /* copy[1] and shifted[1] explicitly output both phases of component 1,
  ** so both model values remain authoritative without conservation. */
  CheckCloseCase("R12 mobile-immobile conservation", "total_mobile[1]",
                 state.total_mobile.data[1], 4.0);
  CheckCloseCase("R12 mobile-immobile conservation", "total_immobile[1]",
                 state.total_immobile.data[1], 24.0);

  FreeAlquimiaState(&state);
  ShutdownEngine(&engine);
}

// | F01 | Linear or identity graph | Baseline tensor and operator compatibility |
static void TestF01LinearAffine(void) {
  OnnxTestEngine engine;
  AlquimiaState state;

  SetupEngine("model_families/linear_affine.json", &engine);
  AllocateState(&engine, &state);
  state.total_mobile.data[0] = 3.0;
  RunInference(&engine, &state);
  CheckClose(state.total_mobile.data[0], 10.0);
  FreeAlquimiaState(&state);
  ShutdownEngine(&engine);
}

// | F02 | SVR | `ai.onnx.ml` regression compatibility |
static void TestF02Svr(void) {
  OnnxTestEngine engine;
  AlquimiaState state;

  SetupEngine("model_families/svr_linear.json", &engine);
  AllocateState(&engine, &state);
  state.total_mobile.data[0] = 3.0;
  RunInference(&engine, &state);
  CHECK(isfinite(state.total_mobile.data[0]));
  FreeAlquimiaState(&state);
  ShutdownEngine(&engine);
}

// | F03 | Small neural network | Standard dense activation graph compatibility |
static void TestF03SmallNeuralNetwork(void) {
  OnnxTestEngine engine;
  AlquimiaState state;

  SetupEngine("model_families/small_neural_network.json", &engine);
  CHECK(engine.sizes.num_primary == 3);
  AllocateState(&engine, &state);
  state.total_mobile.data[0] = 3.0;
  state.total_mobile.data[1] = 4.0;
  RunInference(&engine, &state);

  // This is a little bit suspicious
  // output = ReLU(input × weights + bias)

  // weights = [ 2 ]
  //           [-1 ]

  // bias = 1

  // The test supplies input = [3, 4] at unit_tests/
  // test_alquimia_onnx_routing.c:465, so:

  // linear = 3×2 + 4×(-1) = 2
  // biased = 2 + 1 = 3
  // output = ReLU(3) = 3
  CheckClose(state.total_mobile.data[2], 3.0);
  FreeAlquimiaState(&state);
  ShutdownEngine(&engine);
}

// | F04 | Tree ensemble | `ai.onnx.ml` tree operator compatibility |
static void TestF04TreeEnsemble(void) {
  OnnxTestEngine engine;
  AlquimiaState state;

  SetupEngine("model_families/tree_ensemble.json", &engine);
  AllocateState(&engine, &state);
  state.total_mobile.data[0] = -100.0;
  RunInference(&engine, &state);

  // It has only one node, and that node is a leaf with value 7.0.
  CheckClose(state.total_mobile.data[1], 7.0);
  FreeAlquimiaState(&state);
  ShutdownEngine(&engine);
}

// | F05 | Multi-target model | Compatibility with multiple outputs |
static void TestF05MultiTarget(void) {
  OnnxTestEngine engine;
  AlquimiaState state;

  SetupEngine("model_families/multi_target.json", &engine);
  AllocateState(&engine, &state);
  state.total_mobile.data[0] = 2.0;
  state.total_mobile.data[1] = 3.0;
  RunInference(&engine, &state);
  // For input [2, 3], the first output is:

  // target_sum = 2×1 + 3×1 = 5

  // The second output is:

  // target_affine = 2×2 + 3×(-1) + 5
  //               = 4 - 3 + 5
  //               = 6
  // Using Netro for more info
  CheckClose(state.total_immobile.data[0], 5.0);
  CheckClose(state.gas_concentration.data[0], 6.0);
  FreeAlquimiaState(&state);
  ShutdownEngine(&engine);
}
/* Testing R01-R12 */
static void RunRoutingTests(void) {
  TestR01SingleInputSingleOutput();
  TestR02MultipleInputsSingleOutput();
  TestR03SingleInputMultipleOutputs();
  TestR04MultipleInputsMultipleOutputs();
  TestR05MixedScalarVectorMappings();
  TestR06AllStateCategories();
  TestR07UndersizedOutputVector();
  TestR08RuntimeInferenceFailure();
  TestR09RepeatedInference();
  TestR10IndependentInstances();
  TestR11MultipleScalarInputsOutputs();
  TestR12MobileImmobileConservation();
  printf("ONNX routing cases R01-R12 passed.\n");
}

/*
** | ID | Family | Purpose |
** |---|---|---|
** | F01 | Linear or identity graph | Baseline tensor and operator compatibility |
** | F02 | SVR | `ai.onnx.ml` regression compatibility |
** | F03 | Small neural network | Standard dense activation graph compatibility |
** | F04 | Tree ensemble | `ai.onnx.ml` tree operator compatibility |
** | F05 | Multi-target model | Compatibility with multiple outputs |
*/
static void RunModelFamilyTests(void) {
  TestF01LinearAffine();
  TestF02Svr();
  TestF03SmallNeuralNetwork();
  TestF04TreeEnsemble();
  TestF05MultiTarget();
  printf("ONNX model-family cases F01-F05 passed.\n");
}

#endif

int main(int argc, char **argv) {
#if ALQUIMIA_HAVE_ONNX
  if (argc == 1) {
    // Enter ./test_alquimia_onnx_routing  
    RunRoutingTests();
    RunModelFamilyTests();
    // Test separately
  } else if (argc == 2 && strcmp(argv[1], "routing") == 0) {
    // Enter ./test_alquimia_onnx_routing routing 
    RunRoutingTests();
  } else if (argc == 2 && strcmp(argv[1], "model-family") == 0) {
    // Enter ./test_alquimia_onnx_routing model-family
    RunModelFamilyTests();
  } else {
    fprintf(stderr, "Usage: %s [routing|model-family]\n", argv[0]);
    return EXIT_FAILURE;
  }
#else
  (void)argc;
  (void)argv;
  printf("ONNX not enabled. Skipping ONNX routing tests.\n");
#endif
  return EXIT_SUCCESS;
}
