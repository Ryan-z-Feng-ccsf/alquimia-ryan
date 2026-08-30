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
** ONNX inference-routing and model-family unit tests.
**
** Authors:
**        Zhuolei Feng, Sergi Molins
**
** Notes:
**
**  * This file verifies that configured AlquimiaState values are routed
**    into ONNX input tensors and that ONNX outputs are written back to the
**    correct AlquimiaState fields.
**  * Test IDs use one prefix plus a two-digit stable case number.
**      R01, R02, ...: standard routing/inference behavior.
**      E01, E02, ...: routing/runtime error handling.
**      F01, F02, ...: supported ONNX model-family compatibility.
**      I01: full ALSURF lifecycle integration coverage.
**  * Example: R01 is the first routing case; F01 is the first
**    model-family compatibility case.
**  * Special routing cases, such as repeated calls and independent engine
**    instances, stay at the bottom of the R series.
**  * Tightly coupled: validates by matching substrings in the interface's
**    error message.
**  * Test JSON is provided in the test cases.
**
** ****************************************************************************
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "alquimia/alquimia_constants.h"
#include "alquimia/alquimia_interface.h"
#include "onnx_test_utils.h"

#if ALQUIMIA_HAVE_ONNX

/* ---------- Standard Cases ---------- */

/**
 * @brief Verifies the simplest one-input, one-output routing case.
 *
 * | R01 | Single input, single output | Exact expected state values |
 */
static void TestR01SingleInputSingleOutput(void)
{
  OnnxTestEngine engine;
  AlquimiaState state;

  /* Set up ONNX engine */
  OnnxRequireSetupEngine("deterministic/identity_double.json", false, &engine);

  ONNX_TEST_REQUIRE(engine.sizes.num_primary == 1);

  OnnxAllocateState(&engine, &state);
  state.total_mobile.data[0] = 4.25;

  OnnxRunInference(&engine, &state);
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.total_mobile.data[0], 4.25, 1.0e-12));

  FreeAlquimiaState(&state);
  OnnxRequireShutdownEngine(&engine);
}

/**
 * @brief Verifies multiple mapped inputs can feed one output tensor value.
 *
 * | R02 | Multiple inputs, single output | Tensor names and element positions route correctly |
 */
static void TestR02MultipleInputsSingleOutput(void)
{
  OnnxTestEngine engine;
  AlquimiaState state;

  OnnxRequireSetupEngine("deterministic/add_two_inputs.json", false, &engine);
  ONNX_TEST_REQUIRE(engine.sizes.num_primary == 3);

  /* Set up input tensor */
  OnnxAllocateState(&engine, &state);
  state.total_mobile.data[0] = 2.5;
  state.total_mobile.data[1] = -1.0;
  state.total_mobile.data[2] = 99.0;

  OnnxRunInference(&engine, &state);

  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.total_mobile.data[0], 2.5, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.total_mobile.data[1], -1.0, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.total_mobile.data[2], 1.5, 1.0e-12));

  FreeAlquimiaState(&state);
  OnnxRequireShutdownEngine(&engine);
}

/**
 * @brief Verifies one input tensor value can populate multiple outputs.
 *
 * | R03 | Single input, multiple outputs | Every output mapping is applied |
 */
static void TestR03SingleInputMultipleOutputs(void)
{
  OnnxTestEngine engine;
  AlquimiaState state;

  OnnxRequireSetupEngine("deterministic/single_input_multiple_outputs.json",
                         false, &engine);
  ONNX_TEST_REQUIRE(engine.sizes.num_primary == 4);
  ONNX_TEST_REQUIRE(engine.sizes.num_sorbed == 2);

  OnnxAllocateState(&engine, &state);
  state.total_mobile.data[0] = 3.0;
  state.total_mobile.data[1] = -5.0;

  OnnxRunInference(&engine, &state);

  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.total_mobile.data[2], 3.0, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.total_mobile.data[3], -5.0, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.total_immobile.data[0], 13.0, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.total_immobile.data[1], 15.0, 1.0e-12));

  FreeAlquimiaState(&state);
  OnnxRequireShutdownEngine(&engine);
}

/**
 * @brief Verifies multiple input and output tensors keep their boundaries.
 *
 * | R04 | Multiple inputs, multiple outputs | Tensor boundaries and mappings remain correct |
 */
static void TestR04MultipleInputsMultipleOutputs(void)
{
  OnnxTestEngine engine;
  AlquimiaState state;

  OnnxRequireSetupEngine("deterministic/multiple_inputs_outputs.json", false,
                         &engine);
  OnnxAllocateState(&engine, &state);

  state.porosity = 2.0;
  state.total_mobile.data[0] = 5.0;
  OnnxRunInference(&engine, &state);

  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.temperature, 7.0, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.gas_concentration.data[0], 19.0, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.porosity, 2.0, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.total_mobile.data[0], 5.0, 1.0e-12));

  FreeAlquimiaState(&state);
  OnnxRequireShutdownEngine(&engine);
}

/**
 * @brief Verifies mixed scalar and vector AlquimiaState mappings.
 *
 * | R05 | Mixed scalar and vector mappings | Values reach the correct state fields |
 */
static void TestR05MixedScalarVectorMappings(void)
{
  OnnxTestEngine engine;
  AlquimiaState state;

  OnnxRequireSetupEngine("deterministic/mixed_scalar_vector.json", false,
                         &engine);
  ONNX_TEST_REQUIRE(engine.sizes.num_primary == 2);
  ONNX_TEST_REQUIRE(engine.sizes.num_minerals == 2);
  ONNX_TEST_REQUIRE(engine.sizes.num_surface_sites == 1);
  ONNX_TEST_REQUIRE(engine.sizes.num_gases == 1);

  OnnxAllocateState(&engine, &state);
  state.porosity = 0.35;
  state.total_mobile.data[1] = 8.0;
  state.gas_concentration.data[0] = -2.0;
  OnnxRunInference(&engine, &state);

  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.temperature, 0.35, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.mineral_volume_fraction.data[1], 8.0, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.surface_site_density.data[0], -2.0, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.porosity, 0.35, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.total_mobile.data[1], 8.0, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.gas_concentration.data[0], -2.0, 1.0e-12));

  FreeAlquimiaState(&state);
  OnnxRequireShutdownEngine(&engine);
}

/**
 * @brief Verifies outputs across all state categories avoid cross-overwrites.
 *
 * | R06 | Outputs update different state categories | No cross-category overwrite |
 */
static void TestR06AllStateCategories(void)
{
  OnnxTestEngine engine;
  AlquimiaState state;

  OnnxRequireSetupEngine("deterministic/all_state_categories.json", false,
                         &engine);
  ONNX_TEST_REQUIRE(engine.sizes.num_primary == 1);
  ONNX_TEST_REQUIRE(engine.sizes.num_sorbed == 2);
  ONNX_TEST_REQUIRE(engine.sizes.num_minerals == 2);

  OnnxAllocateState(&engine, &state);

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

  OnnxRunInference(&engine, &state);

  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.water_density, 101.0, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.porosity, 102.0, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.temperature, 103.0, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.aqueous_pressure, 104.0, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.total_mobile.data[0], 105.0, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.total_immobile.data[0], -1.0, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.total_immobile.data[1], 106.0, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.mineral_volume_fraction.data[0], 107.0, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.mineral_volume_fraction.data[1], -2.0, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.mineral_specific_surface_area.data[0], -3.0, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.mineral_specific_surface_area.data[1], 108.0, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.surface_site_density.data[0], 109.0, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.cation_exchange_capacity.data[0], 110.0, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.gas_concentration.data[0], 111.0, 1.0e-12));

  FreeAlquimiaState(&state);
  OnnxRequireShutdownEngine(&engine);
}

/**
 * @brief Verifies rank-0 scalar tensor inputs and outputs.
 *
 * | R07 | Multiple rank-0 scalar inputs and outputs | Preserve scalar tensor rank during inference |
 */
static void TestR07MultipleScalarInputsOutputs(void)
{
  OnnxTestEngine engine;
  AlquimiaState state;

  OnnxRequireSetupEngine("deterministic/multiple_scalar_inputs_outputs.json",
                         false, &engine);

  ONNX_TEST_REQUIRE(engine.sizes.num_primary == 1);

  OnnxAllocateState(&engine, &state);
  state.total_mobile.data[0] = 2.5;
  state.porosity = -1.0;

  OnnxRunInference(&engine, &state);

  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.water_density, 1.5, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.aqueous_pressure, 3.5, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.total_mobile.data[0], 2.5, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.porosity, -1.0, 1.0e-12));

  FreeAlquimiaState(&state);
  OnnxRequireShutdownEngine(&engine);
}

/**
 * @brief Verifies paired mobile/immobile conservation behavior.
 *
 * | R08 | One or both phase outputs change paired components | Conserve only one-sided outputs |
 */
static void TestR08MobileImmobileConservation(void)
{
  OnnxTestEngine engine;
  AlquimiaState state;

  OnnxRequireSetupEngine("deterministic/mobile_immobile_conservation.json",
                         false, &engine);

  ONNX_TEST_REQUIRE(engine.sizes.num_primary == 2);
  ONNX_TEST_REQUIRE(engine.sizes.num_sorbed == 2);

  OnnxAllocateState(&engine, &state);

  state.total_mobile.data[0] = 3.0;
  state.total_mobile.data[1] = 4.0;
  state.total_immobile.data[0] = 17.0;
  state.total_immobile.data[1] = 6.0;

  OnnxRunInference(&engine, &state);

  /* shifted[0] changes mobile component 0 from 3 to 13. Its paired
  ** immobile value must decrease by 10 to preserve the total of 20. */
  ONNX_TEST_REQUIRE(
      OnnxCloseEnough(state.total_mobile.data[0], 13.0, 1.0e-12));
  ONNX_TEST_REQUIRE(
      OnnxCloseEnough(state.total_immobile.data[0], 7.0, 1.0e-12));
  /* copy[1] and shifted[1] explicitly output both phases of component 1,
  ** so both model values remain authoritative without conservation. */
  ONNX_TEST_REQUIRE(
      OnnxCloseEnough(state.total_mobile.data[1], 4.0, 1.0e-12));
  ONNX_TEST_REQUIRE(
      OnnxCloseEnough(state.total_immobile.data[1], 24.0, 1.0e-12));

  FreeAlquimiaState(&state);
  OnnxRequireShutdownEngine(&engine);
}

/* ---------- Special Cases ---------- */

/**
 * @brief Verifies repeated inference does not retain stale buffer values.
 *
 * | R09 | Repeated inference calls | Reusable buffers retain no incorrect prior values |
 */
static void TestR09RepeatedInference(void)
{
  OnnxTestEngine engine;
  AlquimiaState state;

  OnnxRequireSetupEngine("deterministic/affine_double.json", false, &engine);
  OnnxAllocateState(&engine, &state);
  state.total_mobile.data[0] = 1.0;
  OnnxRunInference(&engine, &state);

  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.total_mobile.data[0], 5.0, 1.0e-12));

  state.total_mobile.data[0] = 4.0;
  OnnxRunInference(&engine, &state);

  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.total_mobile.data[0], 11.0, 1.0e-12));

  state.total_mobile.data[0] = -3.0;
  OnnxRunInference(&engine, &state);

  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.total_mobile.data[0], -3.0, 1.0e-12));

  FreeAlquimiaState(&state);
  OnnxRequireShutdownEngine(&engine);
}

/**
 * @brief Verifies independent ONNX engines do not share runtime state.
 *
 * | R10 | Two independent engine instances | No state or buffer sharing between instances |
 */
static void TestR10IndependentInstances(void)
{
  OnnxTestEngine first;
  OnnxTestEngine second;
  AlquimiaState first_state;
  AlquimiaState second_state;

  OnnxRequireSetupEngine("deterministic/affine_double.json", false, &first);
  OnnxRequireSetupEngine("deterministic/affine_double.json", false, &second);
  OnnxAllocateState(&first, &first_state);
  OnnxAllocateState(&second, &second_state);
  first_state.total_mobile.data[0] = 2.0;
  second_state.total_mobile.data[0] = 10.0;
  OnnxRunInference(&first, &first_state);
  OnnxRunInference(&second, &second_state);

  ONNX_TEST_REQUIRE(OnnxCloseEnough(first_state.total_mobile.data[0], 7.0, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(second_state.total_mobile.data[0], 23.0, 1.0e-12));

  first_state.total_mobile.data[0] = -1.0;

  OnnxRunInference(&first, &first_state);

  ONNX_TEST_REQUIRE(OnnxCloseEnough(first_state.total_mobile.data[0], 1.0, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(second_state.total_mobile.data[0], 23.0, 1.0e-12));

  FreeAlquimiaState(&first_state);
  FreeAlquimiaState(&second_state);
  OnnxRequireShutdownEngine(&first);
  OnnxRequireShutdownEngine(&second);
}

/* ---------- Error Cases ---------- */

/**
 * @brief Verifies undersized output vectors fail before out-of-bounds writes.
 *
 * | E01 | Output vector is missing or undersized | Integrity error; no out-of-bounds write |
 */
static void TestE01UndersizedOutputVector(void)
{
  OnnxTestEngine engine;
  AlquimiaState state;

  OnnxRequireSetupEngine("deterministic/add_two_inputs.json", false, &engine);
  OnnxAllocateState(&engine, &state);
  state.total_mobile.data[0] = 4.0;
  state.total_mobile.data[1] = 6.0;
  state.total_mobile.data[2] = 12345.0;
  /* Undersize */
  state.total_mobile.size = 2;

  {
    AlquimiaProperties properties = {0};
    AlquimiaAuxiliaryData auxiliary_data = {0};
    engine.interface.ReactionStepOperatorSplit(
        &engine.engine_state, 1.0, &properties, &state, &auxiliary_data, 0,
        &engine.status);
  }

  ONNX_TEST_REQUIRE(engine.status.error == kAlquimiaErrorEngineIntegrity);
  ONNX_TEST_REQUIRE(strstr(engine.status.message,
                         "Out-of-bounds total_mobile write") != NULL);
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.total_mobile.data[2], 12345.0, 1.0e-12));

  state.total_mobile.size = 3;
  FreeAlquimiaState(&state);
  OnnxRequireShutdownEngine(&engine);
}

/**
 * @brief Verifies ONNX Runtime inference failures become engine errors.
 *
 * | E02 | ONNX Runtime inference fails | Translated status; engine remains safe to shut down |
 */
static void TestE02RuntimeInferenceFailure(void)
{
  OnnxTestEngine engine;
  AlquimiaState state;
  AlquimiaProperties properties = {0};
  AlquimiaAuxiliaryData auxiliary_data = {0};

  OnnxRequireSetupEngine("deterministic/runtime_failure.json", false, &engine);
  OnnxAllocateState(&engine, &state);
  state.total_mobile.data[0] = 1.0;
  state.total_mobile.data[1] = 2.0;

  engine.interface.ReactionStepOperatorSplit(
      &engine.engine_state, 1.0, &properties, &state, &auxiliary_data, 0,
      &engine.status);

  ONNX_TEST_REQUIRE(engine.status.error == kAlquimiaErrorEngineIntegrity);
  ONNX_TEST_REQUIRE(strstr(engine.status.message, "ONNX Runtime Error") != NULL);
  ONNX_TEST_REQUIRE(engine.engine_state != NULL);

  FreeAlquimiaState(&state);
  OnnxRequireShutdownEngine(&engine);
}

/* ---------- Model Family Cases ---------- */

/**
 * @brief Verifies baseline linear graph compatibility.
 *
 * | F01 | Linear or identity graph | Baseline tensor and operator compatibility |
 */
static void TestF01LinearAffine(void)
{
  OnnxTestEngine engine;
  AlquimiaState state;

  OnnxRequireSetupEngine("model_families/linear_affine.json", false, &engine);
  OnnxAllocateState(&engine, &state);
  state.total_mobile.data[0] = 3.0;
  OnnxRunInference(&engine, &state);

  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.total_mobile.data[0], 10.0, 1.0e-12));

  FreeAlquimiaState(&state);
  OnnxRequireShutdownEngine(&engine);
}

/**
 * @brief Verifies ai.onnx.ml SVR regression compatibility.
 *
 * | F02 | SVR | `ai.onnx.ml` regression compatibility |
 */
static void TestF02Svr(void)
{
  OnnxTestEngine engine;
  AlquimiaState state;

  OnnxRequireSetupEngine("model_families/svr_linear.json", false, &engine);
  OnnxAllocateState(&engine, &state);
  state.total_mobile.data[0] = 3.0;
  OnnxRunInference(&engine, &state);

  ONNX_TEST_REQUIRE(isfinite(state.total_mobile.data[0]));

  FreeAlquimiaState(&state);
  OnnxRequireShutdownEngine(&engine);
}

/**
 * @brief Verifies a small dense neural-network graph.
 *
 * | F03 | Small neural network | Standard dense activation graph compatibility |
 */
static void TestF03SmallNeuralNetwork(void)
{
  OnnxTestEngine engine;
  AlquimiaState state;

  OnnxRequireSetupEngine("model_families/small_neural_network.json", false,
                         &engine);
  ONNX_TEST_REQUIRE(engine.sizes.num_primary == 3);

  OnnxAllocateState(&engine, &state);
  state.total_mobile.data[0] = 3.0;
  state.total_mobile.data[1] = 4.0;
  OnnxRunInference(&engine, &state);

  /* output = ReLU(input * weights + bias), with input [3, 4],
  ** weights [2, -1], and bias 1. */
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.total_mobile.data[2], 3.0, 1.0e-12));

  FreeAlquimiaState(&state);
  OnnxRequireShutdownEngine(&engine);
}

/**
 * @brief Verifies ai.onnx.ml tree ensemble operator compatibility.
 *
 * | F04 | Tree ensemble | `ai.onnx.ml` tree operator compatibility |
 */
static void TestF04TreeEnsemble(void)
{
  OnnxTestEngine engine;
  AlquimiaState state;

  OnnxRequireSetupEngine("model_families/tree_ensemble.json", false, &engine);
  OnnxAllocateState(&engine, &state);
  state.total_mobile.data[0] = -100.0;
  OnnxRunInference(&engine, &state);

  ONNX_TEST_REQUIRE(
      OnnxCloseEnough(state.total_mobile.data[1], 7.0, 1.0e-12));

  /* It has only one node, and that node is a leaf with value 7.0. */
  FreeAlquimiaState(&state);
  OnnxRequireShutdownEngine(&engine);
}

/**
 * @brief Verifies multi-target model output routing.
 *
 * | F05 | Multi-target model | Compatibility with multiple outputs |
 */
static void TestF05MultiTarget(void)
{
  OnnxTestEngine engine;
  AlquimiaState state;

  OnnxRequireSetupEngine("model_families/multi_target.json", false, &engine);
  OnnxAllocateState(&engine, &state);
  state.total_mobile.data[0] = 2.0;
  state.total_mobile.data[1] = 3.0;
  OnnxRunInference(&engine, &state);

  /* For input [2, 3], target_sum is 5 and target_affine is 6. */
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.total_immobile.data[0], 5.0, 1.0e-12));
  ONNX_TEST_REQUIRE(OnnxCloseEnough(state.gas_concentration.data[0], 6.0, 1.0e-12));

  FreeAlquimiaState(&state);
  OnnxRequireShutdownEngine(&engine);
}

/**
 * @brief Verifies shared ALSURF model shapes preserve H/Zn output routing.
 *
 * | F06 | Shared ALSURF models | Every supported tensor shape preserves H/Zn output routing and lifecycle |
 */
static void TestF06AlsurfModels(void)
{
  static const double nn_h = 1.2306658377131264e-4;
  static const double nn_zn = 1.47994968124545e-7;
  static const double rf_h = 1.271885656770001e-4;
  static const double rf_zn = 3.9111416932674e-10;
  static const OnnxAlsurfModelCase cases[] = {
      {"F06-NN-1D", "alsurf_nn/zn_h_regressor_integrated_1D.json",
       nn_h, nn_zn},
      {"F06-NN-batch1",
       "alsurf_nn/zn_h_regressor_integrated_batch1.json", nn_h, nn_zn},
      {"F06-NN-dynamic",
       "alsurf_nn/zn_h_regressor_integrated_dyn_batch.json", nn_h, nn_zn},
      {"F06-RF-batch1", "alsurf_rf/alsurf_9_batch1.json", rf_h, rf_zn},
      {"F06-RF-dynamic", "alsurf_rf/alsurf_9_dynamic_batch.json", rf_h,
       rf_zn},
      {"F06-RF-vector", "alsurf_rf/alsurf_9_feature_vector.json", rf_h,
       rf_zn},
      {"F06-RF-6", "alsurf_rf/alsurf_6.json", rf_h, rf_zn},
      {"F06-RF-scalar", "alsurf_rf/alsurf_9_scalar.json", rf_h, rf_zn},
      {"F06-RF-mixed-input-ranks", "alsurf_rf/alsurf_9_mixed_inputs.json",
       rf_h, rf_zn},
  };
  size_t i;

  for (i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
  {
    const OnnxAlsurfModelCase *test_case = &cases[i];
    OnnxTestEngine engine;
    AlquimiaState state;

    printf("Running shared ALSURF model case %s.\n", test_case->test_id);

    OnnxRequireSharedModelEngine(test_case->config_path, true, &engine);
    ONNX_TEST_REQUIRE(engine.sizes.num_primary >= 2);
    ONNX_TEST_REQUIRE(engine.sizes.num_sorbed == 2);

    OnnxAllocateState(&engine, &state);
    OnnxApplyNamedCondition(&engine, "initial", &state);
    OnnxRunInference(&engine, &state);

    OnnxCheckAlsurfPrediction(test_case->test_id, "H(immobile)",
                              state.total_immobile.data[0],
                              test_case->expected_h);
    OnnxCheckAlsurfPrediction(test_case->test_id, "Zn(immobile)",
                              state.total_immobile.data[1],
                              test_case->expected_zn);

    FreeAlquimiaState(&state);
    OnnxRequireShutdownEngine(&engine);
  }
}

/* ---------- Integration Cases ---------- */

/**
 * @brief Verifies the full ALSURF setup, metadata, condition, inference, and shutdown lifecycle.
 *
 * | I01 | Full ALSURF lifecycle | Setup, metadata, condition, inference, and shutdown all succeed |
 */
static void TestI01AlsurfLifecycle(void)
{
  static const char *const features[] = {"H", "Zn"};
  static const double inputs[] = {1.0e-5, 1.0e-7};
  static const double outputs[] = {
      1.2306658377131264e-4, 1.47994968124545e-7};
  AlquimiaAuxiliaryData aux_data = {0};
  AlquimiaGeochemicalCondition condition;
  OnnxTestEngine engine;
  AlquimiaProblemMetaData meta_data;
  AlquimiaProperties properties = {0};
  AlquimiaState state = {0};
  int i;

  printf("Running successful ALSURF ONNX lifecycle.\n");
  if (!OnnxSetupEngineAtPath(ONNX_TEST_ALSURF_NAMED_CONFIG, false, &engine))
  {
    fprintf(stderr, "I01 setup failed: %s\n", engine.status.message);
  }

  ONNX_TEST_REQUIRE(engine.status.error == kAlquimiaNoError);
  ONNX_TEST_REQUIRE(engine.engine_state != NULL);
  ONNX_TEST_REQUIRE(engine.sizes.num_primary == 2);
  ONNX_TEST_REQUIRE(engine.sizes.num_sorbed == 2);
  ONNX_TEST_REQUIRE(engine.sizes.num_minerals == 0);
  ONNX_TEST_REQUIRE(engine.sizes.num_surface_sites == 0);
  ONNX_TEST_REQUIRE(engine.sizes.num_ion_exchange_sites == 0);
  ONNX_TEST_REQUIRE(engine.sizes.num_gases == 0);
  ONNX_TEST_REQUIRE(engine.functionality.operator_splitting);
  ONNX_TEST_REQUIRE(!engine.functionality.thread_safe);

  AllocateAlquimiaProblemMetaData(&engine.sizes, &meta_data);
  engine.interface.GetProblemMetaData(
      &engine.engine_state, &meta_data, &engine.status);

  ONNX_TEST_REQUIRE(engine.status.error == kAlquimiaNoError);
  for (i = 0; i < 2; ++i)
  {
    ONNX_TEST_REQUIRE(strcmp(meta_data.primary_names.data[i], features[i]) == 0);
  }

  OnnxAllocateState(&engine, &state);
  AllocateAlquimiaGeochemicalCondition(8, 2, 0, &condition);
  strcpy(condition.name, "initial");
  for (i = 0; i < 2; ++i)
  {
    OnnxInitializeConstraint(&condition, i, features[i], inputs[i]);
  }

  engine.interface.ProcessCondition(
      &engine.engine_state, &condition, &properties, &state, &aux_data,
      &engine.status);

  ONNX_TEST_REQUIRE(engine.status.error == kAlquimiaNoError);
  for (i = 0; i < 2; ++i)
  {
    ONNX_TEST_REQUIRE(state.total_mobile.data[i] == inputs[i]);
  }

  engine.interface.ReactionStepOperatorSplit(
      &engine.engine_state, 1.0, &properties, &state, &aux_data, 1,
      &engine.status);
  if (engine.status.error != kAlquimiaNoError)
  {
    fprintf(stderr, "I01 inference failed: %s\n", engine.status.message);
  }

  ONNX_TEST_REQUIRE(engine.status.error == kAlquimiaNoError);
  for (i = 0; i < 2; ++i)
  {
    ONNX_TEST_REQUIRE(isfinite(state.total_immobile.data[i]));
    ONNX_TEST_REQUIRE(
        OnnxCloseEnough(state.total_immobile.data[i], outputs[i], 1.0e-15));
  }

  FreeAlquimiaGeochemicalCondition(&condition);
  FreeAlquimiaState(&state);
  FreeAlquimiaProblemMetaData(&meta_data);
  ONNX_TEST_REQUIRE(OnnxShutdownEngine(&engine));
}

/* ---------- Runners ---------- */

/**
 * @brief Runs routing cases that must report controlled errors.
 */
static void RunRoutingErrorTests(void)
{
  TestE01UndersizedOutputVector();
  TestE02RuntimeInferenceFailure();
  printf("ONNX routing error cases E01-E02 passed.\n");
}

/**
 * @brief Runs standard and special ONNX inference-routing cases.
 */
static void RunRoutingTests(void)
{
  TestR01SingleInputSingleOutput();
  TestR02MultipleInputsSingleOutput();
  TestR03SingleInputMultipleOutputs();
  TestR04MultipleInputsMultipleOutputs();
  TestR05MixedScalarVectorMappings();
  TestR06AllStateCategories();
  TestR07MultipleScalarInputsOutputs();
  TestR08MobileImmobileConservation();
  RunRoutingErrorTests();
  TestR09RepeatedInference();
  TestR10IndependentInstances();
  printf("ONNX routing cases R01-R10 passed.\n");
}

/*
** | ID | Family | Purpose |
** |---|---|---|
** | F01 | Linear or identity graph | Baseline tensor and operator compatibility |
** | F02 | SVR | `ai.onnx.ml` regression compatibility |
** | F03 | Small neural network | Standard dense activation graph compatibility |
** | F04 | Tree ensemble | `ai.onnx.ml` tree operator compatibility |
** | F05 | Multi-target model | Compatibility with multiple outputs |
** | F06 | Shared ALSURF models | H/Zn output routing across all supported tensor shapes |
** | I01 | ALSURF lifecycle | Setup, metadata, condition, inference, and shutdown |
*/
/**
 * @brief Runs ONNX model-family compatibility cases.
 */
static void RunModelFamilyTests(void)
{
  TestF01LinearAffine();
  TestF02Svr();
  TestF03SmallNeuralNetwork();
  TestF04TreeEnsemble();
  TestF05MultiTarget();
  TestF06AlsurfModels();
  TestI01AlsurfLifecycle();
  printf("ONNX model-family cases F01-F06 and I01 passed.\n");
}

#endif

/**
 * @brief Runs ONNX routing and model-family test groups.
 */
int main(int argc, char **argv)
{
#if ALQUIMIA_HAVE_ONNX
  if (argc == 1)
  {
    RunRoutingTests();
    RunModelFamilyTests();
  }
  else if (argc == 2 && strcmp(argv[1], "routing") == 0)
  {
    RunRoutingTests();
  }
  else if (argc == 2 && strcmp(argv[1], "model-family") == 0)
  {
    RunModelFamilyTests();
  }
  else
  {
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
