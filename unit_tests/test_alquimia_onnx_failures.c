/* -*-  mode: c; c-default-style: "google"; indent-tabs-mode: nil -*- 
** The goal of this file is to verify the robustness, error handling,
** and lifecycle management of the Alquimia ONNX interface.
**
** Specifically, it covers:
**   1. Setup failure cases (invalid JSON configs, missing models, 
**      unsupported tensor types, bad graph dimensions).
**   2. Condition processing guard checks (NULL pointers, mismatched 
**      vector sizes, mixed scalar/vector mapping).
**   3. Engine lifecycle operations (null-pointer shutdowns, repeated 
**      creation/destruction loops, and leak checking).
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "alquimia/alquimia_constants.h"
#include "alquimia/alquimia_interface.h"
#include "alquimia/alquimia_memory.h"

#if ALQUIMIA_HAVE_ONNX

/* Record the overall failure times */
static int num_failures = 0;

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
** 
*/
#define CHECK_CASE(test_id, expression)                                      \
  do {                                                                        \
    if (!(expression)) {                                                      \
      fprintf(stderr, "%s failed at %s:%d: %s\n",                            \
              test_id, __FILE__, __LINE__, #expression);                      \
      ++num_failures;                                                         \
    }                                                                         \
  } while (0)

typedef struct {
  /* Test id 
  ** S for Setup
  ** C for ProcessCondition
  ** L for Lifecycle
  */
  const char *test_id;
  /* Stands for the relative path
  ** to a JSON config or ONNX model under onnx_test_cases.
  */
  const char *relative_path;
  const char *message_fragment;
} SetupFailureCase;

static void SetupAbsolutePath(const char *relative_path, char *path, size_t size) {
  /* Get the absolute path */
  int result = snprintf(path, size, "%s/onnx_test_cases/%s",
                        CMAKE_CURRENT_SOURCE_DIR, relative_path);
  CHECK_CASE("relative-path", result >= 0 && (size_t)result < size);
}

/* Set up the model path */
static void ModelPath(const char *filename, char *path, size_t size) {
  int result = snprintf(path, size, "%s/../models/%s",
                        CMAKE_CURRENT_SOURCE_DIR, filename);
  /* Expect: result */                        
  CHECK_CASE("model-path", result >= 0 && (size_t)result < size);
}

/* Create the Interface for test_id */
static int CreateOnnxInterface(
    AlquimiaInterface *interface,
    AlquimiaEngineStatus *status,
    const char *test_id) {
  AllocateAlquimiaEngineStatus(status);
  /* Fail status-> error != Null */
  CreateAlquimiaInterface("ONNX", interface, status);
  /* If success, status->error = kAlquimiaNoError 
  ** This will not print out anything
  */
  CHECK_CASE(test_id, status->error == kAlquimiaNoError);
  return status->error == kAlquimiaNoError;
}

static void CheckSetupFailure(const SetupFailureCase *test_case) {
  AlquimiaEngineFunctionality functionality = {0};
  AlquimiaEngineStatus status = {0};
  AlquimiaInterface interface = {0};
  AlquimiaSizes sizes = {0};
  char path[2048];
  void *onnx_engine_state = NULL;
  /* Create Alquimia interface */
  if (!CreateOnnxInterface(&interface, &status, test_case->test_id)) {
    FreeAlquimiaEngineStatus(&status);
    return;
  }
  /* Set up the absolute path for JSON file */
  SetupAbsolutePath(test_case->relative_path, path, sizeof(path));
  interface.Setup(path, false, &onnx_engine_state, &sizes, &functionality, &status);
  /* Test case fail here */
  /* Expect: status.error != kAlquimiaNoError */
  /* This will not print out anything */
  CHECK_CASE(test_case->test_id, status.error != kAlquimiaNoError);
  /* engine.state == Nulll */
  /* This will not print out anything */
  CHECK_CASE(test_case->test_id, onnx_engine_state == NULL);
  if (test_case->message_fragment != NULL) {
    /* Expect: strstr != NULL */
    /* This will not print out anything */
    CHECK_CASE(test_case->test_id,
               strstr(status.message, test_case->message_fragment) != NULL);
  }

  interface.Shutdown(&onnx_engine_state, &status);
  /* | L03 | Setup failure is followed by shutdown | No double free | */
  /* Expect: status.error == Errors */
  CHECK_CASE("L03", status.error == kAlquimiaErrorInvalidEngine);
  /* Expect: onnx_engine_state == NULL */
  CHECK_CASE("L03", onnx_engine_state == NULL);
  FreeAlquimiaEngineStatus(&status);
}
/* | S03 | Raw ONNX file is passed as the setup input | Parse error; engine state remains `NULL` | */
static void CheckRawModelRejected(void) {
  AlquimiaEngineFunctionality functionality = {0};
  AlquimiaEngineStatus status = {0};
  AlquimiaInterface interface = {0};
  AlquimiaSizes sizes = {0};
  char path[2048];
  void *onnx_engine_state = NULL;

  /* Create Alquimia Interface */
  if (!CreateOnnxInterface(&interface, &status, "S03")) {
    FreeAlquimiaEngineStatus(&status);
    return;
  }
  /* Get the model absolute path */
  /* path = model absolute path */
  ModelPath("lsurf_model_2_float_64.onnx", path, sizeof(path));
  /* Setup expects a JSON file */
  interface.Setup(path, false, &onnx_engine_state, &sizes, &functionality, &status);
  /* Expect: status.error == kAlquimiaErrorEngineIntegrity */
  CHECK_CASE("S03", status.error == kAlquimiaErrorEngineIntegrity);
  /* Expect: strstr != NULL */
  CHECK_CASE("S03", strstr(status.message, "strict JSON") != NULL);
  /* Expect: onnx_engine_state == NULL */
  CHECK_CASE("S03", onnx_engine_state == NULL);
  FreeAlquimiaEngineStatus(&status);
}

static int SetupOnnxEngine(
    const char *test_id,
    const char *relative_path,
    AlquimiaInterface *interface,
    AlquimiaEngineStatus *status,
    AlquimiaSizes *sizes,
    /* Pass the pointer-to-pointer 
    ** It will convert to (OnnxEngineState**)
    */
    void **onnx_engine_state) {
  AlquimiaEngineFunctionality functionality = {0};
  char path[2048];

  /* Create ONNX interface */
  if (!CreateOnnxInterface(interface, status, test_id)) {
    return 0;
  }
  /* Set up absolute path for the JSON test file */
  SetupAbsolutePath(relative_path, path, sizeof(path));
  /* Set up the interface */
  interface->Setup(path, false, onnx_engine_state, sizes, &functionality, status);
  /* If success, Expect: status->error == kAlquimiaError */
  CHECK_CASE(test_id, status->error == kAlquimiaNoError);
  /* Expect: engine state != NULL */
  CHECK_CASE(test_id, *onnx_engine_state != NULL);
  return status->error == kAlquimiaNoError && *onnx_engine_state != NULL;
}

static void InitializeConstraint(
    AlquimiaGeochemicalCondition *condition,
    int index,
    const char *name,
    double value) {
  AlquimiaAqueousConstraint *constraint =
      &condition->aqueous_constraints.data[index];
  AllocateAlquimiaAqueousConstraint(constraint);
  snprintf(constraint->primary_species_name, kAlquimiaMaxStringLength,
           "%s", name);
  snprintf(constraint->constraint_type, kAlquimiaMaxStringLength,
           "%s", "total");
  constraint->value = value;
}

/* | ID | Scenario | Expected result or policy gate |
** |---|---|---|
** | C01 | `condition == NULL` | No-op success |
** | C02 | Condition has no aqueous constraints | No-op success |
** | C05 | State pointer is `NULL` | Integrity error |
** | C06 | Mapped state-vector storage is `NULL` | Integrity error |
** | C07 | Runtime state vector is smaller than the derived setup size | Integrity error |
** | C09 | Scalar and vector destinations are mixed | All matching values are assigned correctly |
*/
static void CheckConditionGuards(void) {
  AlquimiaAuxiliaryData aux_data = {0};
  AlquimiaEngineStatus status = {0};
  AlquimiaGeochemicalCondition condition = {0};
  AlquimiaGeochemicalCondition empty_condition = {0};
  AlquimiaInterface interface = {0};
  AlquimiaProperties properties = {0};
  AlquimiaSizes sizes = {0};
  AlquimiaState state = {0};
  double *saved_data;
  int saved_size;
  void *onnx_engine_state = NULL;

  /* Check if the ONNX engine setup is success */
  if (!SetupOnnxEngine("condition-setup", "deterministic/identity_double.json",
                   &interface, &status, &sizes, &onnx_engine_state)) {
    FreeAlquimiaEngineStatus(&status);
    return;
  }
  /* | C01 | `condition == NULL` | No operation(No-op) success | */
  AllocateAlquimiaState(&sizes, &state);
  state.total_mobile.data[0] = 17.0;

  interface.ProcessCondition(&onnx_engine_state, NULL, &properties, &state,
                             &aux_data, &status);
  /* If success, Expect: status.error == kAlquimiaNoError*/
  CHECK_CASE("C01", status.error == kAlquimiaNoError);
  /* Expect: total_mobile.data[0] */
  CHECK_CASE("C01", state.total_mobile.data[0] == 17.0);

  /* | C02 | Condition has no aqueous constraints | No-op success | */
  AllocateAlquimiaGeochemicalCondition(5, 0, 0, &empty_condition);
  interface.ProcessCondition(&onnx_engine_state, &empty_condition, &properties,
                             &state, &aux_data, &status);
  /* Expect: status.error == kAlquimiaNoError */                             
  CHECK_CASE("C02", status.error == kAlquimiaNoError);
  /* Expect: state.total_mobile.data[0] == 17.0 */
  CHECK_CASE("C02", state.total_mobile.data[0] == 17.0);
  FreeAlquimiaGeochemicalCondition(&empty_condition);

  /* | C05 | State pointer is `NULL` | Integrity error | */
  interface.ProcessCondition(&onnx_engine_state, NULL, &properties, NULL,
                             &aux_data, &status);
  /* Expect: status.error == kAlquimiaErrorEngineIntegrity */                             
  CHECK_CASE("C05", status.error == kAlquimiaErrorEngineIntegrity);

  /* | C06 | Mapped state-vector storage is `NULL` | Integrity error | */
  AllocateAlquimiaGeochemicalCondition(5, 1, 0, &condition);
  InitializeConstraint(&condition, 0, "identity_input", 23.0);

  /* Point to the data array */
  saved_data = state.total_mobile.data;
  state.total_mobile.data = NULL;
  interface.ProcessCondition(&onnx_engine_state, &condition, &properties, &state,
                             &aux_data, &status);
  /* Expect: status.error == kAlquimiaErrorEngineIntegrity */                             
  CHECK_CASE("C06", status.error == kAlquimiaErrorEngineIntegrity);
  state.total_mobile.data = saved_data;

  /* | C07 | Runtime state vector is smaller than the derived setup size | Integrity error | */
  saved_size = state.total_mobile.size;
  state.total_mobile.size = 0;
  interface.ProcessCondition(&onnx_engine_state, &condition, &properties, &state,
                             &aux_data, &status);
  /* Expect: status.error == kAlquimiaErrorEngineIntegrity */                             
  CHECK_CASE("C07", status.error == kAlquimiaErrorEngineIntegrity);
  state.total_mobile.size = saved_size;

  FreeAlquimiaGeochemicalCondition(&condition);
  FreeAlquimiaState(&state);
  interface.Shutdown(&onnx_engine_state, &status);
  /* Expect: status.error == kAlquimiaNoError */
  CHECK_CASE("condition-shutdown", status.error == kAlquimiaNoError);
  /* Expect: onnx_engine_state == NULL */
  CHECK_CASE("condition-shutdown", onnx_engine_state == NULL);
  FreeAlquimiaEngineStatus(&status);
}

/* | C09 | Scalar and vector destinations are mixed | All matching values are assigned correctly | */
static void CheckMixedCondition(void) {
  AlquimiaAuxiliaryData aux_data = {0};
  AlquimiaEngineStatus status = {0};
  AlquimiaGeochemicalCondition condition = {0};
  AlquimiaInterface interface = {0};
  AlquimiaProperties properties = {0};
  AlquimiaSizes sizes = {0};
  AlquimiaState state = {0};
  void *onnx_engine_state = NULL;

  /* Setup ONNX Engine */
  if (!SetupOnnxEngine("C09", "deterministic/mixed_scalar_vector.json",
                   &interface, &status, &sizes, &onnx_engine_state)) {
    FreeAlquimiaEngineStatus(&status);
    return;
  }
  AllocateAlquimiaState(&sizes, &state);
  AllocateAlquimiaGeochemicalCondition(5, 3, 0, &condition);
  InitializeConstraint(&condition, 0, "porosity_feature", 0.31);
  InitializeConstraint(&condition, 1, "mobile_feature", 4.25);
  InitializeConstraint(&condition, 2, "gas_feature", 8.5);

  interface.ProcessCondition(&onnx_engine_state, &condition, &properties, &state,
                             &aux_data, &status);
  /* Expect: status.error == kAlquimiaNoError */ 
  /* porosity is a scalar */                            
  CHECK_CASE("C09", status.error == kAlquimiaNoError);
  /* Expect: state.porosity == 0.31 */
  CHECK_CASE("C09", state.porosity == 0.31);
  /* Expect: state.total_mobile.data != NULL */
  CHECK_CASE("C09", state.total_mobile.data != NULL);
  /* Expect: state.total_mobile.size > 1 */
  CHECK_CASE("C09", state.total_mobile.size > 1);
  if (state.total_mobile.data != NULL && state.total_mobile.size > 1) {
    /* Expect: state.total_mobile.data[1] == 4.25 */
    CHECK_CASE("C09", state.total_mobile.data[1] == 4.25);
  }
  /* Expect: state.gas_concentration.data != NULL */
  CHECK_CASE("C09", state.gas_concentration.data != NULL);
  /* Expect: state.gas_concentration.size > 0 */
  CHECK_CASE("C09", state.gas_concentration.size > 0);
  if (state.gas_concentration.data != NULL &&
      state.gas_concentration.size > 0) {
    /* Expect: state.gas_concentration.data[0] == 8.5 */
    CHECK_CASE("C09", state.gas_concentration.data[0] == 8.5);
  }

  FreeAlquimiaGeochemicalCondition(&condition);
  FreeAlquimiaState(&state);
  interface.Shutdown(&onnx_engine_state, &status);
  CHECK_CASE("C09", status.error == kAlquimiaNoError);
  CHECK_CASE("C09", onnx_engine_state == NULL);
  FreeAlquimiaEngineStatus(&status);
}
 
/* | L01 | Shutdown follows successful setup | Engine state becomes `NULL` |
** | L02 | Shutdown receives a null engine pointer | Defined error; no crash |
** | L03 | Setup failure is followed by shutdown | No double free |
** | L04 | Repeated create/setup/run/shutdown loop | No leak under sanitizer or Valgrind |
** | L05 | config parse fails after partial allocation | config strings, arrays, and cJSON tree are released |
*/
static void CheckNullShutdown(void) {
  AlquimiaEngineStatus status = {0};
  AlquimiaInterface interface = {0};
  void *onnx_engine_state = NULL;

  /* | L02 | Shutdown receives a null engine pointer | Defined error; no crash | */
  if (!CreateOnnxInterface(&interface, &status, "L02")) {
    FreeAlquimiaEngineStatus(&status);
    return;
  }
  interface.Shutdown(NULL, &status);
  /* Expect: status.error == kAlquimiaErrorInvalidEngine*/
  CHECK_CASE("L02", status.error == kAlquimiaErrorInvalidEngine);
  interface.Shutdown(&onnx_engine_state, &status);
  /* Expect: status.error == kAlquimiaErrorInvalidEngine */
  CHECK_CASE("L02", status.error == kAlquimiaErrorInvalidEngine);
  /* Expect: onnx_engine_state == NULL */
  CHECK_CASE("L02", onnx_engine_state == NULL);
  FreeAlquimiaEngineStatus(&status);
}

/* | L01 | Shutdown follows successful setup | Engine state becomes `NULL` | 
** | L04 | Repeated create/setup/run/shutdown loop | No leak under sanitizer or Valgrind |
*/
static void CheckRepeatedLifecycle(void) {
  int iteration;

  for (iteration = 0; iteration < 16; ++iteration) {
    AlquimiaAuxiliaryData aux_data = {0};
    AlquimiaEngineStatus status = {0};
    AlquimiaInterface interface = {0};
    AlquimiaProperties properties = {0};
    AlquimiaSizes sizes = {0};
    AlquimiaState state = {0};
    void *onnx_engine_state = NULL;

    /* Set up the ONNX engine */
    if (!SetupOnnxEngine("L04", "deterministic/identity_double.json",
                     &interface, &status, &sizes, &onnx_engine_state)) {
      FreeAlquimiaEngineStatus(&status);
      return;
    }
    AllocateAlquimiaState(&sizes, &state);
    state.total_mobile.data[0] = (double)iteration + 0.5;
    interface.ReactionStepOperatorSplit(
        &onnx_engine_state, 1.0, &properties, &state, &aux_data,
        iteration, &status);
    /* Expect: status.error == kAlquimiaNoError */
    CHECK_CASE("L04", status.error == kAlquimiaNoError);
    /* Expect: state.total_mobile.data[0] == (double)iteration + 0.5 */
    CHECK_CASE("L04", state.total_mobile.data[0] ==
                          (double)iteration + 0.5);
    FreeAlquimiaState(&state);
    interface.Shutdown(&onnx_engine_state, &status);
    /* Expect: status.error == kAlquimiaNoError */
    CHECK_CASE("L01/L04", status.error == kAlquimiaNoError);
    /* Expect: onnx_engine_state == NULL */
    CHECK_CASE("L01/L04", onnx_engine_state == NULL);
    FreeAlquimiaEngineStatus(&status);
  }
}

static void RunSetupFailureCases(void) {
  /* | S01 | config path does not exist | Setup error; engine state remains `NULL` |
  ** | S02 | config exists but is malformed or has trailing content | Parse error; no crash or leak |
  ** | S03 | Raw ONNX file is passed as the setup input | Parse error; engine state remains `NULL` |
  ** | S04 | config model path does not exist | Setup error naming the resolved path |
  ** | S05 | Model file exists but is not ONNX | Setup error; no crash or leak |
  ** | S06 | ONNX graph is invalid | Setup error; no published engine state |
  ** | S07 | Model has zero inputs or zero outputs | Setup error |
  ** | S08 | Input element type is unsupported | Precise setup error |
  ** | S09 | Output element type is unsupported | Precise setup error |
  ** | S10 | Dimensions are dynamic, overflowing, or otherwise unsafe | Setup rejects rather than mis-sizing buffers |
  ** | S11 | ONNX Runtime returns an error during setup | Status is translated and acquired resources are released |
  */
  /* Test message segment is the substring of the error message in the onnx_alquimia_interface.c */
  static const SetupFailureCase cases[] = {
    {"S01", "invalid_models/config_does_not_exist.json",
     "Unable to read ONNX config"},
    {"S02-malformed", "invalid_models/malformed_config.json",
     "not valid strict JSON"},
    {"S02-trailing", "invalid_models/trailing_content.json",
     "not valid strict JSON"},
    {"S04", "invalid_models/missing_model.json",
     "ONNX model file not found"},
    {"S05", "invalid_models/not_onnx.json",
     "ONNX Runtime Error"},
    {"S06", "invalid_models/invalid_graph.json",
     "ONNX Runtime Error"},
    {"S07-zero-inputs", "invalid_models/zero_inputs.json",
     "Model has 0 inputs or outputs"},
    {"S07-zero-outputs", "invalid_models/zero_outputs.json",
     "ONNX Runtime Error"},
    {"S08", "invalid_models/unsupported_input_type.json",
     "ONNX input tensor 'input' must have double elements"},
    {"S09", "invalid_models/unsupported_output_type.json",
     "ONNX output tensor 'output' must have double elements"},
    {"S10", "invalid_models/dynamic_dimension.json",
     "unsupported dynamic extent"},
    {"S11", "invalid_models/runtime_setup_error.json",
     "ONNX Runtime Error"}
  };
  size_t i;

  for (i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
    CheckSetupFailure(&cases[i]);
  }
  /* Check S03 */
  CheckRawModelRejected();
}

#endif

int main(void) {
  /* The test file couples with the error message in the related c file 
  ** Change the error message string could potentially fail the test
  */
#if ALQUIMIA_HAVE_ONNX
  /* Except all the setup test case and L03 */
  RunSetupFailureCases();

  /* Check C01, 02, 05, 06, 07 */
  CheckConditionGuards();
  
  /* Check C09 */
  CheckMixedCondition();

  /* Check L02 */
  CheckNullShutdown();

  /* Check L01, L04 */
  CheckRepeatedLifecycle();

  if (num_failures != 0) {
    fprintf(stderr, "ONNX failure/lifecycle tests had %d failure(s).\n",
            num_failures);
    return EXIT_FAILURE;
  }
  printf("ONNX failure/lifecycle tests passed.\n");
#else
  printf("ONNX not enabled. Skipping ONNX failure/lifecycle tests.\n");
#endif
  return EXIT_SUCCESS;
}
