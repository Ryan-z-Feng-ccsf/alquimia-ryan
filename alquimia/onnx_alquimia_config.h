/* -*-  mode: c; c-default-style: "google"; indent-tabs-mode: nil -*- */

/* Private config types and helpers for parsing JSON mappings used by the
** ONNX Alquimia interface. config-owned storage is released after mappings
** are built or initialization fails. */

#ifndef ALQUIMIA_ONNX_ALQUIMIA_CONFIG_H_
#define ALQUIMIA_ONNX_ALQUIMIA_CONFIG_H_

#include <stdbool.h>
#include <stddef.h>

/* Maps one ONNX tensor element to an AlquimiaState field element.
 * Mobile, immobile, and total_molar tensor values are mol/L liquid water.
 * Other tensor values use the mapped field's native Alquimia units. */
typedef struct {
  /* Tensor names the model input or output tensor. */
  char *tensor;
  /* Tensor_element_index selects one flattened tensor element. */
  size_t tensor_element_index;
  /* Feature names the model input or output feature. */
  char *feature;
  /* Alquimia_state and alquimia_state_index identify its AlquimiaState location. */
  char *alquimia_state;
  int alquimia_state_index;
} OnnxAlquimiaMapping;

/* One condition assignment; nested feature objects are flattened into items. */
typedef struct {
  char *feature;
  /* NULL for scalar shorthand, otherwise the explicit state field. */
  char *alquimia_state;
  /* Native Alquimia units, also for scalar shorthand: total_mobile is mol/L
   * water and total_immobile is mol/m^3 bulk. Conditions do not use tensor units. */
  double value;
} OnnxAlquimiaConditionItem;

/* One condition selected by an exact .cfg initial_condition name. */
typedef struct {
  // Condition name in the .cfg file
  char *name;
  OnnxAlquimiaConditionItem *items;
  size_t num_items;
} OnnxAlquimiaCondition;

/* The input/output tensor would be >=1 */
typedef struct {
  char *model_path;
  /* Optional named initial conditions retained for ProcessCondition. */
  OnnxAlquimiaCondition *conditions;
  size_t num_conditions;
  /* Store the respective input value in the JSON file */
  OnnxAlquimiaMapping *inputs;
  /* Number of the input tensor */
  size_t num_inputs;
  /* Store the respective output value in the JSON file */
  OnnxAlquimiaMapping *outputs;
  /* Number of the output tensor */
  size_t num_outputs;
} OnnxAlquimiaConfig;

/* Parse the JSON file */
bool OnnxAlquimiaLoadConfig(
    /* JSON file path */
    const char *config_path,
    /* Populates the config. On failure, resources are freed internally. 
    ** This can be assigned directly to an OnnxEngine's engine->config;
    ** on success, the interface manages its lifetime and release. */
    OnnxAlquimiaConfig *onnx_config,
    /* Write the error message in the AlquimiaEngineStatus: status*/
    char *error_message,
    /* Specify the error message size */
    size_t error_message_size);

void OnnxAlquimiaFreeConfig(OnnxAlquimiaConfig *config);

#endif /* ALQUIMIA_ONNX_ALQUIMIA_CONFIG_H_ */
