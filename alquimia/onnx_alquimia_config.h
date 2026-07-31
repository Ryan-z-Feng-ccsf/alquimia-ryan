/* -*-  mode: c; c-default-style: "google"; indent-tabs-mode: nil -*- */

/* Private config types and helpers for parsing JSON mappings used by the
** ONNX Alquimia interface. config-owned storage is released after mappings
** are built or initialization fails. */

#ifndef ALQUIMIA_ONNX_ALQUIMIA_CONFIG_H_
#define ALQUIMIA_ONNX_ALQUIMIA_CONFIG_H_

#include <stdbool.h>
#include <stddef.h>

/* Maps one ONNX input tensor element to an AlquimiaState field element. */
typedef struct {
  /* Tensor name the input tensor */
  char *tensor;
  /* Tensor_element_index selects its element. */
  size_t tensor_element_index;
  /* Feature names the model input feature */
  char *feature;
  /* Alquimia_state andalquimia_state_index identify the corresponding AlquimiaState location. */
  char *alquimia_state;
  int alquimia_state_index;
} OnnxAlquimiaInputMapping;

/* Maps one ONNX output tensor element to an AlquimiaState field element. */
typedef struct {
  /* Tensor name the input tensor */
  char *tensor;
  /* Tensor_element_index selects its element. */
  size_t tensor_element_index;
  /* Alquimia_state andalquimia_state_index identify the corresponding AlquimiaState location. */
  char *alquimia_state;
  int alquimia_state_index;
} OnnxAlquimiaOutputMapping;

/* One named feature value supplied by a JSON initial condition. */
/* Item -> key: value */
typedef struct {
  char *feature;
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
  OnnxAlquimiaInputMapping *inputs;
  /* Number of the input tensor */
  size_t num_inputs;
  /* Store the respective output value in the JSON file */
  OnnxAlquimiaOutputMapping *outputs;
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
