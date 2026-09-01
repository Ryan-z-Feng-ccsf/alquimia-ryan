# ONNX Unit-Test Cases

This folder contains small ONNX graphs and JSON configs used only by the ONNX unit tests. The `.onnx` files are intentionally tiny mock models, so tests can verify routing, validation, and lifecycle behavior without depending on a large trained chemistry model.

Each JSON file maps ONNX tensor elements to `AlquimiaState` fields. The adapter loads the JSON config, resolves the referenced `.onnx` file, runs ONNX Runtime, and writes outputs back to Alquimia state storage.

## Folders

| Folder | Purpose |
|---|---|
| `configs/` | Config-only fixtures, for relative model path behavior. |
| `deterministic/` | Valid mock graphs with predictable math for routing tests. |
| `invalid_models/` | Broken configs or ONNX files used to test setup errors. |
| `model_families/` | Small valid graphs covering supported ONNX operator families. |

## Deterministic Mock Models

| Model | Inputs | Outputs | Graph Behavior | Used For |
|---|---|---|---|---|
| `add_two_inputs.onnx` | `left: double[1]`, `right: double[1]` | `sum: double[1]` | `sum = left + right`. | Multiple inputs feeding one output. |
| `affine_double.onnx` | `input: double[1]` | `output: double[1]` | `output = 2 * input + 3`. | Repeated inference and independent engine-instance checks. |
| `all_state_categories.onnx` | `input: double[11]` | `output: double[11]` | Identity over all mapped state categories. | Cross-category overwrite checks. |
| `identity_double.onnx` | `input: double[1]` | `output: double[1]` | Identity. | Basic one-input, one-output routing. |
| `mixed_scalar_vector.onnx` | `input: double[3]` | `output: double[3]` | Identity over a three-element vector. | Mixed scalar and vector `AlquimiaState` mappings. |
| `multiple_inputs_outputs.onnx` | `left: double[1]`, `right: double[1]` | `sum: double[1]`, `weighted: double[1]` | Produces two outputs from two scalar inputs. | Tensor-boundary routing. |
| `multiple_scalar_inputs_outputs.onnx` | `left: double[]`, `right: double[]` | `sum: double[]`, `difference: double[]` | Rank-0 scalar add/subtract. | Scalar tensor support. |
| `runtime_failure.onnx` | `input: double[2]` | `output: double[1]` | Invalid gather at runtime. | Controlled ONNX Runtime inference failure. |
| `single_input_multiple_outputs.onnx` | `input: double[2]` | `copy: double[2]`, `shifted: double[2]` | Copies input and emits shifted values. | Multiple outputs and mobile/immobile conservation. |

`named_condition.json` uses the shared ALSURF neural-network model from [**`../../models/alsurf_nn/`**](../../models/alsurf_nn/) and adds a named `initial` condition for condition-routing tests.

## Invalid Models

| Fixture | What Is Broken | Expected Test Coverage |
|---|---|---|
| `dynamic_dimension.onnx` | Tensor extent is unresolved. | Adapter rejects unsafe dynamic feature extents. |
| `invalid_graph.onnx` | Graph has invalid structure. | ONNX Runtime graph validation failure. |
| `malformed_config.json` | JSON syntax is incomplete. | Strict JSON parse failure. |
| `missing_model.json` | Referenced model file does not exist. | Missing model diagnostic. |
| `not_onnx.onnx` | File is not a valid ONNX protobuf. | ONNX Runtime model-load failure. |
| `runtime_setup_error.onnx` | Graph uses a fake operator. | ONNX Runtime setup error path. |
| `trailing_content.json` | Valid JSON followed by extra text. | Strict JSON trailing-content rejection. |
| `unsupported_input_type.onnx` | Input tensor is `float32`. | Adapter requires double input tensors. |
| `unsupported_output_type.onnx` | Output tensor is `float32`. | Adapter requires double output tensors. |
| `zero_inputs.onnx` | Graph publishes no inputs. | Adapter rejects zero-input models. |
| `zero_outputs.onnx` | Graph publishes no outputs. | Adapter rejects zero-output models. |

## Model-Family Fixtures

| Model | Inputs | Outputs | Operator Family | Used For |
|---|---|---|---|---|
| `linear_affine.onnx` | `input: double[1]` | `output: double[1]` | `Mul`, `Add` | Baseline arithmetic graph support. |
| `multi_target.onnx` | `input: double[1,2]` | `target_sum: double[1,1]`, `target_affine: double[1,1]` | `MatMul`, `Add` | Multiple output tensors. |
| `small_neural_network.onnx` | `input: double[1,2]` | `output: double[1,1]` | `MatMul`, `Add`, `Relu` | Dense neural-network graph support. |
| `svr_linear.onnx` | `input: double[1,1]` | `output: double[1,1]` | `ai.onnx.ml` `SVMRegressor` | SVR model-family support. |
| `tree_ensemble.onnx` | `input: double[1,1]` | `output: double[1,1]` | `ai.onnx.ml` `TreeEnsembleRegressor` | Tree model-family support. |

## Config Mapping Pattern

Every config has:

| Field | Meaning |
|---|---|
| `schema_version` | Config schema version expected by the adapter. |
| `model` | Relative or absolute path to the ONNX graph. |
| `conditions` | Optional named input values for `ProcessCondition`. |
| `inputs` | ONNX tensor elements read from `AlquimiaState`. |
| `outputs` | ONNX tensor elements written back to `AlquimiaState`. |

The mock models should stay small and deterministic. Larger trained models under [**`../../models/`**](../../models/), not here.
