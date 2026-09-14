# EX8 ONNX Models

This file documents only the EX8 model folders used by the ONNX tests: `ex8_nn/` and `ex8_rf/`.

The JSON files are Alquimia ONNX configs. The `.onnx` files are the actual ONNX Runtime graphs. The configs map `AlquimiaState` values into model tensors and map model outputs back into `AlquimiaState`.

## `ex8_nn/`

Neural-network EX8 models for H/Zn [molarity] sorption.
Input features: **H (total)**, **Zn (total)** 
Output features: **H (immobile)**, **Zn (immobile)**

### Neural-Network ONNX Files

| Model | Inputs | Outputs | Graph Contents | Notes |
|---|---|---|---|---|
| `ex8_nn_1d.onnx` | `chemical_input_raw: double[2]` | `sorbed_output_raw: double[2]` | Scaler math plus `MatMul`, `Add`, `Relu` | Raw H/Zn input and raw sorbed output; used by unit tests. |
| `ex8_nn_batch1.onnx` | `chemical_input_raw: double[1,2]` | `sorbed_output_raw: double[1,2]` | Scaler math plus `Gemm`, `Relu` | Fixed batch-1 raw interface. |
| `ex8_nn_dynamic_batch.onnx` | `chemical_input_raw: double[batch_size,2]` | `sorbed_output_raw: double[batch_size,2]` | Scaler math plus `Gemm`, `Relu` | Dynamic-batch raw interface. |

### Neural-Network Configs

| Config | Model | Mapping |
|---|---|---|
| `ex8_nn_1d.json` | `ex8_nn_1d.onnx` | Reads H/Zn from `total_mobile[2]` and `[3]`; writes reactive H/Zn to `total_immobile[0]` and `[1]`. |
| `ex8_nn_batch1.json` | `ex8_nn_batch1.onnx` | Batch = 1 version of the H/Zn total-mobile to total-immobile mapping. |
| `ex8_nn_dynamic_batch.json` | `ex8_nn_dynamic_batch.onnx` | Dynamic-batch version of the H/Zn total-mobile to total-immobile mapping. |

Each config includes an `initial` condition with H = `9.999999999999999e-06` and Zn = `1e-7`.
The inference results should be H(immobile) = `-1.230666e-04`, Zn(immobile) = `-1.479950e-07`.

## `ex8_rf/`

Random-forest EX8 models for H/Zn [molarity] sorption.

The ONNX exports preserve double-precision (`float64`) inputs and outputs for Alquimia, but internally cast to single-precision (`float32`) during `Scaler` and `TreeEnsemble` opset specifically to truncate floating-point differences.

### Random-Forest ONNX Files

| Model | Inputs | Outputs | Graph Contents | Notes |
|---|---|---|---|---|
| `ex8_rf_6_dynamic_batch.onnx` | `double_input: double[dynamic,6]` | `double_output: double[dynamic,2]` | Scaler math, casts, `TreeEnsemble` | Six-feature forest. |
| `ex8_rf_9_batch1.onnx` | `chemical_input_raw: double[1,9]` | `sorbed_output_raw: double[1,2]` | Scaler math, casts, `TreeEnsemble` | Fixed batch-1 nine-feature forest. |
| `ex8_rf_9_dynamic_batch.onnx` | `chemical_input_raw: double[batch_size,9]` | `sorbed_output_raw: double[batch_size,2]` | Scaler math, casts, `TreeEnsemble` | Dynamic-batch nine-feature forest. |
| `ex8_rf_9_1d.onnx` | `chemical_input_raw: double[9]` | `sorbed_output_raw: double[2]` | Reshape helpers, scaler math, casts, `TreeEnsemble` | One 1-D feature vector. |
| `ex8_rf_9_scalar.onnx` | Nine `rank-0` feature tensors | `SURF-H+: double[]`, `SURF-Zn++: double[]` | Scalar packing, scaler math, casts, `TreeEnsemble` | One scalar tensor per feature. |
| `ex8_rf_9_mixed_inputs.onnx` | `chemical_dynamic: double[batch_size,3]`, `chemical_vector: double[3]`, `chemical_batch1: double[1,2]`, `chemical_scalar: double[]` | `sorbed_output_raw: double[batch_size,2]` | Mixed-rank packing, scaler math, casts, `TreeEnsemble` | Tests mixed input-rank support. |

### Random-Forest Configs

| Config | Model | Mapping |
|---|---|---|
| `ex8_rf_6_dynamic_batch.json` | `ex8_rf_6_dynamic_batch.onnx` | Reads six aqueous species from `total_mobile[2]` through `[7]`; writes H/Zn to `total_immobile[0]` and `[1]`. |
| `ex8_rf_9_batch1.json` | `ex8_rf_9_batch1.onnx` | Reads nine aqueous species from `total_mobile[2]` through `[10]`; writes H/Zn immobile totals. |
| `ex8_rf_9_dynamic_batch.json` | `ex8_rf_9_dynamic_batch.onnx` | Same nine-feature mapping with dynamic-batch tensor shape. |
| `ex8_rf_9_1d.json` | `ex8_rf_9_1d.onnx` | Same nine-feature mapping with a 1-D feature-vector tensor. |
| `ex8_rf_9_scalar.json` | `ex8_rf_9_scalar.onnx` | Maps each feature to its own rank-0 tensor and reads scalar H/Zn outputs. |
| `ex8_rf_9_mixed_inputs.json` | `ex8_rf_9_mixed_inputs.onnx` | Splits nine features across dynamic, vector, batch-1, and scalar input tensors. |

The nine-feature configs use: `Zn(OH)2(aq)`, `Zn(OH)3-`, `Zn(OH)4--`, `ZnOH+`, `Zn++`, `Na+`, `NO3-`, `Fe++`, and `O2(aq)`.

All random-forest configs include an `initial` condition for the mapped aqueous species and write model outputs to:

| Output | Alquimia Destination | Unit |
|---|---|---|
| H sorbed amount | `total_immobile[0]` | moles/m^3 bulk|
| Zn sorbed amount | `total_immobile[1]` | moles/m^3 bulk|
