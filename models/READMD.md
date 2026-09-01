# ALSURF ONNX Models

This file documents only the ALSURF model folders used by the ONNX tests: `alsurf_nn/` and `alsurf_rf/`.

The JSON files are Alquimia ONNX configs. The `.onnx` files are the actual ONNX Runtime graphs. The configs map `AlquimiaState` values into model tensors and map model outputs back into `AlquimiaState`.

## `alsurf_nn/`

Neural-network ALSURF models for H/Zn sorption.

| File | Purpose |
|---|---|
| `meta.json` | Records the raw model feature names: input: `H (total)`, `Zn (total)`, output: `H (immobile)`, `Zn (immobile)`. |
| `x_scalar.joblib` | Input scaler used before neural-network inference. |
| `y_scalar.joblib` | Output scaler used after neural-network inference. |
| `zn_h_regressor_base.pt` | Source PyTorch model checkpoint. |
| `zn_h_regressor.onnx.data` | External tensor data used by one exported ONNX graph. |

### Neural-Network ONNX Files

| Model | Inputs | Outputs | Graph Contents | Notes |
|---|---|---|---|---|
| `zn_h_regressor_1D.onnx` | `input_scaled: double[2]` | `output_scaled: double[2]` | `MatMul`, `Add`, `Relu` | Scaled 1-D model interface. |
| `zn_h_regressor_batch1.onnx` | `input_scaled: double[1,2]` | `output_scaled: double[1,2]` | `Gemm`, `Relu` | Scaled fixed-batch interface. |
| `zn_h_regressor_dropout.onnx` | `input_scaled: double[batch_size,2]` | `output_scaled: double[batch_size,2]` | `Gemm`, `Relu`, `Dropout` | Dynamic-batch export with dropout nodes present. |
| `zn_h_regressor_dyn_batch.onnx` | `input_scaled: double[batch_size,2]` | `output_scaled: double[batch_size,2]` | `Gemm`, `Relu` | Dynamic-batch scaled model. |
| `zn_h_regressor_integrated_1D.onnx` | `chemical_input_raw: double[2]` | `sorbed_output_raw: double[2]` | Scaler math plus `MatMul`, `Add`, `Relu` | Raw H/Zn input and raw sorbed output; used by unit tests. |
| `zn_h_regressor_integrated_batch1.onnx` | `chemical_input_raw: double[1,2]` | `sorbed_output_raw: double[1,2]` | Scaler math plus `Gemm`, `Relu` | Fixed batch-1 raw interface. |
| `zn_h_regressor_integrated_dyn_batch.onnx` | `chemical_input_raw: double[batch_size,2]` | `sorbed_output_raw: double[batch_size,2]` | Scaler math plus `Gemm`, `Relu` | Dynamic-batch raw interface. |

### Neural-Network Configs

| Config | Model | Mapping |
|---|---|---|
| `zn_h_regressor_integrated_1D.json` | `zn_h_regressor_integrated_1D.onnx` | Reads H/Zn from `total_mobile[2]` and `[3]`; writes reactive H/Zn to `total_immobile[0]` and `[1]`. |
| `zn_h_regressor_integrated_batch1.json` | `zn_h_regressor_integrated_batch1.onnx` | Batch = 1 version of the H/Zn total-mobile to total-immobile mapping. |
| `zn_h_regressor_integrated_dyn_batch.json` | `zn_h_regressor_integrated_dyn_batch.onnx` | Dynamic-batch version of the H/Zn total-mobile to total-immobile mapping. |

Each config includes an `initial` condition with H = `1e-5` and Zn = `1e-7`.
The inference results should be H(immobile) = `-1.230666e-04`, Zn(immobile) = `-1.479950e-07`.

## `alsurf_rf/`

Random-forest ALSURF models for H/Zn sorption.

| File | Purpose |
|---|---|
| `metadata.json` | Lists selected feature sets and random-forest hyperparameters for `alsurf_1` through `alsurf_10`. |
| `alsurf_*.joblib` | Source scikit-learn random-forest models. |
| `x_scalar.joblib` | Input scaler used before random-forest inference. |
| `y_scalar.joblib` | Output scaler used after random-forest inference. |

The ONNX exports preserve double-precision (`float64`) inputs and outputs for Alquimia, but internally cast to single-precision (`float32`) during `Scaler` and `TreeEnsemble` opset specifically to truncate floating-point differences.

### Random-Forest ONNX Files

| Model | Inputs | Outputs | Graph Contents | Notes |
|---|---|---|---|---|
| `alsurf_6.onnx` | `double_input: double[dynamic,6]` | `double_output: double[dynamic,2]` | Scaler math, casts, `TreeEnsemble` | Six-feature forest. |
| `alsurf_9_batch1.onnx` | `chemical_input_raw: double[1,9]` | `sorbed_output_raw: double[1,2]` | Scaler math, casts, `TreeEnsemble` | Fixed batch-1 nine-feature forest. |
| `alsurf_9_dynamic_batch.onnx` | `chemical_input_raw: double[batch_size,9]` | `sorbed_output_raw: double[batch_size,2]` | Scaler math, casts, `TreeEnsemble` | Dynamic-batch nine-feature forest. |
| `alsurf_9_feature_vector.onnx` | `chemical_input_raw: double[9]` | `sorbed_output_raw: double[2]` | Reshape helpers, scaler math, casts, `TreeEnsemble` | One 1-D feature vector. |
| `alsurf_9_scalar.onnx` | Nine `rank-0` feature tensors | `SURF-H+: double[]`, `SURF-Zn++: double[]` | Scalar packing, scaler math, casts, `TreeEnsemble` | One scalar tensor per feature. |
| `alsurf_9_mixed_inputs.onnx` | `chemical_dynamic: double[batch_size,3]`, `chemical_vector: double[3]`, `chemical_batch1: double[1,2]`, `chemical_scalar: double[]` | `sorbed_output_raw: double[batch_size,2]` | Mixed-rank packing, scaler math, casts, `TreeEnsemble` | Tests mixed input-rank support. |

### Random-Forest Configs

| Config | Model | Mapping |
|---|---|---|
| `alsurf_6.json` | `alsurf_6.onnx` | Reads six aqueous species from `total_mobile[2]` through `[7]`; writes H/Zn to `total_immobile[0]` and `[1]`. |
| `alsurf_9_batch1.json` | `alsurf_9_batch1.onnx` | Reads nine aqueous species from `total_mobile[2]` through `[10]`; writes H/Zn immobile totals. |
| `alsurf_9_dynamic_batch.json` | `alsurf_9_dynamic_batch.onnx` | Same nine-feature mapping with dynamic-batch tensor shape. |
| `alsurf_9_feature_vector.json` | `alsurf_9_feature_vector.onnx` | Same nine-feature mapping with a 1-D feature-vector tensor. |
| `alsurf_9_scalar.json` | `alsurf_9_scalar.onnx` | Maps each feature to its own rank-0 tensor and reads scalar H/Zn outputs. |
| `alsurf_9_mixed_inputs.json` | `alsurf_9_mixed_inputs.onnx` | Splits nine features across dynamic, vector, batch-1, and scalar input tensors. |

The nine-feature configs use: `Zn(OH)2(aq)`, `Zn(OH)3-`, `Zn(OH)4--`, `ZnOH+`, `Zn++`, `Na+`, `NO3-`, `Fe++`, and `O2(aq)`.

All random-forest configs include an `initial` condition for the mapped aqueous species and write model outputs to:

| Output | Alquimia Destination |
|---|---|
| H sorbed amount | `total_immobile[0]` |
| Zn sorbed amount | `total_immobile[1]` |
